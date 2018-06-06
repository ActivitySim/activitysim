# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import numpy as np
import pandas as pd

from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import pipeline
from activitysim.core import simulate
from activitysim.core import inject

from activitysim.core.util import reindex
from activitysim.core.util import assign_in_place

from .util import logsums
from .util import expressions

from .util.tour_destination import tour_destination_size_terms
from activitysim.core.skim import DataFrameMatrix

from activitysim.core.interaction_sample_simulate import interaction_sample_simulate
from activitysim.core.interaction_sample import interaction_sample

from activitysim.abm.models.util.trip import cleanup_failed_trips


logger = logging.getLogger(__name__)


def get_spec_for_purpose(model_settings, spec_name, purpose):
    configs_dir = inject.get_injectable('configs_dir')
    omnibus_spec = simulate.read_model_spec(configs_dir, model_settings[spec_name])
    spec = omnibus_spec[[purpose]]

    # might as well ignore any spec rows with 0 utility
    spec = spec[spec.iloc[:, 0] != 0]
    assert spec.shape[0] > 0

    return spec


def trip_destination_sample(
        primary_purpose,
        trips,
        alternatives,
        model_settings,
        size_term_matrix, skims,
        chunk_size, trace_hh_id,
        trace_label):

    trace_label = tracing.extend_trace_label(trace_label, 'trip_destination_sample')

    spec = get_spec_for_purpose(model_settings, 'DESTINATION_SAMPLE_SPEC', primary_purpose)

    sample_size = model_settings["SAMPLE_SIZE"]
    alt_dest_col_name = model_settings["ALT_DEST"]

    logger.info("Running %s with %d trips" % (trace_label, trips.shape[0]))

    locals_dict = config.get_model_constants(model_settings).copy()
    locals_dict.update({
        'size_terms': size_term_matrix
    })
    locals_dict.update(skims)

    destination_sample = interaction_sample(
        choosers=trips,
        alternatives=alternatives,
        sample_size=sample_size,
        alt_col_name=alt_dest_col_name,
        allow_zero_probs=True,
        spec=spec,
        skims=skims,
        locals_d=locals_dict,
        chunk_size=chunk_size,
        trace_label=trace_label)

    return destination_sample


def compute_logsums(
        primary_purpose,
        trips,
        destination_sample,
        tours_merged,
        model_settings,
        skims,
        chunk_size, trace_hh_id,
        trace_label):

    trace_label = tracing.extend_trace_label(trace_label, 'compute_logsums')
    logger.info("Running %s with %d samples" % (trace_label, destination_sample.shape[0]))

    configs_dir = inject.get_injectable('configs_dir')
    logsum_settings = config.read_model_settings(configs_dir, model_settings['LOGSUM_SETTINGS'])
    preprocessor_settings = logsum_settings.get('preprocessor_settings', None)

    nest_spec = config.get_logit_model_settings(logsum_settings)
    locals_dict = config.get_model_constants(logsum_settings)

    logsum_spec = logsums.get_logsum_spec(
        logsum_settings,
        selector='trip',
        segment=primary_purpose,
        configs_dir=configs_dir,
        want_tracing=trace_hh_id)

    # - trips_merged - merge trips and tours_merged
    trip_chooser_columns = model_settings['LOGSUM_TRIP_COLUMNS'] + ['tour_id']
    trips_merged = pd.merge(
        trips[trip_chooser_columns],
        tours_merged,
        left_on='tour_id',
        right_index=True,
        how="left")
    assert trips_merged.index.equals(trips.index)

    # - choosers - merge destination_sample and trips_merged
    # re/set index because pandas merge does not preserve left index if it has duplicate values!
    choosers = pd.merge(destination_sample,
                        trips_merged.reset_index(),
                        left_index=True,
                        right_on='trip_id',
                        how="left").set_index('trip_id')
    assert choosers.index.equals(destination_sample.index)

    # - od_logsums
    locals_dict['dest_col_name'] = model_settings['ALT_DEST']
    od_skims = {
        "odt_skims": skims['odt_skims'],
        "dot_skims": skims['dot_skims'],
        "od_skims": skims['od_skims'],
    }
    locals_dict.update(od_skims)

    if preprocessor_settings:

        simulate.add_skims(choosers, od_skims)

        expressions.assign_columns(
            df=choosers,
            model_settings=preprocessor_settings,
            locals_dict=locals_dict,
            trace_label=tracing.extend_trace_label(trace_label, 'od'))

    od_logsums = simulate.simple_simulate_logsums(
        choosers,
        logsum_spec,
        nest_spec,
        skims=od_skims,
        locals_d=locals_dict,
        chunk_size=chunk_size,
        trace_label=tracing.extend_trace_label(trace_label, 'od'))

    assert od_logsums.index.equals(choosers.index)

    # - dp_logsums
    locals_dict['dest_col_name'] = model_settings['PRIMARY_DEST']
    od_skims = {
        "odt_skims": skims['dpt_skims'],
        "dot_skims": skims['pdt_skims'],
        "od_skims": skims['dp_skims'],
    }
    locals_dict.update(od_skims)

    if preprocessor_settings:
        simulate.add_skims(choosers, od_skims)

        expressions.assign_columns(
            df=choosers,
            model_settings=preprocessor_settings,
            locals_dict=locals_dict,
            trace_label=tracing.extend_trace_label(trace_label, 'dp'))

    dp_logsums = simulate.simple_simulate_logsums(
        choosers,
        logsum_spec,
        nest_spec,
        skims=od_skims,
        locals_d=locals_dict,
        chunk_size=chunk_size,
        trace_label=tracing.extend_trace_label(trace_label, 'dp'))

    assert dp_logsums.index.equals(choosers.index)

    destination_sample['od_logsum'] = od_logsums
    destination_sample['dp_logsum'] = dp_logsums


def trip_destination_simulate(
        primary_purpose,
        trips,
        destination_sample,
        model_settings,
        size_term_matrix, skims,
        chunk_size, trace_hh_id,
        trace_label):

    trace_label = tracing.extend_trace_label(trace_label, 'trip_destination_simulate')

    spec = get_spec_for_purpose(model_settings, 'DESTINATION_SPEC', primary_purpose)

    alt_dest_col_name = model_settings["ALT_DEST"]

    logger.info("Running trip_destination_simulate with %d trips" % len(trips))

    locals_dict = config.get_model_constants(model_settings).copy()
    locals_dict.update({
        'size_terms': size_term_matrix
    })
    locals_dict.update(skims)

    choices = interaction_sample_simulate(
        choosers=trips,
        alternatives=destination_sample,
        spec=spec,
        choice_column=alt_dest_col_name,
        skims=skims,
        locals_d=locals_dict,
        chunk_size=chunk_size,
        trace_label=trace_label,
        trace_choice_name='trip_dest')

    return choices


def choose_trip_destination(
        primary_purpose,
        trips,
        alternatives,
        tours_merged,
        model_settings,
        size_term_matrix, skims,
        chunk_size, trace_hh_id,
        trace_label):

    logger.info("choose_trip_destination %s with %d trips" % (trace_label, trips.shape[0]))

    # - trip_destination_sample
    destination_sample = trip_destination_sample(
        primary_purpose=primary_purpose,
        trips=trips,
        alternatives=alternatives,
        model_settings=model_settings,
        size_term_matrix=size_term_matrix, skims=skims,
        chunk_size=chunk_size, trace_hh_id=trace_hh_id,
        trace_label=trace_label)

    dropped_trips = ~trips.index.isin(destination_sample.index.unique())
    if dropped_trips.any():
        logger.warn("%s suppressing %s trips without viable destination alternatives"
                    % (trace_label, dropped_trips.sum()))
        trips = trips[~dropped_trips]

    if trips.empty:
        return pd.Series(index=trips.index)

    # - compute logsums
    compute_logsums(
        primary_purpose=primary_purpose,
        trips=trips,
        destination_sample=destination_sample,
        tours_merged=tours_merged,
        model_settings=model_settings,
        skims=skims,
        chunk_size=chunk_size, trace_hh_id=trace_hh_id,
        trace_label=trace_label)

    # - trip_destination_simulate
    destinations = trip_destination_simulate(
        primary_purpose=primary_purpose,
        trips=trips,
        destination_sample=destination_sample,
        model_settings=model_settings,
        size_term_matrix=size_term_matrix, skims=skims,
        chunk_size=chunk_size, trace_hh_id=trace_hh_id,
        trace_label=trace_label)

    return destinations


def wrap_skims(model_settings):
    """
    wrap skims of trip destination using origin, dest column names from model settings.
    Various of these are used by destination_sample, compute_logsums, and destination_simulate
    so we create them all here with canonical names.

    Note that compute_logsums aliases their names so it can use the same equations to compute
    logsums from origin to alt_dest, and from alt_dest to primarly destination

    odt_skims - SkimStackWrapper: trip origin, trip alt_dest, time_of_day
    dot_skims - SkimStackWrapper: trip alt_dest, trip origin, time_of_day
    dpt_skims - SkimStackWrapper: trip alt_dest, trip primary_dest, time_of_day
    pdt_skims - SkimStackWrapper: trip primary_dest,trip alt_dest, time_of_day
    od_skims - SkimDictWrapper: trip origin, trip alt_dest
    dp_skims - SkimDictWrapper: trip alt_dest, trip primary_dest

    Parameters
    ----------
    model_settings

    Returns
    -------
        dict containing skims, keyed by canonical names relative to tour orientation
    """

    skim_dict = inject.get_injectable('skim_dict')
    skim_stack = inject.get_injectable('skim_stack')

    o = model_settings['TRIP_ORIGIN']
    d = model_settings['ALT_DEST']
    p = model_settings['PRIMARY_DEST']

    skims = {
        "odt_skims": skim_stack.wrap(left_key=o, right_key=d, skim_key='time_period'),
        "dot_skims": skim_stack.wrap(left_key=d, right_key=o, skim_key='time_period'),
        "dpt_skims": skim_stack.wrap(left_key=d, right_key=p, skim_key='time_period'),
        "pdt_skims": skim_stack.wrap(left_key=p, right_key=d, skim_key='time_period'),
        "od_skims": skim_dict.wrap(o, d),
        "dp_skims": skim_dict.wrap(d, p),
    }
    return skims


def run_trip_destination(
        trips,
        tours_merged,
        configs_dir, chunk_size, trace_hh_id,
        trace_label):

    model_settings = config.read_model_settings(configs_dir, 'trip_destination.yaml')
    preprocessor_settings = model_settings.get('preprocessor_settings', None)
    logsum_settings = config.read_model_settings(configs_dir, model_settings['LOGSUM_SETTINGS'])

    land_use = inject.get_table('land_use')
    size_terms = inject.get_table('size_terms')

    # - initialize trip origin and destination to those of half-tour
    # (we will sequentially adjust intermediate trips origin and destination as we choose them)
    tour_destination = reindex(tours_merged.destination, trips.tour_id).astype(int)
    tour_origin = reindex(tours_merged.origin, trips.tour_id).astype(int)
    trips['destination'] = np.where(trips.outbound, tour_destination, tour_origin)
    trips['origin'] = np.where(trips.outbound, tour_origin, tour_destination)
    trips['failed'] = False

    trips = trips.sort_index()
    trips['next_trip_id'] = np.roll(trips.index, -1)
    trips.next_trip_id = trips.next_trip_id.where(~trips['last'], 0)

    # - filter tours_merged (AFTER copying destination and origin columns to trips)
    # tours_merged is used for logsums, we filter it here upfront to save space and time
    tours_merged = logsums.filter_chooser_columns(tours_merged, logsum_settings, model_settings)

    # - skims
    skims = wrap_skims(model_settings)

    # - size_terms and alternatives
    alternatives = tour_destination_size_terms(land_use, size_terms, 'trip')
    size_term_matrix = DataFrameMatrix(alternatives)
    # don't need size terms in alternatives, just TAZ index
    alternatives = alternatives.drop(alternatives.columns, axis=1)
    alternatives.index.name = model_settings['ALT_DEST']

    # - process intermediate trips in ascending trip_num order
    intermediate = ~trips['last']
    if intermediate.any():

        first_trip_num = trips[intermediate].trip_num.min()
        last_trip_num = trips[intermediate].trip_num.max()

        # iterate over trips in ascending trip_num order
        for trip_num in range(first_trip_num, last_trip_num + 1):

            nth_trips = trips[intermediate & (trips.trip_num == trip_num)]
            nth_trace_label = tracing.extend_trace_label(trace_label, 'trip_num_%s' % trip_num)

            # - annotate nth_trips
            if preprocessor_settings:
                expressions.assign_columns(
                    df=nth_trips,
                    model_settings=preprocessor_settings,
                    locals_dict=config.get_model_constants(model_settings),
                    trace_label=nth_trace_label)

            logger.info("Running %s with %d trips" % (nth_trace_label, nth_trips.shape[0]))

            # - choose destination for nth_trips, segmented by primary_purpose
            choices_list = []
            for primary_purpose, trips_segment in nth_trips.groupby('primary_purpose'):
                choices = choose_trip_destination(
                    primary_purpose,
                    trips_segment,
                    alternatives,
                    tours_merged,
                    model_settings,
                    size_term_matrix, skims,
                    chunk_size, trace_hh_id,
                    trace_label=tracing.extend_trace_label(nth_trace_label, primary_purpose))

                choices_list.append(choices)

            destinations = pd.concat(choices_list)

            failed_trip_ids = nth_trips.index.difference(destinations.index)
            if failed_trip_ids.any():
                logger.warn("%s sidelining %s trips without viable destination alternatives\n"
                            % (nth_trace_label, failed_trip_ids.shape[0]))
                next_trip_ids = nth_trips.next_trip_id.reindex(failed_trip_ids)
                trips.loc[failed_trip_ids, 'failed'] = True
                trips.loc[failed_trip_ids, 'destination'] = -1
                trips.loc[next_trip_ids, 'origin'] = trips.loc[failed_trip_ids].origin.values

            # - assign choices to these trips destinations and to next trips origin
            assign_in_place(trips, destinations.to_frame('destination'))
            destinations.index = nth_trips.next_trip_id.reindex(destinations.index)
            assign_in_place(trips, destinations.to_frame('origin'))

    del trips['next_trip_id']

    return trips


@inject.step()
def trip_destination(
        trips,
        tours_merged,
        configs_dir, chunk_size, trace_hh_id):

    trace_label = 'trip_destination'
    model_settings = config.read_model_settings(configs_dir, 'trip_destination.yaml')
    CLEANUP = model_settings.get('CLEANUP', True)

    trips_df = trips.to_frame()
    tours_merged_df = tours_merged.to_frame()

    trips_df = run_trip_destination(
        trips_df,
        tours_merged_df,
        configs_dir, chunk_size, trace_hh_id,
        trace_label)

    if trips_df.failed.any():
        logger.warn("%s %s failed trips" % (trace_label, trips_df.failed.sum()))
        file_name = "%s_failed_trips" % trace_label
        logger.info("writing failed trips to %s" % file_name)
        tracing.write_csv(trips_df[trips_df.failed], file_name=file_name)

    if CLEANUP:
        trips_df = cleanup_failed_trips(trips_df)
    elif trips_df.failed.any():
        logger.warn("%s keeping %s sidelined failed trips" % (trace_label, trips_df.failed.sum()))

    pipeline.replace_table("trips", trips_df)
