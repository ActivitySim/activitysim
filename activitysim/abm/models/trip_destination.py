# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402
from builtins import range

import logging

import numpy as np
import pandas as pd

from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import pipeline
from activitysim.core import simulate
from activitysim.core import inject

from activitysim.core.tracing import print_elapsed_time

from activitysim.core.util import reindex
from activitysim.core.util import assign_in_place

from .util import expressions

from activitysim.core import assign

from activitysim.abm.tables.size_terms import tour_destination_size_terms

from activitysim.core.skim import DataFrameMatrix

from activitysim.core.interaction_sample_simulate import interaction_sample_simulate
from activitysim.core.interaction_sample import interaction_sample

from activitysim.abm.models.util.trip import cleanup_failed_trips


logger = logging.getLogger(__name__)

NO_DESTINATION = -1


def get_spec_for_purpose(model_settings, spec_name, purpose):

    omnibus_spec = simulate.read_model_spec(file_name=model_settings[spec_name])

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
    """

    Returns
    -------
    destination_sample: pandas.dataframe
        choices_df from interaction_sample with (up to) sample_size alts for each chooser row
        index (non unique) is trip_id from trips (duplicated for each alt)
        and columns dest_taz, prob, and pick_count

        dest_taz: int
            alt identifier (dest_taz) from alternatives[<alt_col_name>]
        prob: float
            the probability of the chosen alternative
        pick_count : int
            number of duplicate picks for chooser, alt
    """
    trace_label = tracing.extend_trace_label(trace_label, 'trip_destination_sample')

    spec = get_spec_for_purpose(model_settings, 'DESTINATION_SAMPLE_SPEC', primary_purpose)

    sample_size = model_settings["SAMPLE_SIZE"]
    alt_dest_col_name = model_settings["ALT_DEST"]

    logger.info("Running %s with %d trips", trace_label, trips.shape[0])

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


def compute_ood_logsums(
        choosers,
        logsum_settings,
        od_skims,
        locals_dict,
        chunk_size,
        trace_label):
    """
    Compute one (of two) out-of-direction logsums for destination alternatives

    Will either be trip_origin -> alt_dest or alt_dest -> primary_dest
    """

    locals_dict.update(od_skims)

    expressions.annotate_preprocessors(
        choosers, locals_dict, od_skims,
        logsum_settings,
        trace_label)

    nest_spec = config.get_logit_model_settings(logsum_settings)
    logsum_spec = simulate.read_model_spec(file_name=logsum_settings['SPEC'])

    logsums = simulate.simple_simulate_logsums(
        choosers,
        logsum_spec,
        nest_spec,
        skims=od_skims,
        locals_d=locals_dict,
        chunk_size=chunk_size,
        trace_label=trace_label)

    assert logsums.index.equals(choosers.index)

    # FIXME not strictly necessary, but would make trace files more legible?
    # logsums = logsums.replace(-np.inf, -999)

    return logsums


def compute_logsums(
        primary_purpose,
        trips,
        destination_sample,
        tours_merged,
        model_settings,
        skims,
        chunk_size, trace_hh_id,
        trace_label):
    """
    Calculate mode choice logsums using the same recipe as for trip_mode_choice, but do it twice
    for each alternative since we need out-of-direction logsum
    (i.e . origin to alt_dest, and alt_dest to half-tour destination)

    Returns
    -------
        adds od_logsum and dp_logsum columns to trips (in place)
    """
    trace_label = tracing.extend_trace_label(trace_label, 'compute_logsums')
    logger.info("Running %s with %d samples", trace_label, destination_sample.shape[0])

    # - trips_merged - merge trips and tours_merged
    trips_merged = pd.merge(
        trips,
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
                        how="left",
                        suffixes=('', '_r')).set_index('trip_id')
    assert choosers.index.equals(destination_sample.index)

    logsum_settings = config.read_model_settings(model_settings['LOGSUM_SETTINGS'])

    omnibus_coefficient_spec = \
        assign.read_constant_spec(config.config_file_path(logsum_settings['COEFFS']))

    coefficient_spec = omnibus_coefficient_spec[primary_purpose]

    constants = config.get_model_constants(logsum_settings)
    locals_dict = assign.evaluate_constants(coefficient_spec, constants=constants)
    locals_dict.update(constants)

    # - od_logsums
    od_skims = {
        'ORIGIN': model_settings['TRIP_ORIGIN'],
        'DESTINATION': model_settings['ALT_DEST'],
        "odt_skims": skims['odt_skims'],
        "od_skims": skims['od_skims'],
    }
    destination_sample['od_logsum'] = compute_ood_logsums(
        choosers,
        logsum_settings,
        od_skims,
        locals_dict,
        chunk_size,
        trace_label=tracing.extend_trace_label(trace_label, 'od'))

    # - dp_logsums
    dp_skims = {
        'ORIGIN': model_settings['ALT_DEST'],
        'DESTINATION': model_settings['PRIMARY_DEST'],
        "odt_skims": skims['dpt_skims'],
        "od_skims": skims['dp_skims'],
    }
    destination_sample['dp_logsum'] = compute_ood_logsums(
        choosers,
        logsum_settings,
        dp_skims,
        locals_dict,
        chunk_size,
        trace_label=tracing.extend_trace_label(trace_label, 'dp'))


def trip_destination_simulate(
        primary_purpose,
        trips,
        destination_sample,
        model_settings,
        size_term_matrix, skims,
        chunk_size, trace_hh_id,
        trace_label):
    """
    Chose destination from destination_sample (with od_logsum and dp_logsum columns added)


    Returns
    -------
    choices - pandas.Series
        destination alt chosen
    """
    trace_label = tracing.extend_trace_label(trace_label, 'trip_destination_simulate')

    spec = get_spec_for_purpose(model_settings, 'DESTINATION_SPEC', primary_purpose)

    alt_dest_col_name = model_settings["ALT_DEST"]

    logger.info("Running trip_destination_simulate with %d trips", len(trips))

    locals_dict = config.get_model_constants(model_settings).copy()
    locals_dict.update({
        'size_terms': size_term_matrix
    })
    locals_dict.update(skims)

    destinations = interaction_sample_simulate(
        choosers=trips,
        alternatives=destination_sample,
        spec=spec,
        choice_column=alt_dest_col_name,
        allow_zero_probs=True, zero_prob_choice_val=NO_DESTINATION,
        skims=skims,
        locals_d=locals_dict,
        chunk_size=chunk_size,
        trace_label=trace_label,
        trace_choice_name='trip_dest')

    # drop any failed zero_prob destinations
    if (destinations == NO_DESTINATION).any():
        # logger.debug("dropping %s failed destinations", destinations == NO_DESTINATION).sum()
        destinations = destinations[destinations != NO_DESTINATION]

    return destinations


def choose_trip_destination(
        primary_purpose,
        trips,
        alternatives,
        tours_merged,
        model_settings,
        size_term_matrix, skims,
        chunk_size, trace_hh_id,
        trace_label):

    logger.info("choose_trip_destination %s with %d trips", trace_label, trips.shape[0])

    t0 = print_elapsed_time()

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
        logger.warning("%s trip_destination_sample %s trips "
                       "without viable destination alternatives" %
                       (trace_label, dropped_trips.sum()))
        trips = trips[~dropped_trips]

    t0 = print_elapsed_time("%s.trip_destination_sample" % trace_label, t0)

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

    t0 = print_elapsed_time("%s.compute_logsums" % trace_label, t0)

    # - trip_destination_simulate
    destinations = trip_destination_simulate(
        primary_purpose=primary_purpose,
        trips=trips,
        destination_sample=destination_sample,
        model_settings=model_settings,
        size_term_matrix=size_term_matrix, skims=skims,
        chunk_size=chunk_size, trace_hh_id=trace_hh_id,
        trace_label=trace_label)

    dropped_trips = ~trips.index.isin(destinations.index)
    if dropped_trips.any():
        logger.warning("%s trip_destination_simulate %s trips "
                       "without viable destination alternatives" %
                       (trace_label, dropped_trips.sum()))

    t0 = print_elapsed_time("%s.trip_destination_simulate" % trace_label, t0)

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
        "odt_skims": skim_stack.wrap(left_key=o, right_key=d, skim_key='trip_period'),
        "dot_skims": skim_stack.wrap(left_key=d, right_key=o, skim_key='trip_period'),
        "dpt_skims": skim_stack.wrap(left_key=d, right_key=p, skim_key='trip_period'),
        "pdt_skims": skim_stack.wrap(left_key=p, right_key=d, skim_key='trip_period'),
        "od_skims": skim_dict.wrap(o, d),
        "dp_skims": skim_dict.wrap(d, p),
    }

    return skims


def run_trip_destination(
        trips,
        tours_merged,
        chunk_size, trace_hh_id,
        trace_label):
    """
    trip destination - main functionality separated from model step so it can be called iteratively

    Run the trip_destination model, assigning destinations for each (intermediate) trip
    (last trips already have a destination - either the tour primary destination or Home)

    Set trip destination and origin columns, and a boolean failed flag for any failed trips
    (destination for flagged failed trips will be set to -1)

    Parameters
    ----------
    trips
    tours_merged
    chunk_size
    trace_hh_id
    trace_label

    Returns
    -------

    """

    model_settings = config.read_model_settings('trip_destination.yaml')
    preprocessor_settings = model_settings.get('preprocessor', None)
    logsum_settings = config.read_model_settings(model_settings['LOGSUM_SETTINGS'])

    land_use = inject.get_table('land_use')
    size_terms = inject.get_injectable('size_terms')

    # - initialize trip origin and destination to those of half-tour
    # (we will sequentially adjust intermediate trips origin and destination as we choose them)
    tour_destination = reindex(tours_merged.destination, trips.tour_id).astype(int)
    tour_origin = reindex(tours_merged.origin, trips.tour_id).astype(int)
    trips['destination'] = np.where(trips.outbound, tour_destination, tour_origin)
    trips['origin'] = np.where(trips.outbound, tour_origin, tour_destination)
    trips['failed'] = False

    trips = trips.sort_index()
    trips['next_trip_id'] = np.roll(trips.index, -1)
    trips.next_trip_id = trips.next_trip_id.where(trips.trip_num < trips.trip_count, 0)

    # - filter tours_merged (AFTER copying destination and origin columns to trips)
    # tours_merged is used for logsums, we filter it here upfront to save space and time
    tours_merged_cols = logsum_settings['TOURS_MERGED_CHOOSER_COLUMNS']
    if 'REDUNDANT_TOURS_MERGED_CHOOSER_COLUMNS' in model_settings:
        redundant_cols = model_settings['REDUNDANT_TOURS_MERGED_CHOOSER_COLUMNS']
        tours_merged_cols = [c for c in tours_merged_cols if c not in redundant_cols]
    tours_merged = tours_merged[tours_merged_cols]

    # - skims
    skims = wrap_skims(model_settings)

    # - size_terms and alternatives
    alternatives = tour_destination_size_terms(land_use, size_terms, 'trip')

    # DataFrameMatrix alows us to treat dataframe as virtual a 2-D array, indexed by TAZ, purpose
    # e.g. size_terms.get(df.dest_taz, df.purpose)
    # returns a series of size_terms for each chooser's dest_taz and purpose with chooser index
    size_term_matrix = DataFrameMatrix(alternatives)

    # don't need size terms in alternatives, just TAZ index
    alternatives = alternatives.drop(alternatives.columns, axis=1)
    alternatives.index.name = model_settings['ALT_DEST']

    # - process intermediate trips in ascending trip_num order
    intermediate = trips.trip_num < trips.trip_count
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

            logger.info("Running %s with %d trips", nth_trace_label, nth_trips.shape[0])

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
                logger.warning("%s sidelining %s trips without viable destination alternatives" %
                               (nth_trace_label, failed_trip_ids.shape[0]))
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
        chunk_size, trace_hh_id):
    """
    Choose a destination for all 'intermediate' trips based on trip purpose.

    Final trips already have a destination (the primary tour destination for outbound trips,
    and home for inbound trips.)


    """
    trace_label = 'trip_destination'
    model_settings = config.read_model_settings('trip_destination.yaml')
    CLEANUP = model_settings.get('CLEANUP', True)

    trips_df = trips.to_frame()
    tours_merged_df = tours_merged.to_frame()

    logger.info("Running %s with %d trips", trace_label, trips_df.shape[0])

    trips_df = run_trip_destination(
        trips_df,
        tours_merged_df,
        chunk_size, trace_hh_id,
        trace_label)

    if trips_df.failed.any():
        logger.warning("%s %s failed trips", trace_label, trips_df.failed.sum())
        file_name = "%s_failed_trips" % trace_label
        logger.info("writing failed trips to %s", file_name)
        tracing.write_csv(trips_df[trips_df.failed], file_name=file_name, transpose=False)

    if CLEANUP:
        trips_df = cleanup_failed_trips(trips_df)
    elif trips_df.failed.any():
        logger.warning("%s keeping %s sidelined failed trips" %
                       (trace_label, trips_df.failed.sum()))

    pipeline.replace_table("trips", trips_df)

    print("trips_df\n", trips_df.shape)

    if trace_hh_id:
        tracing.trace_df(trips_df,
                         label=trace_label,
                         slicer='trip_id',
                         index_label='trip_id',
                         warn_if_empty=True)
