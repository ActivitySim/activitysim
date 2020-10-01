# ActivitySim
# See full license in LICENSE.txt.
import logging

import numpy as np
import pandas as pd

from activitysim.core import logit
from activitysim.core import config
from activitysim.core import inject
from activitysim.core import tracing
from activitysim.core import chunk
from activitysim.core import pipeline
from activitysim.core import expressions

logger = logging.getLogger(__name__)


PROBS_JOIN_COLUMNS = ['primary_purpose', 'outbound', 'person_type']


def trip_purpose_probs():
    f = config.config_file_path('trip_purpose_probs.csv')
    df = pd.read_csv(f, comment='#')
    return df


def trip_purpose_calc_row_size(choosers, spec, trace_label):
    """
    rows_per_chunk calculator for trip_purpose
    """

    sizer = chunk.RowSizeEstimator(trace_label)

    chooser_row_size = len(choosers.columns)
    spec_columns = spec.shape[1] - len(PROBS_JOIN_COLUMNS)

    sizer.add_elements(chooser_row_size + spec_columns, 'choosers')

    row_size = sizer.get_hwm()
    return row_size


def choose_intermediate_trip_purpose(trips, probs_spec, trace_hh_id, trace_label):
    """
    chose purpose for intermediate trips based on probs_spec
    which assigns relative weights (summing to 1) to the possible purpose choices

    Returns
    -------
    purpose: pandas.Series of purpose (str) indexed by trip_id
    """

    non_purpose_cols = PROBS_JOIN_COLUMNS + ['depart_range_start', 'depart_range_end']
    purpose_cols = [c for c in probs_spec.columns if c not in non_purpose_cols]

    num_trips = len(trips.index)
    have_trace_targets = trace_hh_id and tracing.has_trace_targets(trips)

    # probs should sum to 1 across rows
    sum_probs = probs_spec[purpose_cols].sum(axis=1)
    probs_spec.loc[:, purpose_cols] = probs_spec.loc[:, purpose_cols].div(sum_probs, axis=0)

    # left join trips to probs (there may be multiple rows per trip for multiple depart ranges)
    choosers = pd.merge(trips.reset_index(), probs_spec, on=PROBS_JOIN_COLUMNS,
                        how='left').set_index('trip_id')
    chunk.log_df(trace_label, 'choosers', choosers)

    # select the matching depart range (this should result on in exactly one chooser row per trip)
    chooser_probs = \
        (choosers.start >= choosers['depart_range_start']) & (choosers.start <= choosers['depart_range_end'])

    # if we failed to match a row in probs_spec
    if chooser_probs.sum() < num_trips:
        # this can happen if the spec doesn't have probs for the trips matching a trip's probs_join_cols
        missing_trip_ids = trips.index[~trips.index.isin(choosers.index[chooser_probs])].values
        unmatched_choosers = choosers[choosers.index.isin(missing_trip_ids)]
        unmatched_choosers = unmatched_choosers[['person_id', 'start'] + non_purpose_cols]

        # join to persons for better diagnostics
        persons = inject.get_table('persons').to_frame()
        persons_cols = ['age', 'is_worker', 'is_student', 'is_gradeschool', 'is_highschool', 'is_university']
        unmatched_choosers = pd.merge(unmatched_choosers, persons[persons_cols],
                                      left_on='person_id', right_index=True, how='left')

        file_name = '%s.UNMATCHED_PROBS' % trace_label
        logger.error("%s %s of %s intermediate trips could not be matched to probs based on join columns  %s" %
                     (trace_label, len(unmatched_choosers), len(choosers), probs_join_cols))
        logger.info("Writing %s unmatched choosers to %s" % (len(unmatched_choosers), file_name,))
        tracing.write_csv(unmatched_choosers, file_name=file_name, transpose=False)
        raise RuntimeError("Some trips could not be matched to probs based on join columns %s." % probs_join_cols)

    # select the matching depart range (this should result on in exactly one chooser row per trip)
    choosers = choosers[chooser_probs]

    # choosers should now match trips row for row
    assert choosers.index.identical(trips.index)

    choices, rands = logit.make_choices(
        choosers[purpose_cols],
        trace_label=trace_label, trace_choosers=choosers)

    if have_trace_targets:
        tracing.trace_df(choices, '%s.choices' % trace_label, columns=[None, 'trip_purpose'])
        tracing.trace_df(rands, '%s.rands' % trace_label, columns=[None, 'rand'])

    choices = choices.map(pd.Series(purpose_cols))
    return choices


def run_trip_purpose(
        trips_df,
        chunk_size,
        trace_hh_id,
        trace_label):
    """
    trip purpose - main functionality separated from model step so it can be called iteratively

    For each intermediate stop on a tour (i.e. trip other than the last trip outbound or inbound)
    Each trip is assigned a purpose based on an observed frequency distribution

    The distribution is segmented by tour purpose, tour direction, person type,
    and, optionally, trip depart time .

    Returns
    -------
    purpose: pandas.Series of purpose (str) indexed by trip_id
    """

    model_settings = config.read_model_settings('trip_purpose.yaml')
    probs_spec = trip_purpose_probs()

    result_list = []

    # - last trip of outbound tour gets primary_purpose
    last_trip = (trips_df.trip_num == trips_df.trip_count)
    purpose = trips_df.primary_purpose[last_trip & trips_df.outbound]
    result_list.append(purpose)
    logger.info("assign purpose to %s last outbound trips", purpose.shape[0])

    # - last trip of inbound tour gets home (or work for atwork subtours)
    purpose = trips_df.primary_purpose[last_trip & ~trips_df.outbound]
    purpose = pd.Series(np.where(purpose == 'atwork', 'Work', 'Home'), index=purpose.index)
    result_list.append(purpose)
    logger.info("assign purpose to %s last inbound trips", purpose.shape[0])

    # - intermediate stops (non-last trips) purpose assigned by probability table
    trips_df = trips_df[~last_trip]
    logger.info("assign purpose to %s intermediate trips", trips_df.shape[0])

    preprocessor_settings = model_settings.get('preprocessor', None)
    if preprocessor_settings:
        locals_dict = config.get_model_constants(model_settings)
        expressions.assign_columns(
            df=trips_df,
            model_settings=preprocessor_settings,
            locals_dict=locals_dict,
            trace_label=trace_label)

    row_size = chunk_size and trip_purpose_calc_row_size(trips_df, probs_spec, trace_label)

    for i, trips_chunk, chunk_trace_label in \
            chunk.adaptive_chunked_choosers(trips_df, chunk_size, row_size, trace_label):

        choices = choose_intermediate_trip_purpose(
            trips_chunk,
            probs_spec,
            trace_hh_id,
            trace_label=chunk_trace_label)

        result_list.append(choices)

    if len(result_list) > 1:
        choices = pd.concat(result_list)

    return choices


@inject.step()
def trip_purpose(
        trips,
        chunk_size,
        trace_hh_id):

    """
    trip purpose model step - calls run_trip_purpose to run the actual model

    adds purpose column to trips
    """
    trace_label = "trip_purpose"

    trips_df = trips.to_frame()

    choices = run_trip_purpose(
        trips_df,
        chunk_size=chunk_size,
        trace_hh_id=trace_hh_id,
        trace_label=trace_label
    )

    trips_df['purpose'] = choices

    # we should have assigned a purpose to all trips
    assert not trips_df.purpose.isnull().any()

    pipeline.replace_table("trips", trips_df)

    if trace_hh_id:
        tracing.trace_df(trips_df,
                         label=trace_label,
                         slicer='trip_id',
                         index_label='trip_id',
                         warn_if_empty=True)
