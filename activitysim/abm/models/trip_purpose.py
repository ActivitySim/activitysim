# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402

import logging

import numpy as np
import pandas as pd

from activitysim.core import logit
from activitysim.core import config
from activitysim.core import inject
from activitysim.core import tracing
from activitysim.core import chunk
from activitysim.core import pipeline

from .util import expressions

logger = logging.getLogger(__name__)


def trip_purpose_probs():
    f = config.config_file_path('trip_purpose_probs.csv')
    df = pd.read_csv(f, comment='#')
    return df


def trip_purpose_rpc(chunk_size, choosers, spec, trace_label):
    """
    rows_per_chunk calculator for trip_purpose
    """

    num_choosers = len(choosers.index)

    # if not chunking, then return num_choosers
    # if chunk_size == 0:
    #     return num_choosers, 0

    chooser_row_size = len(choosers.columns)

    # extra columns from spec
    extra_columns = spec.shape[1]

    row_size = chooser_row_size + extra_columns

    # logger.debug("%s #chunk_calc choosers %s", trace_label, choosers.shape)
    # logger.debug("%s #chunk_calc spec %s", trace_label, spec.shape)
    # logger.debug("%s #chunk_calc extra_columns %s", trace_label, extra_columns)

    return chunk.rows_per_chunk(chunk_size, row_size, num_choosers, trace_label)


def choose_intermediate_trip_purpose(trips, probs_spec, trace_hh_id, trace_label):
    """
    chose purpose for intermediate trips based on probs_spec
    which assigns relative weights (summing to 1) to the possible purpose choices

    Returns
    -------
    purpose: pandas.Series of purpose (str) indexed by trip_id
    """

    probs_join_cols = ['primary_purpose', 'outbound', 'person_type']
    non_purpose_cols = probs_join_cols + ['depart_range_start', 'depart_range_end']
    purpose_cols = [c for c in probs_spec.columns if c not in non_purpose_cols]

    num_trips = len(trips.index)
    have_trace_targets = trace_hh_id and tracing.has_trace_targets(trips)

    # probs shold sum to 1 across rows
    sum_probs = probs_spec[purpose_cols].sum(axis=1)
    probs_spec.loc[:, purpose_cols] = probs_spec.loc[:, purpose_cols].div(sum_probs, axis=0)

    # left join trips to probs (there may be multiple rows per trip for multiple depart ranges)
    choosers = pd.merge(trips.reset_index(), probs_spec, on=probs_join_cols,
                        how='left').set_index('trip_id')

    chunk.log_df(trace_label, 'choosers', choosers)

    # select the matching depart range (this should result on in exactly one chooser row per trip)
    choosers = choosers[(choosers.start >= choosers['depart_range_start']) & (
                choosers.start <= choosers['depart_range_end'])]

    # choosers should now match trips row for row
    assert choosers.index.is_unique
    assert len(choosers.index) == num_trips

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

    rows_per_chunk, effective_chunk_size = \
        trip_purpose_rpc(chunk_size, trips_df, probs_spec, trace_label=trace_label)

    for i, num_chunks, trips_chunk in chunk.chunked_choosers(trips_df, rows_per_chunk):

        logger.info("Running chunk %s of %s size %d", i, num_chunks, len(trips_chunk))

        chunk_trace_label = tracing.extend_trace_label(trace_label, 'chunk_%s' % i) \
            if num_chunks > 1 else trace_label

        chunk.log_open(chunk_trace_label, chunk_size, effective_chunk_size)

        choices = choose_intermediate_trip_purpose(
            trips_chunk,
            probs_spec,
            trace_hh_id,
            trace_label=chunk_trace_label)

        chunk.log_close(chunk_trace_label)

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
