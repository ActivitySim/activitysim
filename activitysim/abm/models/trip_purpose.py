# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import numpy as np
import pandas as pd

from activitysim.core import logit
from activitysim.core import config
from activitysim.core import inject
from activitysim.core import tracing
from activitysim.core import chunk
from activitysim.core import pipeline

from activitysim.core.util import assign_in_place
from .util import expressions
from activitysim.core.util import reindex

logger = logging.getLogger(__name__)


@inject.injectable()
def trip_purpose_settings(configs_dir):
    return config.read_model_settings(configs_dir, 'trip_purpose.yaml')


@inject.injectable()
def trip_purpose_probs(configs_dir):

    f = os.path.join(configs_dir, 'trip_purpose_probs.csv')
    df = pd.read_csv(f, comment='#')
    return df


def trip_purpose_rpc(chunk_size, choosers, spec, trace_label):
    """
    rows_per_chunk calculator for trip_purpose
    """

    num_choosers = len(choosers.index)

    # if not chunking, then return num_choosers
    if chunk_size == 0:
        return num_choosers

    chooser_row_size = len(choosers.columns)

    # extra columns from spec
    extra_columns = spec.shape[1]

    row_size = chooser_row_size + extra_columns

    logger.debug("%s #chunk_calc choosers %s" % (trace_label, choosers.shape))
    logger.debug("%s #chunk_calc spec %s" % (trace_label, spec.shape))
    logger.debug("%s #chunk_calc extra_columns %s" % (trace_label, extra_columns))

    return chunk.rows_per_chunk(chunk_size, row_size, num_choosers, trace_label)


def choose_trip_purpose(trips, probs_spec, trace_label):

    probs_join_cols = ['primary_purpose', 'outbound', 'person_type']
    non_purpose_cols = probs_join_cols + ['depart_range_start', 'depart_range_end']
    purpose_cols = [c for c in probs_spec.columns if c not in non_purpose_cols]

    num_trips = len(trips.index)
    have_trace_targets = trace_label and tracing.has_trace_targets(trips)

    # left join trips to probs (there may be multiple rows per trip for multiple depart ranges)
    choosers = pd.merge(trips.reset_index(), probs_spec, on=probs_join_cols,
                        how='left').set_index('trip_id')

    # select the matching depart range (this should result on in exactly one chooser row per trip)
    choosers = choosers[(choosers.start >= choosers['depart_range_start']) & (
                choosers.start <= choosers['depart_range_end'])]

    # choosers should now match trips row for row
    assert choosers.index.is_unique
    assert len(choosers.index) == num_trips

    choices, rands = logit.make_choices(
        choosers[purpose_cols],
        trace_label=trace_label, trace_choosers=choosers)

    cum_size = chunk.log_df_size(trace_label, 'choosers', choosers, cum_size=None)
    chunk.log_chunk_size(trace_label, cum_size)

    if have_trace_targets:
        tracing.trace_df(choices, '%s.choices' % trace_label, columns=[None, 'trip_purpose'])
        tracing.trace_df(rands, '%s.rands' % trace_label, columns=[None, 'rand'])

    choices = choices.map(pd.Series(purpose_cols))
    return choices


@inject.step()
def trip_purpose(
        trips,
        trip_purpose_settings,
        trip_purpose_probs,
        chunk_size,
        trace_hh_id):
    """
    trip purpose
    """

    trace_label = "trip_purpose"

    trips_df = trips.to_frame()
    probs_spec = trip_purpose_probs

    preprocessor_settings = trip_purpose_settings.get('preprocessor_settings', None)
    if preprocessor_settings:
        locals_dict = config.get_model_constants(trip_purpose_settings)
        expressions.assign_columns(
            df=trips_df,
            model_settings=preprocessor_settings,
            locals_dict=locals_dict,
            trace_label=trace_label)

    rows_per_chunk = \
        trip_purpose_rpc(chunk_size, trips_df, probs_spec, trace_label=trace_label)

    logger.info("simple_simulate rows_per_chunk %s num_choosers %s" %
                (rows_per_chunk, len(trips_df.index)))

    result_list = []
    # segment by person type and pick the right spec for each person type
    for i, num_chunks, trips_chunk in chunk.chunked_choosers(trips_df, rows_per_chunk):

        logger.info("Running chunk %s of %s size %d" % (i, num_chunks, len(trips_chunk)))

        chunk_trace_label = tracing.extend_trace_label(trace_label, 'chunk_%s' % i) \
            if num_chunks > 1 else trace_label

        choices = choose_trip_purpose(
            trips_chunk,
            probs_spec,
            trace_label=chunk_trace_label)

        result_list.append(choices)

    if len(result_list) > 1:
        choices = pd.concat(result_list)

    trips_df = trips.to_frame()
    trips_df['purpose'] = choices.reindex(trips_df.index)
    pipeline.replace_table("trips", trips_df)
