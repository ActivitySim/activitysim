# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402
from builtins import range

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

from activitysim.abm.models.util.trip import failed_trip_cohorts
from activitysim.abm.models.util.trip import cleanup_failed_trips


logger = logging.getLogger(__name__)

"""
StopDepartArrivePeriodModel

StopDepartArriveProportions.csv
tourpurp,isInbound,interval,trip,p1,p2,p3,p4,p5...p40

"""

NO_TRIP_ID = 0
NO_DEPART = 0

DEPART_ALT_BASE = 'DEPART_ALT_BASE'

FAILFIX = 'FAILFIX'
FAILFIX_CHOOSE_MOST_INITIAL = 'choose_most_initial'
FAILFIX_DROP_AND_CLEANUP = 'drop_and_cleanup'
FAILFIX_DEFAULT = FAILFIX_CHOOSE_MOST_INITIAL


def set_tour_hour(trips, tours):
    """
    add columns 'tour_hour', 'earliest', 'latest' to trips

    Parameters
    ----------
    trips: pd.DataFrame
    tours: pd.DataFrame

    Returns
    -------
    modifies trips in place
    """

    # all trips must depart between tour start and end
    trips['earliest'] = reindex(tours.start, trips.tour_id)
    trips['latest'] = reindex(tours.end, trips.tour_id)

    # tour_hour is start for outbound trips, and end for inbound trips
    trips['tour_hour'] = np.where(
        trips.outbound,
        trips['earliest'],
        trips['latest']).astype(np.int8)

    # subtours indexed by parent_tour_id
    subtours = tours.loc[tours.primary_purpose == 'atwork',
                         ['tour_num', 'tour_count', 'parent_tour_id', 'start', 'end']]
    subtours = subtours.astype(int).set_index('parent_tour_id')

    # bool series
    trip_has_subtours = trips.tour_id.isin(subtours.index)

    outbound = trip_has_subtours & trips.outbound
    trips.loc[outbound, 'latest'] = \
        reindex(subtours[subtours.tour_num == 1]['start'], trips[outbound].tour_id)

    inbound = trip_has_subtours & ~trips.outbound
    trips.loc[inbound, 'earliest'] = \
        reindex(subtours[subtours.tour_num == subtours.tour_count]['end'], trips[inbound].tour_id)


def clip_probs(trips, probs, model_settings):
    """
    zero out probs before trips.earliest or after trips.latest

    Parameters
    ----------
    trips: pd.DataFrame
    probs: pd.DataFrame
        one row per trip, one column per time period, with float prob of picking that time period

    depart_alt_base: int
        int to add to probs column index to get time period it represents.
        e.g. depart_alt_base = 5 means first column (column 0) represents 5 am

    Returns
    -------
    probs: pd.DataFrame
        clipped version of probs

    """

    depart_alt_base = model_settings.get(DEPART_ALT_BASE)

    # there should be one row in probs per trip
    assert trips.shape[0] == probs.shape[0]

    # probs should sum to 1 across rows before clipping
    probs = probs.div(probs.sum(axis=1), axis=0)

    num_rows, num_cols = probs.shape
    ix_map = np.tile(np.arange(0, num_cols), num_rows).reshape(num_rows, num_cols) + depart_alt_base
    # 5 6 7 8 9 10...
    # 5 6 7 8 9 10...
    # 5 6 7 8 9 10...

    clip_mask = ((ix_map >= trips.earliest.values.reshape(num_rows, 1)) &
                 (ix_map <= trips.latest.values.reshape(num_rows, 1))) * 1
    #  [0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0]
    #  [0 0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0]
    #  [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0]...

    probs = probs*clip_mask

    return probs


def report_bad_choices(bad_row_map, df, filename, trace_label, trace_choosers=None):
    """

    Parameters
    ----------
    bad_row_map
    df : pandas.DataFrame
        utils or probs dataframe
    trace_choosers : pandas.dataframe
        the choosers df (for interaction_simulate) to facilitate the reporting of hh_id
        because we  can't deduce hh_id from the interaction_dataset which is indexed on index
        values from alternatives df

    """

    df = df[bad_row_map]
    if trace_choosers is None:
        hh_ids = tracing.hh_id_for_chooser(df.index, df)
    else:
        hh_ids = tracing.hh_id_for_chooser(df.index, trace_choosers)
    df['household_id'] = hh_ids

    filename = "%s.%s" % (trace_label, filename)

    logger.info("dumping %s" % filename)
    tracing.write_csv(df, file_name=filename, transpose=False)

    # log the indexes of the first MAX_PRINT offending rows
    MAX_PRINT = 0
    for idx in df.index[:MAX_PRINT].values:

        row_msg = "%s : failed %s = %s (hh_id = %s)" % \
                  (trace_label, df.index.name, idx, df.household_id.loc[idx])

        logger.warning(row_msg)


def schedule_nth_trips(
        trips,
        probs_spec,
        model_settings,
        first_trip_in_leg,
        report_failed_trips,
        trace_hh_id,
        trace_label):
    """
    We join each trip with the appropriate row in probs_spec by joining on probs_join_cols,
    which should exist in both trips, probs_spec dataframe.

    Parameters
    ----------
    trips: pd.DataFrame
    probs_spec: pd.DataFrame
        Dataframe of probs for choice of depart times and join columns to match them with trips.
        Depart columns names are irrelevant. Instead, they are position dependent,
        time period choice is their index + depart_alt_base
    depart_alt_base: int
        int to add to probs column index to get time period it represents.
        e.g. depart_alt_base = 5 means first column (column 0) represents 5 am
    report_failed_trips : bool
    trace_hh_id
    trace_label

    Returns
    -------
    choices: pd.Series
        time periods depart choices, one per trip (except for trips with zero probs)
    """

    depart_alt_base = model_settings.get('DEPART_ALT_BASE')

    probs_join_cols = ['primary_purpose', 'outbound', 'tour_hour', 'trip_num']
    probs_cols = [c for c in probs_spec.columns if c not in probs_join_cols]

    # left join trips to probs (there may be multiple rows per trip for multiple depart ranges)
    choosers = pd.merge(trips.reset_index(), probs_spec, on=probs_join_cols,
                        how='left').set_index('trip_id')
    chunk.log_df(trace_label, "choosers", choosers)

    if trace_hh_id and tracing.has_trace_targets(trips):
        tracing.trace_df(choosers, '%s.choosers' % trace_label)

    # choosers should now match trips row for row
    assert choosers.index.is_unique
    assert len(choosers.index) == len(trips.index)

    # zero out probs outside earliest-latest window
    chooser_probs = clip_probs(trips, choosers[probs_cols], model_settings)

    chunk.log_df(trace_label, "chooser_probs", chooser_probs)

    if first_trip_in_leg:
        # probs should sum to 1 unless all zero
        chooser_probs = chooser_probs.div(chooser_probs.sum(axis=1), axis=0).fillna(0)

    # probs should sum to 1 with residual probs resulting in choice of 'fail'
    chooser_probs['fail'] = 1 - chooser_probs.sum(axis=1).clip(0, 1)

    if trace_hh_id and tracing.has_trace_targets(trips):
        tracing.trace_df(chooser_probs, '%s.chooser_probs' % trace_label)

    choices, rands = logit.make_choices(
        chooser_probs,
        trace_label=trace_label, trace_choosers=choosers)

    chunk.log_df(trace_label, "choices", choices)
    chunk.log_df(trace_label, "rands", rands)

    if trace_hh_id and tracing.has_trace_targets(trips):
        tracing.trace_df(choices, '%s.choices' % trace_label, columns=[None, 'depart'])
        tracing.trace_df(rands, '%s.rands' % trace_label, columns=[None, 'rand'])

    # convert alt choice index to depart time (setting failed choices to -1)
    failed = (choices == chooser_probs.columns.get_loc('fail'))
    choices = (choices + depart_alt_base).where(~failed, -1)

    chunk.log_df(trace_label, "failed", failed)

    # report failed trips while we have the best diagnostic info
    if report_failed_trips and failed.any():
        report_bad_choices(
            bad_row_map=failed,
            df=choosers,
            filename='failed_choosers',
            trace_label=trace_label,
            trace_choosers=None)

    # trace before removing failures
    if trace_hh_id and tracing.has_trace_targets(trips):
        tracing.trace_df(choices, '%s.choices' % trace_label, columns=[None, 'depart'])
        tracing.trace_df(rands, '%s.rands' % trace_label, columns=[None, 'rand'])

    # remove any failed choices
    if failed.any():
        choices = choices[~failed]

    assert (choices >= trips.earliest[~failed]).all()
    assert (choices <= trips.latest[~failed]).all()

    return choices


def schedule_trips_in_leg(
        outbound,
        trips,
        probs_spec,
        model_settings,
        last_iteration,
        trace_hh_id, trace_label):
    """

    Parameters
    ----------
    outbound
    trips
    probs_spec
    depart_alt_base
    last_iteration
    trace_hh_id
    trace_label

    Returns
    -------
    choices: pd.Series
        depart choice for trips, indexed by trip_id
    """

    failfix = model_settings.get(FAILFIX, FAILFIX_DEFAULT)

    # logger.debug("%s scheduling %s trips" % (trace_label, trips.shape[0]))

    assert (trips.outbound == outbound).all()

    # initial trip of leg and all atwork trips get tour_hour
    is_initial = (trips.trip_num == 1) if outbound else (trips.trip_num == trips.trip_count)
    no_scheduling = is_initial | (trips.primary_purpose == 'atwork')
    choices = trips.tour_hour[no_scheduling]

    if no_scheduling.all():
        return choices

    result_list = []
    result_list.append(choices)
    trips = trips[~no_scheduling]

    # add next_trip_id temp column (temp as trips is now a copy, as result of slicing)
    trips = trips.sort_index()
    trips['next_trip_id'] = np.roll(trips.index, -1 if outbound else 1)
    is_final = (trips.trip_num == trips.trip_count) if outbound else (trips.trip_num == 1)
    trips.next_trip_id = trips.next_trip_id.where(is_final, NO_TRIP_ID)

    # iterate over outbound trips in ascending trip_num order, skipping the initial trip
    # iterate over inbound trips in descending trip_num order, skipping the finial trip
    first_trip_in_leg = True
    for i in range(trips.trip_num.min(), trips.trip_num.max() + 1):

        if outbound:
            nth_trips = trips[trips.trip_num == i]
        else:
            nth_trips = trips[trips.trip_num == trips.trip_count - i]

        nth_trace_label = tracing.extend_trace_label(trace_label, 'num_%s' % i)

        chunk.log_open(nth_trace_label, chunk_size=0, effective_chunk_size=0)

        choices = schedule_nth_trips(
            nth_trips,
            probs_spec,
            model_settings,
            first_trip_in_leg=first_trip_in_leg,
            report_failed_trips=last_iteration,
            trace_hh_id=trace_hh_id,
            trace_label=nth_trace_label)

        chunk.log_close(nth_trace_label)

        # if outbound, this trip's depart constrains next trip's earliest depart option
        # if inbound, we are handling in reverse order, so it constrains latest depart instead
        ADJUST_NEXT_DEPART_COL = 'earliest' if outbound else 'latest'

        # most initial departure (when no choice was made because all probs were zero)
        if last_iteration and (failfix == FAILFIX_CHOOSE_MOST_INITIAL):
            choices = choices.reindex(nth_trips.index)
            logger.warning("%s coercing %s depart choices to most initial" %
                           (nth_trace_label, choices.isna().sum()))
            choices = choices.fillna(trips[ADJUST_NEXT_DEPART_COL])

        # adjust allowed depart range of next trip
        has_next_trip = (nth_trips.next_trip_id != NO_TRIP_ID)
        if has_next_trip.any():
            next_trip_ids = nth_trips.next_trip_id[has_next_trip]
            # patch choice any trips with next_trips that weren't scheduled
            trips.loc[next_trip_ids, ADJUST_NEXT_DEPART_COL] = \
                choices.reindex(next_trip_ids.index).fillna(trips[ADJUST_NEXT_DEPART_COL]).values

        result_list.append(choices)

        first_trip_in_leg = False

    if len(result_list) > 1:
        choices = pd.concat(result_list)

    return choices


def trip_scheduling_rpc(chunk_size, choosers, spec, trace_label):

    # NOTE we chunk chunk_id
    num_choosers = choosers['chunk_id'].max() + 1

    # if not chunking, then return num_choosers
    # if chunk_size == 0:
    #     return num_choosers, 0

    # extra columns from spec
    extra_columns = spec.shape[1]

    chooser_row_size = choosers.shape[1] + extra_columns

    # scale row_size by average number of chooser rows per chunk_id
    rows_per_chunk_id = choosers.shape[0] / num_choosers
    row_size = (rows_per_chunk_id * chooser_row_size)

    # print "num_choosers", num_choosers
    # print "choosers.shape", choosers.shape
    # print "rows_per_chunk_id", rows_per_chunk_id
    # print "chooser_row_size", chooser_row_size
    # print "(rows_per_chunk_id * chooser_row_size)", (rows_per_chunk_id * chooser_row_size)
    # print "row_size", row_size
    # #bug

    return chunk.rows_per_chunk(chunk_size, row_size, num_choosers, trace_label)


def run_trip_scheduling(
        trips,
        tours,
        probs_spec,
        model_settings,
        last_iteration,
        chunk_size,
        trace_hh_id,
        trace_label):

    set_tour_hour(trips, tours)

    rows_per_chunk, effective_chunk_size = \
        trip_scheduling_rpc(chunk_size, trips, probs_spec, trace_label)

    result_list = []
    for i, num_chunks, trips_chunk in chunk.chunked_choosers_by_chunk_id(trips, rows_per_chunk):

        if num_chunks > 1:
            chunk_trace_label = tracing.extend_trace_label(trace_label, 'chunk_%s' % i)
            logger.info("%s of %s size %d" % (chunk_trace_label, num_chunks, len(trips_chunk)))
        else:
            chunk_trace_label = trace_label

        leg_trace_label = tracing.extend_trace_label(chunk_trace_label, 'outbound')
        chunk.log_open(leg_trace_label, chunk_size, effective_chunk_size)
        choices = \
            schedule_trips_in_leg(
                outbound=True,
                trips=trips_chunk[trips_chunk.outbound],
                probs_spec=probs_spec,
                model_settings=model_settings,
                last_iteration=last_iteration,
                trace_hh_id=trace_hh_id,
                trace_label=leg_trace_label)
        result_list.append(choices)
        chunk.log_close(leg_trace_label)

        leg_trace_label = tracing.extend_trace_label(chunk_trace_label, 'inbound')
        chunk.log_open(leg_trace_label, chunk_size, effective_chunk_size)
        choices = \
            schedule_trips_in_leg(
                outbound=False,
                trips=trips_chunk[~trips_chunk.outbound],
                probs_spec=probs_spec,
                model_settings=model_settings,
                last_iteration=last_iteration,
                trace_hh_id=trace_hh_id,
                trace_label=leg_trace_label)
        result_list.append(choices)
        chunk.log_close(leg_trace_label)

    choices = pd.concat(result_list)

    return choices


@inject.step()
def trip_scheduling(
        trips,
        tours,
        chunk_size,
        trace_hh_id):

    """
    Trip scheduling assigns depart times for trips within the start, end limits of the tour.

    The algorithm is simplistic:

    The first outbound trip starts at the tour start time, and subsequent outbound trips are
    processed in trip_num order, to ensure that subsequent trips do not depart before the
    trip that preceeds them.

    Inbound trips are handled similarly, except in reverse order, starting with the last trip,
    and working backwards to ensure that inbound trips do not depart after the trip that
    succeeds them.

    The probability spec assigns probabilities for depart times, but those possible departs must
    be clipped to disallow depart times outside the tour limits, the departs of prior trips, and
    in the case of work tours, the start/end times of any atwork subtours.

    Scheduling can fail if the probability table assigns zero probabilities to all the available
    depart times in a trip's depart window. (This could be avoided by giving every window a small
    probability, rather than zero, but the existing mtctm1 prob spec does not do this. I believe
    this is due to the its having been generated from a small household travel survey sample
    that lacked any departs for some time periods.)

    Rescheduling the trips that fail (along with their inbound or outbound leg-mates) can sometimes
    fix this problem, if it was caused by an earlier trip's depart choice blocking a subsequent
    trip's ability to schedule a depart within the resulting window. But it can also happen if
    a tour is very short (e.g. one time period) and the prob spec having a zero probability for
    that tour hour.

    Therefor we need to handle trips that could not be scheduled. There are two ways (at least)
    to solve this problem:

    1) CHOOSE_MOST_INITIAL
    simply assign a depart time to the trip, even if it has a zero probability. It makes
    most sense, in this case, to assign the 'most initial' depart time, so that subsequent trips
    are minimally impacted. This can be done in the final iteration, thus affecting only the
    trips that could no be scheduled by the standard approach

    2) drop_and_cleanup
    drop trips that could no be scheduled, and adjust their leg mates, as is done for failed
    trips in trip_destination.

    For now we are choosing among these approaches with a manifest constant, but this could
    be made a model setting...

    """
    trace_label = "trip_scheduling"

    model_settings = config.read_model_settings('trip_scheduling.yaml')
    assert 'DEPART_ALT_BASE' in model_settings

    failfix = model_settings.get(FAILFIX, FAILFIX_DEFAULT)

    probs_spec = pd.read_csv(config.config_file_path('trip_scheduling_probs.csv'), comment='#')

    trips_df = trips.to_frame()
    tours = tours.to_frame()

    # add tour-based chunk_id so we can chunk all trips in tour together
    trips_df['chunk_id'] = \
        reindex(pd.Series(list(range(tours.shape[0])), tours.index), trips_df.tour_id)

    max_iterations = model_settings.get('MAX_ITERATIONS', 1)
    assert max_iterations > 0

    choices_list = []
    i = 0
    while (i < max_iterations) and not trips_df.empty:

        i += 1
        last_iteration = (i == max_iterations)

        trace_label_i = tracing.extend_trace_label(trace_label, "i%s" % i)
        logger.info("%s scheduling %s trips", trace_label_i, trips_df.shape[0])

        choices = \
            run_trip_scheduling(
                trips_df,
                tours,
                probs_spec,
                model_settings,
                last_iteration=last_iteration,
                trace_hh_id=trace_hh_id,
                chunk_size=chunk_size,
                trace_label=trace_label_i)

        # boolean series of trips whose individual trip scheduling failed
        failed = choices.reindex(trips_df.index).isnull()
        logger.info("%s %s failed", trace_label_i, failed.sum())

        if not last_iteration:
            # boolean series of trips whose leg scheduling failed
            failed_cohorts = failed_trip_cohorts(trips_df, failed)
            trips_df = trips_df[failed_cohorts]
            choices = choices[~failed_cohorts]

        choices_list.append(choices)

    trips_df = trips.to_frame()

    choices = pd.concat(choices_list)
    choices = choices.reindex(trips_df.index)
    if choices.isnull().any():
        logger.warning("%s of %s trips could not be scheduled after %s iterations" %
                       (choices.isnull().sum(), trips_df.shape[0], i))

        if failfix != FAILFIX_DROP_AND_CLEANUP:
            raise RuntimeError("%s setting '%s' not enabled in settings" %
                               (FAILFIX, FAILFIX_DROP_AND_CLEANUP))

        trips_df['failed'] = choices.isnull()
        trips_df = cleanup_failed_trips(trips_df)
        choices = choices.reindex(trips_df.index)

    trips_df['depart'] = choices

    assert not trips_df.depart.isnull().any()

    pipeline.replace_table("trips", trips_df)
