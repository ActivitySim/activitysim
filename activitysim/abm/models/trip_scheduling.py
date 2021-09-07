# ActivitySim
# See full license in LICENSE.txt.
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

from activitysim.core.util import reindex

from activitysim.abm.models.util.trip import failed_trip_cohorts
from activitysim.abm.models.util.trip import cleanup_failed_trips

from activitysim.abm.models.util import estimation


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

PROBS_JOIN_COLUMNS = ['primary_purpose', 'outbound', 'tour_hour', 'trip_num']


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

    subtours.parent_tour_id = subtours.parent_tour_id.astype(np.int64)
    subtours = subtours.set_index('parent_tour_id')
    subtours = subtours.astype(np.int16)  # remaining columns are all small ints

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

    probs_cols = [c for c in probs_spec.columns if c not in PROBS_JOIN_COLUMNS]

    # left join trips to probs (there may be multiple rows per trip for multiple depart ranges)
    choosers = pd.merge(trips.reset_index(), probs_spec, on=PROBS_JOIN_COLUMNS,
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
    chunk.log_df(trace_label, "chooser_probs", chooser_probs)

    if trace_hh_id and tracing.has_trace_targets(trips):
        tracing.trace_df(chooser_probs, '%s.chooser_probs' % trace_label)

    choices, rands = logit.make_choices(chooser_probs, trace_label=trace_label, trace_choosers=choosers)

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
        is_last_iteration,
        trace_hh_id, trace_label):
    """

    Parameters
    ----------
    outbound
    trips
    probs_spec
    depart_alt_base
    is_last_iteration
    trace_hh_id
    trace_label

    Returns
    -------
    choices: pd.Series
        depart choice for trips, indexed by trip_id
    """

    failfix = model_settings.get(FAILFIX, FAILFIX_DEFAULT)

    # logger.debug("%s scheduling %s trips" % (trace_label, trips.shape[0]))

    assert len(trips) > 0

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
    trips.next_trip_id = trips.next_trip_id.where(~is_final, NO_TRIP_ID)

    # iterate over outbound trips in ascending trip_num order, skipping the initial trip
    # iterate over inbound trips in descending trip_num order, skipping the finial trip
    first_trip_in_leg = True
    for i in range(trips.trip_num.min(), trips.trip_num.max() + 1):

        if outbound:
            nth_trips = trips[trips.trip_num == i]
        else:
            nth_trips = trips[trips.trip_num == trips.trip_count - i]

        nth_trace_label = tracing.extend_trace_label(trace_label, 'num_%s' % i)

        choices = schedule_nth_trips(
            nth_trips,
            probs_spec,
            model_settings,
            first_trip_in_leg=first_trip_in_leg,
            report_failed_trips=is_last_iteration,
            trace_hh_id=trace_hh_id,
            trace_label=nth_trace_label)

        # if outbound, this trip's depart constrains next trip's earliest depart option
        # if inbound, we are handling in reverse order, so it constrains latest depart instead
        ADJUST_NEXT_DEPART_COL = 'earliest' if outbound else 'latest'

        # most initial departure (when no choice was made because all probs were zero)
        if is_last_iteration and (failfix == FAILFIX_CHOOSE_MOST_INITIAL):
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

        chunk.log_df(trace_label, f'result_list', result_list)

        first_trip_in_leg = False

    if len(result_list) > 1:
        choices = pd.concat(result_list)

    return choices


def run_trip_scheduling(
        trips_chunk,
        tours,
        probs_spec,
        model_settings,
        estimator,
        is_last_iteration,
        chunk_size,
        chunk_tag,
        trace_hh_id,
        trace_label):

    # only non-initial trips require scheduling, segment handing first such trip in tour will use most space
    # is_outbound_chooser = (trips.trip_num > 1) & trips.outbound & (trips.primary_purpose != 'atwork')
    # is_inbound_chooser = (trips.trip_num < trips.trip_count) & ~trips.outbound & (trips.primary_purpose != 'atwork')
    # num_choosers = (is_inbound_chooser | is_outbound_chooser).sum()

    result_list = []

    if trips_chunk.outbound.any():
        leg_chunk = trips_chunk[trips_chunk.outbound]
        leg_trace_label = tracing.extend_trace_label(trace_label, 'outbound')
        choices = \
            schedule_trips_in_leg(
                outbound=True,
                trips=leg_chunk,
                probs_spec=probs_spec,
                model_settings=model_settings,
                is_last_iteration=is_last_iteration,
                trace_hh_id=trace_hh_id,
                trace_label=leg_trace_label)
        result_list.append(choices)

        chunk.log_df(trace_label, f'result_list', result_list)

    if (~trips_chunk.outbound).any():
        leg_chunk = trips_chunk[~trips_chunk.outbound]
        leg_trace_label = tracing.extend_trace_label(trace_label, 'inbound')
        choices = \
            schedule_trips_in_leg(
                outbound=False,
                trips=leg_chunk,
                probs_spec=probs_spec,
                model_settings=model_settings,
                is_last_iteration=is_last_iteration,
                trace_hh_id=trace_hh_id,
                trace_label=leg_trace_label)
        result_list.append(choices)

        chunk.log_df(trace_label, f'result_list', result_list)

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

    Therefore we need to handle trips that could not be scheduled. There are two ways (at least)
    to solve this problem:

    1) choose_most_initial
    simply assign a depart time to the trip, even if it has a zero probability. It makes
    most sense, in this case, to assign the 'most initial' depart time, so that subsequent trips
    are minimally impacted. This can be done in the final iteration, thus affecting only the
    trips that could no be scheduled by the standard approach

    2) drop_and_cleanup
    drop trips that could no be scheduled, and adjust their leg mates, as is done for failed
    trips in trip_destination.

    Which option is applied is determined by the FAILFIX model setting

    """
    trace_label = "trip_scheduling"
    model_settings_file_name = 'trip_scheduling.yaml'
    model_settings = config.read_model_settings(model_settings_file_name)

    trips_df = trips.to_frame()
    tours = tours.to_frame()

    # add columns 'tour_hour', 'earliest', 'latest' to trips
    set_tour_hour(trips_df, tours)

    # trip_scheduling is a probabilistic model ane we don't support estimation,
    # but we do need to override choices in estimation mode
    estimator = estimation.manager.begin_estimation('trip_scheduling')
    if estimator:
        estimator.write_spec(model_settings, tag='PROBS_SPEC')
        estimator.write_model_settings(model_settings, model_settings_file_name)
        chooser_cols_for_estimation = ['person_id',  'household_id',  'tour_id',  'trip_num', 'trip_count',
                                       'primary_purpose', 'outbound', 'earliest', 'latest', 'tour_hour', ]
        estimator.write_choosers(trips_df[chooser_cols_for_estimation])

    probs_spec = pd.read_csv(config.config_file_path('trip_scheduling_probs.csv'), comment='#')
    # FIXME for now, not really doing estimation for probabilistic model - just overwriting choices
    # besides, it isn't clear that named coefficients would be helpful if we had some form of estimation
    # coefficients_df = simulate.read_model_coefficients(model_settings)
    # probs_spec = map_coefficients(probs_spec, coefficients_df)

    # add tour-based chunk_id so we can chunk all trips in tour together
    trips_df['chunk_id'] = reindex(pd.Series(list(range(len(tours))), tours.index), trips_df.tour_id)

    assert 'DEPART_ALT_BASE' in model_settings
    failfix = model_settings.get(FAILFIX, FAILFIX_DEFAULT)

    max_iterations = model_settings.get('MAX_ITERATIONS', 1)
    assert max_iterations > 0

    choices_list = []

    for chunk_i, trips_chunk, chunk_trace_label in chunk.adaptive_chunked_choosers_by_chunk_id(trips_df,
                                                                                               chunk_size,
                                                                                               trace_label,
                                                                                               trace_label):

        i = 0
        while (i < max_iterations) and not trips_chunk.empty:

            # only chunk log first iteration since memory use declines with each iteration
            with chunk.chunk_log(trace_label) if i == 0 else chunk.chunk_log_skip():

                i += 1
                is_last_iteration = (i == max_iterations)

                trace_label_i = tracing.extend_trace_label(trace_label, "i%s" % i)
                logger.info("%s scheduling %s trips within chunk %s", trace_label_i, trips_chunk.shape[0], chunk_i)

                choices = \
                    run_trip_scheduling(
                        trips_chunk,
                        tours,
                        probs_spec,
                        model_settings,
                        estimator=estimator,
                        is_last_iteration=is_last_iteration,
                        trace_hh_id=trace_hh_id,
                        chunk_size=chunk_size,
                        chunk_tag=trace_label,
                        trace_label=trace_label_i)

                # boolean series of trips whose individual trip scheduling failed
                failed = choices.reindex(trips_chunk.index).isnull()
                logger.info("%s %s failed", trace_label_i, failed.sum())

                if not is_last_iteration:
                    # boolean series of trips whose leg scheduling failed
                    failed_cohorts = failed_trip_cohorts(trips_chunk, failed)
                    trips_chunk = trips_chunk[failed_cohorts]
                    choices = choices[~failed_cohorts]

                choices_list.append(choices)

    trips_df = trips.to_frame()

    choices = pd.concat(choices_list)
    choices = choices.reindex(trips_df.index)

    if estimator:
        estimator.write_choices(choices)
        choices = estimator.get_survey_values(choices, 'trips', 'depart')  # override choices
        estimator.write_override_choices(choices)
        estimator.end_estimation()
        assert not choices.isnull().any()

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
