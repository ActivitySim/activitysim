# ActivitySim
# See full license in LICENSE.txt.
import logging
from builtins import range

import numpy as np
import pandas as pd

from activitysim.abm.models.util import estimation
from activitysim.abm.models.util.trip import cleanup_failed_trips, failed_trip_cohorts
from activitysim.core import chunk, config, inject, logit, pipeline, tracing
from activitysim.core.util import reindex

from .util import probabilistic_scheduling as ps

logger = logging.getLogger(__name__)

"""
StopDepartArrivePeriodModel

StopDepartArriveProportions.csv
tourpurp,isInbound,interval,trip,p1,p2,p3,p4,p5...p40

"""

NO_TRIP_ID = 0
NO_DEPART = 0

DEPART_ALT_BASE = "DEPART_ALT_BASE"

FAILFIX = "FAILFIX"
FAILFIX_CHOOSE_MOST_INITIAL = "choose_most_initial"
FAILFIX_DROP_AND_CLEANUP = "drop_and_cleanup"
FAILFIX_DEFAULT = FAILFIX_CHOOSE_MOST_INITIAL

DEPARTURE_MODE = "departure"
DURATION_MODE = "stop_duration"
PROBS_JOIN_COLUMNS_DEPARTURE_BASED = [
    "primary_purpose",
    "outbound",
    "tour_hour",
    "trip_num",
]
PROBS_JOIN_COLUMNS_DURATION_BASED = ["outbound", "stop_num"]


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
    trips["earliest"] = reindex(tours.start, trips.tour_id)
    trips["latest"] = reindex(tours.end, trips.tour_id)

    # tour_hour is start for outbound trips, and end for inbound trips
    trips["tour_hour"] = np.where(
        trips.outbound, trips["earliest"], trips["latest"]
    ).astype(np.int8)

    # subtours indexed by parent_tour_id
    subtours = tours.loc[
        tours.primary_purpose == "atwork",
        ["tour_num", "tour_count", "parent_tour_id", "start", "end"],
    ]

    subtours.parent_tour_id = subtours.parent_tour_id.astype(np.int64)
    subtours = subtours.set_index("parent_tour_id")
    subtours = subtours.astype(np.int16)  # remaining columns are all small ints

    # bool series
    trip_has_subtours = trips.tour_id.isin(subtours.index)

    outbound = trip_has_subtours & trips.outbound
    trips.loc[outbound, "latest"] = reindex(
        subtours[subtours.tour_num == 1]["start"], trips[outbound].tour_id
    )

    inbound = trip_has_subtours & ~trips.outbound
    trips.loc[inbound, "earliest"] = reindex(
        subtours[subtours.tour_num == subtours.tour_count]["end"],
        trips[inbound].tour_id,
    )


def set_stop_num(trips):
    """
    Convert trip_num to stop_num in order to work with duration-based
    probs that are keyed on stop num. For outbound trips, trip n chooses
    the duration of stop n-1 (the trip origin). For inbound trips, trip n
    chooses the duration of stop n (the trip destination). This means
    outbound trips technically choose a departure time while inbound trips
    choose an arrival.
    """
    trips["stop_num"] = trips["trip_num"] - 1
    trips["stop_num"] = trips.stop_num.where(trips["outbound"], trips["trip_num"])


def update_tour_earliest(trips, outbound_choices):
    """
    Updates "earliest" column for inbound trips based on
    the maximum outbound trip departure time of the tour.
    This is done to ensure inbound trips do not depart
    before the last outbound trip of a tour.

    Parameters
    ----------
    trips: pd.DataFrame
    outbound_choices: pd.Series
        time periods depart choices, one per trip (except for trips with
        zero probs)
    Returns
    -------
    modifies trips in place
    """
    # append outbound departure times to trips
    tmp_trips = trips.copy()
    tmp_trips["outbound_departure"] = outbound_choices.reindex(tmp_trips.index)

    # get max outbound trip departure times for all person-tours
    max_outbound_person_departures = tmp_trips.groupby(["person_id", "tour_id"])[
        "outbound_departure"
    ].max()
    max_outbound_person_departures.name = "max_outbound_departure"

    # append max outbound trip departure times to trips
    tmp_trips = tmp_trips.merge(
        max_outbound_person_departures,
        left_on=["person_id", "tour_id"],
        right_index=True,
    )

    # set the trips "earliest" column equal to the max outbound departure
    # time for all inbound trips. preserve values that were used for outbound trips
    tmp_trips["earliest"] = tmp_trips["earliest"].where(
        tmp_trips["outbound"], tmp_trips["max_outbound_departure"]
    )

    trips["earliest"] = tmp_trips["earliest"].reindex(trips.index)

    return


def schedule_trips_in_leg(
    outbound,
    trips,
    probs_spec,
    model_settings,
    is_last_iteration,
    trace_hh_id,
    trace_label,
):
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
    depart_alt_base = model_settings.get("DEPART_ALT_BASE", 0)
    scheduling_mode = model_settings.get("scheduling_mode", "departure")

    if scheduling_mode == "departure":
        probs_join_cols = model_settings.get(
            "probs_join_cols", PROBS_JOIN_COLUMNS_DEPARTURE_BASED
        )
    elif scheduling_mode == "stop_duration":
        probs_join_cols = model_settings.get(
            "probs_join_cols", PROBS_JOIN_COLUMNS_DURATION_BASED
        )
    else:
        logger.error(
            "Invalid scheduling mode specified: {0}.".format(scheduling_mode),
            "Please select one of ['departure', 'stop_duration'] and try again.",
        )

    # logger.debug("%s scheduling %s trips" % (trace_label, trips.shape[0]))

    assert len(trips) > 0
    assert (trips.outbound == outbound).all()

    result_list = []

    # trips to/from tour origin or atwork get tour_hour departure times
    # no need to schedule them if there are no intermediate stops
    to_from_tour_orig = (
        (trips.trip_num == 1) if outbound else (trips.trip_num == trips.trip_count)
    )
    do_not_schedule = to_from_tour_orig | (trips.primary_purpose == "atwork")
    choices = trips.tour_hour[do_not_schedule]

    if do_not_schedule.all():
        return choices

    result_list.append(choices)
    trips = trips[~do_not_schedule]

    # add next_trip_id temp column, and specificy departure constraint column to update
    trips = trips.sort_index()
    if outbound or scheduling_mode == DURATION_MODE:
        trips["next_trip_id"] = np.roll(trips.index, -1)
        is_final = trips.trip_num == trips.trip_count
        # each trip's depart constrains next trip's earliest depart option
        ADJUST_NEXT_DEPART_COL = "earliest"
    else:
        trips["next_trip_id"] = np.roll(trips.index, 1)
        is_final = trips.trip_num == 1
        # if inbound, we are handling in reverse order, so each choice
        # constrains latest depart of the preceding trip
        ADJUST_NEXT_DEPART_COL = "latest"
    trips.next_trip_id = trips.next_trip_id.where(~is_final, NO_TRIP_ID)

    first_trip_in_leg = True
    for i in range(trips.trip_num.min(), trips.trip_num.max() + 1):

        if outbound or scheduling_mode == DURATION_MODE:
            # iterate in ascending trip_num order
            nth_trips = trips[trips.trip_num == i]
        else:
            # iterate over inbound trips in descending trip_num order, skipping the final trip
            nth_trips = trips[trips.trip_num == trips.trip_count - i]

        nth_trace_label = tracing.extend_trace_label(trace_label, "num_%s" % i)

        choices = ps.make_scheduling_choices(
            nth_trips,
            scheduling_mode,
            probs_spec,
            probs_join_cols,
            depart_alt_base,
            first_trip_in_leg=first_trip_in_leg,
            report_failed_trips=is_last_iteration,
            trace_hh_id=trace_hh_id,
            trace_label=nth_trace_label,
        )

        # most initial departure (when no choice was made because all probs were zero)
        if is_last_iteration and (failfix == FAILFIX_CHOOSE_MOST_INITIAL):
            choices = choices.reindex(nth_trips.index)
            logger.warning(
                "%s coercing %s depart choices to most initial"
                % (nth_trace_label, choices.isna().sum())
            )
            choices = choices.fillna(trips[ADJUST_NEXT_DEPART_COL])

        # adjust allowed depart range of next trip
        has_next_trip = nth_trips.next_trip_id != NO_TRIP_ID
        if has_next_trip.any():
            next_trip_ids = nth_trips.next_trip_id[has_next_trip]
            # patch choice any trips with next_trips that weren't scheduled
            trips.loc[next_trip_ids, ADJUST_NEXT_DEPART_COL] = (
                choices.reindex(next_trip_ids.index)
                .fillna(trips[ADJUST_NEXT_DEPART_COL])
                .values
            )

        result_list.append(choices)

        chunk.log_df(trace_label, f"result_list", result_list)

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
    trace_hh_id,
    trace_label,
):

    set_tour_hour(trips_chunk, tours)
    set_stop_num(trips_chunk)

    # only non-initial trips require scheduling, segment handing first such trip in tour will use most space
    # is_outbound_chooser = (trips.trip_num > 1) & trips.outbound & (trips.primary_purpose != 'atwork')
    # is_inbound_chooser = (trips.trip_num < trips.trip_count) & ~trips.outbound & (trips.primary_purpose != 'atwork')
    # num_choosers = (is_inbound_chooser | is_outbound_chooser).sum()

    result_list = []

    if trips_chunk.outbound.any():
        leg_chunk = trips_chunk[trips_chunk.outbound]
        leg_trace_label = tracing.extend_trace_label(trace_label, "outbound")
        choices = schedule_trips_in_leg(
            outbound=True,
            trips=leg_chunk,
            probs_spec=probs_spec,
            model_settings=model_settings,
            is_last_iteration=is_last_iteration,
            trace_hh_id=trace_hh_id,
            trace_label=leg_trace_label,
        )
        result_list.append(choices)

        chunk.log_df(trace_label, f"result_list", result_list)

        # departure time of last outbound trips must constrain
        # departure times for initial inbound trips
        update_tour_earliest(trips_chunk, choices)

    if (~trips_chunk.outbound).any():
        leg_chunk = trips_chunk[~trips_chunk.outbound]
        leg_trace_label = tracing.extend_trace_label(trace_label, "inbound")
        choices = schedule_trips_in_leg(
            outbound=False,
            trips=leg_chunk,
            probs_spec=probs_spec,
            model_settings=model_settings,
            is_last_iteration=is_last_iteration,
            trace_hh_id=trace_hh_id,
            trace_label=leg_trace_label,
        )
        result_list.append(choices)

        chunk.log_df(trace_label, f"result_list", result_list)

    choices = pd.concat(result_list)

    return choices


@inject.step()
def trip_scheduling(trips, tours, chunk_size, trace_hh_id):

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
    model_settings_file_name = "trip_scheduling.yaml"
    model_settings = config.read_model_settings(model_settings_file_name)

    trips_df = trips.to_frame()
    tours = tours.to_frame()

    # add columns 'tour_hour', 'earliest', 'latest' to trips
    set_tour_hour(trips_df, tours)

    # trip_scheduling is a probabilistic model ane we don't support estimation,
    # but we do need to override choices in estimation mode
    estimator = estimation.manager.begin_estimation("trip_scheduling")
    if estimator:
        estimator.write_spec(model_settings, tag="PROBS_SPEC")
        estimator.write_model_settings(model_settings, model_settings_file_name)
        chooser_cols_for_estimation = [
            "person_id",
            "household_id",
            "tour_id",
            "trip_num",
            "trip_count",
            "primary_purpose",
            "outbound",
            "earliest",
            "latest",
            "tour_hour",
        ]
        estimator.write_choosers(trips_df[chooser_cols_for_estimation])

    probs_spec = pd.read_csv(
        config.config_file_path("trip_scheduling_probs.csv"), comment="#"
    )
    # FIXME for now, not really doing estimation for probabilistic model - just overwriting choices
    # besides, it isn't clear that named coefficients would be helpful if we had some form of estimation
    # coefficients_df = simulate.read_model_coefficients(model_settings)
    # probs_spec = map_coefficients(probs_spec, coefficients_df)

    # add tour-based chunk_id so we can chunk all trips in tour together
    trips_df["chunk_id"] = reindex(
        pd.Series(list(range(len(tours))), tours.index), trips_df.tour_id
    )

    assert "DEPART_ALT_BASE" in model_settings
    failfix = model_settings.get(FAILFIX, FAILFIX_DEFAULT)

    max_iterations = model_settings.get("MAX_ITERATIONS", 1)
    assert max_iterations > 0

    choices_list = []

    for (
        chunk_i,
        trips_chunk,
        chunk_trace_label,
    ) in chunk.adaptive_chunked_choosers_by_chunk_id(
        trips_df, chunk_size, trace_label, trace_label
    ):

        i = 0
        while (i < max_iterations) and not trips_chunk.empty:

            # only chunk log first iteration since memory use declines with each iteration
            with chunk.chunk_log(trace_label) if i == 0 else chunk.chunk_log_skip():

                i += 1
                is_last_iteration = i == max_iterations

                trace_label_i = tracing.extend_trace_label(trace_label, "i%s" % i)
                logger.info(
                    "%s scheduling %s trips within chunk %s",
                    trace_label_i,
                    trips_chunk.shape[0],
                    chunk_i,
                )

                choices = run_trip_scheduling(
                    trips_chunk,
                    tours,
                    probs_spec,
                    model_settings,
                    estimator=estimator,
                    is_last_iteration=is_last_iteration,
                    chunk_size=chunk_size,
                    trace_hh_id=trace_hh_id,
                    trace_label=trace_label_i,
                )

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
        choices = estimator.get_survey_values(
            choices, "trips", "depart"
        )  # override choices
        estimator.write_override_choices(choices)
        estimator.end_estimation()
        assert not choices.isnull().any()

    if choices.isnull().any():
        logger.warning(
            "%s of %s trips could not be scheduled after %s iterations"
            % (choices.isnull().sum(), trips_df.shape[0], i)
        )

        if failfix != FAILFIX_DROP_AND_CLEANUP:
            raise RuntimeError(
                "%s setting '%s' not enabled in settings"
                % (FAILFIX, FAILFIX_DROP_AND_CLEANUP)
            )

        trips_df["failed"] = choices.isnull()
        trips_df = cleanup_failed_trips(trips_df)
        choices = choices.reindex(trips_df.index)

    trips_df["depart"] = choices

    assert not trips_df.depart.isnull().any()

    pipeline.replace_table("trips", trips_df)
