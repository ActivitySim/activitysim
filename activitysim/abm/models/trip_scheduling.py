# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging
import warnings
from builtins import range
from typing import Any, List, Literal

import numpy as np
import pandas as pd

from activitysim.abm.models.util import probabilistic_scheduling as ps
from activitysim.abm.models.util.school_escort_tours_trips import (
    split_out_school_escorting_trips,
)
from activitysim.abm.models.util.trip import cleanup_failed_trips, failed_trip_cohorts
from activitysim.core import chunk, config, estimation, expressions, tracing, workflow
from activitysim.core.configuration.base import PreprocessorSettings, PydanticReadable
from activitysim.core.util import reindex

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
RELATIVE_MODE = "relative"
PROBS_JOIN_COLUMNS_DEPARTURE_BASED: list[str] = [
    "primary_purpose",
    "outbound",
    "tour_hour",
    "trip_num",
]
PROBS_JOIN_COLUMNS_DURATION_BASED: list[str] = ["outbound", "stop_num"]
PROBS_JOIN_COLUMNS_RELATIVE_BASED: list[str] = ["outbound", "periods_left"]


def _logic_version(model_settings: TripSchedulingSettings):
    logic_version = model_settings.logic_version
    if logic_version is None:
        warnings.warn(
            "The trip_scheduling component now has a logic_version setting "
            "to control how the scheduling rules are applied.  The default "
            "logic_version is currently set at `1` but may be moved up in "
            "the future. Explicitly set `logic_version` to 2 in the model "
            "settings to upgrade your model logic now, or set it to 1 to "
            "suppress this message.",
            FutureWarning,
        )
        logic_version = 1
    return logic_version


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


def update_tour_earliest(trips, outbound_choices, logic_version: int):
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
    logic_version : int
        Logic version 1 is the original ActivitySim implementation, which
        sets the "earliest" value to the max outbound departure for all
        inbound trips, regardless of what that max outbound departure value
        is (even if it is NA).  Logic version 2 introduces a change whereby
        that assignment is only made if the max outbound departure value is
        not NA.

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
    # FIXME - extra logic added because max_outbound_departure can be NA if previous failed trip was removed
    if logic_version == 1:
        tmp_trips["earliest"] = tmp_trips["earliest"].where(
            tmp_trips["outbound"], tmp_trips["max_outbound_departure"]
        )
    elif logic_version > 1:
        tmp_trips["earliest"] = np.where(
            ~tmp_trips["outbound"] & ~tmp_trips["max_outbound_departure"].isna(),
            tmp_trips["max_outbound_departure"],
            tmp_trips["earliest"],
        )
    else:
        raise ValueError(f"bad logic_version: {logic_version}")

    trips["earliest"] = tmp_trips["earliest"].reindex(trips.index)

    return


def schedule_trips_in_leg(
    state: workflow.State,
    outbound,
    trips,
    probs_spec,
    model_settings: TripSchedulingSettings,
    is_last_iteration,
    trace_label,
    *,
    chunk_sizer: chunk.ChunkSizer,
):
    """

    Parameters
    ----------
    state
    outbound
    trips
    probs_spec
    depart_alt_base
    is_last_iteration
    trace_label

    Returns
    -------
    choices: pd.Series
        depart choice for trips, indexed by trip_id
    """

    failfix = model_settings.FAILFIX
    depart_alt_base = model_settings.DEPART_ALT_BASE
    scheduling_mode = model_settings.scheduling_mode
    preprocessor_settings = model_settings.preprocessor

    probs_join_cols = model_settings.probs_join_cols
    if probs_join_cols is None:
        if scheduling_mode == "departure":
            probs_join_cols = PROBS_JOIN_COLUMNS_DEPARTURE_BASED
        elif scheduling_mode == "stop_duration":
            probs_join_cols = PROBS_JOIN_COLUMNS_DURATION_BASED
        elif scheduling_mode == "relative":
            probs_join_cols = PROBS_JOIN_COLUMNS_RELATIVE_BASED
        else:
            logger.error(
                "Invalid scheduling mode specified: {0}.".format(scheduling_mode),
                "Please select one of ['departure', 'stop_duration', 'relative'] and try again.",
            )
            raise ValueError(f"Invalid scheduling mode specified: {scheduling_mode}")

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

    network_los = state.get_injectable("network_los")
    locals_dict = {"network_los": network_los}
    locals_dict.update(config.get_model_constants(model_settings))

    first_trip_in_leg = True
    for i in range(trips.trip_num.min(), trips.trip_num.max() + 1):
        nth_trace_label = tracing.extend_trace_label(trace_label, "num_%s" % i)

        # - annotate trips
        if preprocessor_settings:
            expressions.assign_columns(
                state,
                df=trips,
                model_settings=preprocessor_settings,
                locals_dict=locals_dict,
                trace_label=nth_trace_label,
            )

        if (
            outbound
            or (scheduling_mode == DURATION_MODE)
            or (scheduling_mode == RELATIVE_MODE)
        ):
            # iterate in ascending trip_num order
            nth_trips = trips[trips.trip_num == i]
        else:
            # iterate over inbound trips in descending trip_num order, skipping the final trip
            nth_trips = trips[trips.trip_num == trips.trip_count - i]

        choices = ps.make_scheduling_choices(
            state,
            nth_trips,
            scheduling_mode,
            probs_spec,
            probs_join_cols,
            depart_alt_base,
            first_trip_in_leg=first_trip_in_leg,
            report_failed_trips=is_last_iteration,
            trace_label=nth_trace_label,
            chunk_sizer=chunk_sizer,
        )

        # most initial departure (when no choice was made because all probs were zero)
        if is_last_iteration and (failfix == FAILFIX_CHOOSE_MOST_INITIAL):
            choices = choices.reindex(nth_trips.index)
            logger.warning(
                "%s coercing %s depart choices to most initial"
                % (nth_trace_label, choices.isna().sum())
            )
            choices = choices.fillna(trips[ADJUST_NEXT_DEPART_COL])

        if scheduling_mode == RELATIVE_MODE:
            # choices are relative to the previous departure time
            choices = nth_trips.earliest + choices
            # need to update the departure time based on the choice
            logic_version = _logic_version(model_settings)
            if logic_version == 1:
                raise ValueError(
                    "cannot use logic version 1 with 'relative' scheduling mode"
                )
            update_tour_earliest(trips, choices, logic_version)

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

        chunk_sizer.log_df(trace_label, "result_list", result_list)

        first_trip_in_leg = False

    if len(result_list) > 1:
        choices = pd.concat(result_list)

    return choices


def run_trip_scheduling(
    state: workflow.State,
    trips_chunk,
    tours,
    probs_spec,
    model_settings,
    estimator,
    is_last_iteration,
    trace_label,
    *,
    chunk_sizer: chunk.ChunkSizer,
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
            state,
            outbound=True,
            trips=leg_chunk,
            probs_spec=probs_spec,
            model_settings=model_settings,
            is_last_iteration=is_last_iteration,
            trace_label=leg_trace_label,
            chunk_sizer=chunk_sizer,
        )
        result_list.append(choices)

        chunk_sizer.log_df(trace_label, "result_list", result_list)

        # departure time of last outbound trips must constrain
        # departure times for initial inbound trips
        update_tour_earliest(trips_chunk, choices, _logic_version(model_settings))

    if (~trips_chunk.outbound).any():
        leg_chunk = trips_chunk[~trips_chunk.outbound]
        leg_trace_label = tracing.extend_trace_label(trace_label, "inbound")
        choices = schedule_trips_in_leg(
            state,
            outbound=False,
            trips=leg_chunk,
            probs_spec=probs_spec,
            model_settings=model_settings,
            is_last_iteration=is_last_iteration,
            trace_label=leg_trace_label,
            chunk_sizer=chunk_sizer,
        )
        result_list.append(choices)

        chunk_sizer.log_df(trace_label, "result_list", result_list)

    choices = pd.concat(result_list)

    return choices


class TripSchedulingSettings(PydanticReadable):
    """
    Settings for the `trip_scheduling` component.
    """

    PROBS_SPEC: str = "trip_scheduling_probs.csv"
    """Filename for the trip scheduling probabilities (.csv) file."""

    COEFFICIENTS: str = "trip_scheduling_coefficients.csv"
    """Filename for the trip scheduling coefficients file"""

    FAILFIX: str = "choose_most_initial"
    """ """

    MAX_ITERATIONS: int = 1
    """Maximum iterations."""

    DEPART_ALT_BASE: int = 5
    """Integer to add to probs column index to get time period it represents.
    e.g. depart_alt_base = 5 means first column (column 0) represents 5 am"""

    scheduling_mode: Literal["departure", "stop_duration", "relative"] = "departure"

    probs_join_cols: list[str] | None = None

    preprocessor: PreprocessorSettings | None = None

    logic_version: int | None = None

    CONSTANTS: dict[str, Any] = {}


@workflow.step(copy_tables=False)
def trip_scheduling(
    state: workflow.State,
    trips: pd.DataFrame,
    tours: pd.DataFrame,
    model_settings: TripSchedulingSettings | None = None,
    model_settings_file_name: str = "trip_scheduling.yaml",
    trace_label: str = "trip_scheduling",
) -> None:
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

    if model_settings is None:
        model_settings = TripSchedulingSettings.read_settings_file(
            state.filesystem,
            model_settings_file_name,
        )

    trips_df = trips.copy()

    if state.is_table("school_escort_trips"):
        school_escort_trips = state.get_dataframe("school_escort_trips")
        # separate out school escorting trips to exclude them from the model and estimation data bundle
        trips_df, se_trips_df, full_trips_index = split_out_school_escorting_trips(
            trips_df, school_escort_trips
        )
        non_se_trips_df = trips_df

    # add columns 'tour_hour', 'earliest', 'latest' to trips
    set_tour_hour(trips_df, tours)

    # trip_scheduling is a probabilistic model ane we don't support estimation,
    # but we do need to override choices in estimation mode
    estimator = estimation.manager.begin_estimation(state, "trip_scheduling")
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

    probs_spec_file = model_settings.PROBS_SPEC
    probs_spec = pd.read_csv(
        state.filesystem.get_config_file_path(probs_spec_file), comment="#"
    )
    # FIXME for now, not really doing estimation for probabilistic model - just overwriting choices
    # besides, it isn't clear that named coefficients would be helpful if we had some form of estimation
    # coefficients_df = state.filesystem.read_model_coefficients(model_settings)
    # probs_spec = map_coefficients(probs_spec, coefficients_df)

    # add tour-based chunk_id so we can chunk all trips in tour together
    trips_df["chunk_id"] = reindex(
        pd.Series(list(range(len(tours))), tours.index), trips_df.tour_id
    )

    failfix = model_settings.FAILFIX

    max_iterations = model_settings.MAX_ITERATIONS
    assert max_iterations > 0

    choices_list = []

    for (
        chunk_i,
        trips_chunk,
        chunk_trace_label,
        chunk_sizer,
    ) in chunk.adaptive_chunked_choosers_by_chunk_id(
        state, trips_df, trace_label, trace_label
    ):
        i = 0
        while (i < max_iterations) and not trips_chunk.empty:
            # only chunk log first iteration since memory use declines with each iteration
            with chunk.chunk_log(
                state, trace_label
            ) if i == 0 else chunk.chunk_log_skip():
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
                    state,
                    trips_chunk,
                    tours,
                    probs_spec,
                    model_settings,
                    estimator=estimator,
                    is_last_iteration=is_last_iteration,
                    trace_label=trace_label_i,
                    chunk_sizer=chunk_sizer,
                )

                # boolean series of trips whose individual trip scheduling failed
                failed = choices.reindex(trips_chunk.index).isnull()
                logger.info("%s %s failed", trace_label_i, failed.sum())

                if (failed.sum() > 0) & (model_settings.scheduling_mode == "relative"):
                    raise RuntimeError("failed trips with relative scheduling mode")

                if not is_last_iteration:
                    # boolean series of trips whose leg scheduling failed
                    failed_cohorts = failed_trip_cohorts(trips_chunk, failed)
                    trips_chunk = trips_chunk[failed_cohorts]
                    choices = choices[~failed_cohorts]

                choices_list.append(choices)

    trips_df = trips.copy()

    if state.is_table("school_escort_trips"):
        # separate out school escorting trips to exclude them from the model and estimation data bundle
        trips_df, se_trips_df, full_trips_index = split_out_school_escorting_trips(
            trips_df, school_escort_trips
        )
        non_se_trips_df = trips_df

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
        trips_df = cleanup_failed_trips(state, trips_df)
        choices = choices.reindex(trips_df.index)

    trips_df["depart"] = choices

    if state.is_table("school_escort_trips"):
        # setting destination for school escort trips
        se_trips_df["depart"] = reindex(school_escort_trips.depart, se_trips_df.index)
        non_se_trips_df["depart"] = reindex(trips_df.depart, non_se_trips_df.index)
        # merge trips back together
        full_trips_df = pd.concat([non_se_trips_df, se_trips_df])
        full_trips_df["depart"] = full_trips_df["depart"].astype(int)
        # want to preserve the original order, but first need to remove trips that were dropped
        new_full_trips_index = full_trips_index[
            full_trips_index.isin(trips_df.index)
            | full_trips_index.isin(se_trips_df.index)
        ]
        trips_df = full_trips_df.reindex(new_full_trips_index)

    assert not trips_df.depart.isnull().any()

    state.add_table("trips", trips_df)
