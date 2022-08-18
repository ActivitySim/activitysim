import logging

import numpy as np
import pandas as pd

from activitysim.abm.models.util.trip import (
    generate_alternative_sizes,
    get_time_windows,
)
from activitysim.core import (
    chunk,
    config,
    expressions,
    inject,
    pipeline,
    simulate,
    tracing,
)
from activitysim.core.interaction_sample_simulate import _interaction_sample_simulate

logger = logging.getLogger(__name__)

TOUR_DURATION_COLUMN = "duration"
NUM_ALTERNATIVES = "num_alts"
MAIN_LEG_DURATION = "main_leg_duration"
IB_DURATION = "inbound_duration"
OB_DURATION = "outbound_duration"
NUM_OB_STOPS = "num_outbound_stops"
NUM_IB_STOPS = "num_inbound_stops"
HAS_OB_STOPS = "has_outbound_stops"
HAS_IB_STOPS = "has_inbound_stops"
LAST_OB_STOP = "last_outbound_stop"
FIRST_IB_STOP = "last_inbound_stop"

SCHEDULE_ID = "schedule_id"

OUTBOUND_FLAG = "outbound"

TEMP_COLS = [NUM_OB_STOPS, LAST_OB_STOP, NUM_IB_STOPS, FIRST_IB_STOP, NUM_ALTERNATIVES]


def generate_schedule_alternatives(tours):
    """
    For a set of tours, build out the potential schedule alternatives
    for the main leg, outbound leg, and inbound leg. This process handles
    the change in three steps.

    Definitions:
      - Main Leg: The time from last outbound stop to the first inbound stop.
                  If the tour does not include any intermediate stops this
                  will represent the full tour duration.
      - Outbound Leg: The time from the tour origin to the last outbound stop
      - Inbound Leg: The time from the first inbound stop to the tour origin

    1. For tours with no intermediate stops, it simple asserts a main leg
       duration equal to the tour duration.

    2. For tours with an intermediate stop on one of the legs, calculate
       all possible time combinations that are allowed in the duration

    3. For tours with an intermediate stop on both legs, calculate
       all possible time combinations that are allowed in the tour
       duration

    :param tours: pd.Dataframe: Must include a field for tour duration
            and boolean fields indicating intermediate inbound or outbound
            stops.
    :return: pd.Dataframe: Potential time duration windows.
    """
    assert set([NUM_IB_STOPS, NUM_OB_STOPS, TOUR_DURATION_COLUMN]).issubset(
        tours.columns
    )

    stop_pattern = tours[HAS_OB_STOPS].astype(int) + tours[HAS_IB_STOPS].astype(int)

    no_stops = no_stops_patterns(tours[stop_pattern == 0])
    one_way = stop_one_way_only_patterns(tours[stop_pattern == 1])
    two_way = stop_two_way_only_patterns(tours[stop_pattern > 1])

    schedules = pd.concat([no_stops, one_way, two_way], sort=True)
    schedules[SCHEDULE_ID] = np.arange(1, schedules.shape[0] + 1)

    return schedules


def no_stops_patterns(tours):
    """
    Asserts the tours with no intermediate stops have a main leg duration equal
    to the tour duration and set inbound and outbound windows equal to zero.
    :param tours: pd.Dataframe: Tours with no intermediate stops.
    :return: pd.Dataframe: Main leg duration, outbound leg duration, and inbound leg duration
    """
    alternatives = tours[[TOUR_DURATION_COLUMN]].rename(
        columns={TOUR_DURATION_COLUMN: MAIN_LEG_DURATION}
    )
    alternatives[[IB_DURATION, OB_DURATION]] = 0
    return alternatives.astype(int)


def stop_one_way_only_patterns(tours, travel_duration_col=TOUR_DURATION_COLUMN):
    """
    Calculates potential time windows for tours with a single leg with intermediate
    stops. It calculates all possibilities for the main leg and one tour leg to sum to
    the tour duration. The other leg is asserted with a duration of zero.
    :param tours: pd.Dataframe: Tours with no intermediate stops.
    :return: pd.Dataframe: Main leg duration, outbound leg duration, and inbound leg duration
            The return dataframe is indexed to the tour input index
    """
    if tours.empty:
        return None

    assert travel_duration_col in tours.columns

    indexes, patterns, pattern_sizes = get_pattern_index_and_arrays(
        tours.index, tours[travel_duration_col], one_way=True
    )
    direction = np.repeat(tours[HAS_OB_STOPS], pattern_sizes)

    inbound = np.where(direction == 0, patterns[:, 1], 0)
    outbound = np.where(direction == 1, patterns[:, 1], 0)

    patterns = pd.DataFrame(
        index=indexes,
        data=np.column_stack((patterns[:, 0], outbound, inbound)),
        columns=[MAIN_LEG_DURATION, OB_DURATION, IB_DURATION],
    )
    patterns.index.name = tours.index.name

    return patterns


def stop_two_way_only_patterns(tours, travel_duration_col=TOUR_DURATION_COLUMN):
    """
    Calculates potential time windows for tours with intermediate stops on both
    legs. It calculates all possibilities for the main leg and both tour legs to
    sum to the tour duration.
    :param tours: pd.Dataframe: Tours with no intermediate stops.
    :return: pd.Dataframe: Main leg duration, outbound leg duration, and inbound leg duration
            The return dataframe is indexed to the tour input index
    """
    if tours.empty:
        return None

    assert travel_duration_col in tours.columns

    indexes, patterns, _ = get_pattern_index_and_arrays(
        tours.index, tours[travel_duration_col], one_way=False
    )

    patterns = pd.DataFrame(
        index=indexes,
        data=patterns,
        columns=[MAIN_LEG_DURATION, OB_DURATION, IB_DURATION],
    )
    patterns.index.name = tours.index.name

    return patterns


def get_pattern_index_and_arrays(tour_indexes, durations, one_way=True):
    """
    A helper method to quickly calculate all of the potential time windows
    for a given set of tour indexes and durations.
    :param tour_indexes: List of tour indexes
    :param durations: List of tour durations
    :param one_way: If True, calculate windows for only one tour leg. If False,
                    calculate tour windows for both legs
    :return: np.array: Tour indexes repeated for valid pattern
             np.array: array with a column for main tour leg, outbound leg, and inbound leg
             np.array: array with the number of patterns for each tour
    """
    max_columns = 2 if one_way else 3
    max_duration = np.max(durations)
    time_windows = get_time_windows(max_duration, max_columns)

    patterns = []
    pattern_sizes = []

    for duration in durations:
        possible_windows = time_windows[
            :max_columns, np.where(time_windows.sum(axis=0) == duration)[0]
        ]
        possible_windows = np.unique(possible_windows, axis=1).transpose()
        patterns.append(possible_windows)
        pattern_sizes.append(possible_windows.shape[0])

    indexes = np.repeat(tour_indexes, pattern_sizes)

    patterns = np.concatenate(patterns)
    # If we've done everything right, the indexes
    # calculated above should be the same length as
    # the pattern options
    assert patterns.shape[0] == len(indexes)

    return indexes, patterns, pattern_sizes


def get_spec_for_segment(model_settings, spec_name, segment):
    """
    Read in the model spec
    :param model_settings: model settings file
    :param spec_name: name of the key in the settings file
    :param segment: which segment of the spec file do you want to read
    :return: array of utility equations
    """

    omnibus_spec = simulate.read_model_spec(file_name=model_settings[spec_name])

    spec = omnibus_spec[[segment]]

    # might as well ignore any spec rows with 0 utility
    spec = spec[spec.iloc[:, 0] != 0]
    assert spec.shape[0] > 0

    return spec


def run_trip_scheduling_choice(
    spec, tours, skims, locals_dict, chunk_size, trace_hh_id, trace_label
):

    NUM_TOUR_LEGS = 3
    trace_label = tracing.extend_trace_label(trace_label, "interaction_sample_simulate")

    # FIXME: The duration, start, and end should be ints well before we get here...
    tours[TOUR_DURATION_COLUMN] = tours[TOUR_DURATION_COLUMN].astype(np.int8)

    # Setup boolean columns to make it easier to identify
    # intermediate stops later in the model.
    tours[HAS_OB_STOPS] = tours[NUM_OB_STOPS] >= 1
    tours[HAS_IB_STOPS] = tours[NUM_IB_STOPS] >= 1

    # Calculate a matrix with the appropriate alternative sizes
    # based on the total tour duration. This is used to calculate
    # chunk sizes.
    max_duration = tours[TOUR_DURATION_COLUMN].max()
    alt_sizes = generate_alternative_sizes(max_duration, NUM_TOUR_LEGS)

    # Assert the number of tour leg schedule alternatives for each tour
    tours[NUM_ALTERNATIVES] = 1
    tours.loc[tours[HAS_OB_STOPS] != tours[HAS_IB_STOPS], NUM_ALTERNATIVES] = (
        tours[TOUR_DURATION_COLUMN] + 1
    )
    tours.loc[
        tours[HAS_OB_STOPS] & tours[HAS_IB_STOPS], NUM_ALTERNATIVES
    ] = tours.apply(lambda x: alt_sizes[1, x.duration], axis=1)

    # If no intermediate stops on the tour, then then main leg duration
    # equals the tour duration and the intermediate durations are zero
    tours.loc[~tours[HAS_OB_STOPS] & ~tours[HAS_IB_STOPS], MAIN_LEG_DURATION] = tours[
        TOUR_DURATION_COLUMN
    ]
    tours.loc[
        ~tours[HAS_OB_STOPS] & ~tours[HAS_IB_STOPS], [IB_DURATION, OB_DURATION]
    ] = 0

    # We only need to determine schedules for tours with intermediate stops
    indirect_tours = tours.loc[tours[HAS_OB_STOPS] | tours[HAS_IB_STOPS]]

    if len(indirect_tours) > 0:

        # Iterate through the chunks
        result_list = []
        for i, choosers, chunk_trace_label in chunk.adaptive_chunked_choosers(
            indirect_tours, chunk_size, trace_label
        ):

            # Sort the choosers and get the schedule alternatives
            choosers = choosers.sort_index()
            schedules = generate_schedule_alternatives(choosers).sort_index()

            # Assuming we did the max_alt_size calculation correctly,
            # we should get the same sizes here.
            assert choosers[NUM_ALTERNATIVES].sum() == schedules.shape[0]

            # Run the simulation
            choices = _interaction_sample_simulate(
                choosers=choosers,
                alternatives=schedules,
                spec=spec,
                choice_column=SCHEDULE_ID,
                allow_zero_probs=True,
                zero_prob_choice_val=-999,
                log_alt_losers=False,
                want_logsums=False,
                skims=skims,
                locals_d=locals_dict,
                trace_label=chunk_trace_label,
                trace_choice_name="trip_schedule_stage_1",
                estimator=None,
            )

            assert len(choices.index) == len(choosers.index)

            choices = schedules[schedules[SCHEDULE_ID].isin(choices)]

            result_list.append(choices)

            chunk.log_df(trace_label, f"result_list", result_list)

        # FIXME: this will require 2X RAM
        # if necessary, could append to hdf5 store on disk:
        # http://pandas.pydata.org/pandas-docs/stable/io.html#id2
        if len(result_list) > 1:
            choices = pd.concat(result_list)

        assert len(choices.index) == len(indirect_tours.index)

        # The choices here are only the indirect tours, so the durations
        # need to be updated on the main tour dataframe.
        tours.update(choices[[MAIN_LEG_DURATION, OB_DURATION, IB_DURATION]])

    # Cleanup data types and drop temporary columns
    tours[[MAIN_LEG_DURATION, OB_DURATION, IB_DURATION]] = tours[
        [MAIN_LEG_DURATION, OB_DURATION, IB_DURATION]
    ].astype(np.int8)
    tours = tours.drop(columns=TEMP_COLS)

    return tours


@inject.step()
def trip_scheduling_choice(trips, tours, skim_dict, chunk_size, trace_hh_id):

    trace_label = "trip_scheduling_choice"
    model_settings = config.read_model_settings("trip_scheduling_choice.yaml")
    spec = get_spec_for_segment(model_settings, "SPECIFICATION", "stage_one")

    trips_df = trips.to_frame()
    tours_df = tours.to_frame()

    outbound_trips = trips_df[trips_df[OUTBOUND_FLAG]]
    inbound_trips = trips_df[~trips_df[OUTBOUND_FLAG]]

    last_outbound_trip = trips_df.loc[
        outbound_trips.groupby("tour_id")["trip_num"].idxmax()
    ]
    first_inbound_trip = trips_df.loc[
        inbound_trips.groupby("tour_id")["trip_num"].idxmin()
    ]

    tours_df[NUM_OB_STOPS] = (
        outbound_trips.groupby("tour_id").size().reindex(tours.index) - 1
    )
    tours_df[NUM_IB_STOPS] = (
        inbound_trips.groupby("tour_id").size().reindex(tours.index) - 1
    )
    tours_df[LAST_OB_STOP] = (
        last_outbound_trip[["tour_id", "origin"]]
        .set_index("tour_id")
        .reindex(tours.index)
    )
    tours_df[FIRST_IB_STOP] = (
        first_inbound_trip[["tour_id", "destination"]]
        .set_index("tour_id")
        .reindex(tours.index)
    )

    preprocessor_settings = model_settings.get("PREPROCESSOR", None)

    if preprocessor_settings:
        # hack: preprocessor adds origin column in place if it does not exist already
        od_skim_stack_wrapper = skim_dict.wrap("origin", "destination")
        do_skim_stack_wrapper = skim_dict.wrap("destination", "origin")
        obib_skim_stack_wrapper = skim_dict.wrap(LAST_OB_STOP, FIRST_IB_STOP)

        skims = [od_skim_stack_wrapper, do_skim_stack_wrapper, obib_skim_stack_wrapper]

        locals_dict = {
            "od_skims": od_skim_stack_wrapper,
            "do_skims": do_skim_stack_wrapper,
            "obib_skims": obib_skim_stack_wrapper,
        }

        simulate.set_skim_wrapper_targets(tours_df, skims)

        expressions.assign_columns(
            df=tours_df,
            model_settings=preprocessor_settings,
            locals_dict=locals_dict,
            trace_label=trace_label,
        )

    tours_df = run_trip_scheduling_choice(
        spec, tours_df, skims, locals_dict, chunk_size, trace_hh_id, trace_label
    )

    pipeline.replace_table("tours", tours_df)
