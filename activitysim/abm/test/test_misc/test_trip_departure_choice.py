import numpy as np
import pandas as pd
import pytest
import os

import activitysim.abm.models.trip_departure_choice as tdc
from activitysim.abm.models.util.trip import get_time_windows
from activitysim.core import workflow

from .setup_utils import setup_dirs


@pytest.fixture(scope="module")
def trips():
    trips = pd.DataFrame(
        data={
            "tour_id": [1, 1, 2, 2, 2, 2, 2, 3, 3, 4],
            "trip_duration": [2, 2, 7, 7, 7, 12, 12, 4, 4, 5],
            "inbound_duration": [0, 0, 7, 7, 7, 0, 0, 4, 4, 5],
            "main_leg_duration": [4, 4, 2, 2, 2, 2, 2, 1, 1, 2],
            "outbound_duration": [2, 2, 0, 0, 0, 12, 12, 0, 0, 5],
            "trip_count": [2, 2, 3, 3, 3, 2, 2, 2, 2, 1],
            "trip_num": [1, 2, 1, 2, 3, 1, 2, 1, 2, 1],
            "outbound": [
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                False,
                False,
                True,
            ],
            "chunk_id": [1, 1, 2, 2, 2, 2, 2, 3, 3, 4],
            "is_work": [
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
            ],
            "is_school": [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                False,
            ],
            "is_eatout": [
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
            ],
            "start": [8, 8, 18, 18, 18, 18, 18, 24, 24, 19],
            "end": [14, 14, 39, 39, 39, 39, 39, 29, 29, 26],
            "origin": [3, 5, 15, 12, 24, 8, 17, 8, 9, 6],
            "destination": [5, 9, 12, 24, 20, 17, 18, 9, 11, 14],
        },
        index=range(10),
    )

    trips.index.name = "trip_id"
    return trips


@pytest.fixture(scope="module")
def settings():
    return {
        "skims_file": "skims.omx",
        "skim_time_periods": {"labels": ["EA", "AM", "MD", "PM", "NT"]},
    }


def add_canonical_dirs(configs_dir_name):
    state = workflow.State()
    configs_dir = os.path.join(os.path.dirname(__file__), f"{configs_dir_name}")
    data_dir = os.path.join(os.path.dirname(__file__), f"data")
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    state.initialize_filesystem(
        working_dir=os.path.dirname(__file__),
        configs_dir=(configs_dir,),
        output_dir=output_dir,
        data_dir=(data_dir,),
    )
    return state


@pytest.fixture(scope="module")
def model_spec():
    index = [
        "@(df['stop_time_duration'] * df['is_work'].astype(int)).astype(int)",
        "@(df['stop_time_duration'] * df['is_school'].astype(int)).astype(int)",
        "@(df['stop_time_duration'] * df['is_eatout'].astype(int)).astype(int)",
    ]

    values = {
        "inbound": [0.933020, 0.370260, 0.994840],
        "outbound": [0.933020, 0.370260, 0.994840],
    }

    return pd.DataFrame(index=index, data=values)


def test_build_patterns(trips):
    time_windows = get_time_windows(48, 3)
    patterns = tdc.build_patterns(trips, time_windows)
    patterns = patterns.sort_values(["tour_id", "outbound", "trip_num"])

    assert patterns.shape[0] == 34
    assert patterns.shape[1] == 6
    assert patterns.index.name == tdc.TOUR_LEG_ID

    output_columns = [
        tdc.TOUR_ID,
        tdc.PATTERN_ID,
        tdc.TRIP_NUM,
        tdc.STOP_TIME_DURATION,
        tdc.TOUR_ID,
        tdc.OUTBOUND,
    ]

    assert set(output_columns).issubset(patterns.columns)


def test_get_tour_legs(trips):
    tour_legs = tdc.get_tour_legs(trips)
    assert tour_legs.index.name == tdc.TOUR_LEG_ID
    assert (
        np.unique(tour_legs[tdc.TOUR_ID].values).shape[0]
        == np.unique(trips[tdc.TOUR_ID].values).shape[0]
    )


def test_generate_alternative(trips):
    alts = tdc.generate_alternatives(trips, tdc.STOP_TIME_DURATION)
    assert alts.shape[0] == 67
    assert alts.shape[1] == 1

    assert alts.index.name == tdc.TRIP_ID
    assert alts.columns[0] == tdc.STOP_TIME_DURATION

    pd.testing.assert_series_equal(
        trips.groupby(trips.index)["trip_duration"].max(),
        alts.groupby(alts.index)[tdc.STOP_TIME_DURATION].max(),
        check_names=False,
    )


def test_apply_stage_two_model(model_spec, trips):
    setup_dirs()
    state = add_canonical_dirs("configs_test_misc").default_settings()

    # A settings object is needed to pass to the model application function,
    # but for testing we can just use the default settings.
    # In non-testing use cases, the SPEC would actually be read from the yaml file
    # instead of being passed directly as a dataframe.
    model_settings = tdc.TripDepartureChoiceSettings()

    departures = tdc.apply_stage_two_model(
        state,
        model_spec,
        trips,
        0,
        "TEST Trip Departure",
        model_settings=model_settings,
    )
    assert len(departures) == len(trips)
    pd.testing.assert_index_equal(departures.index, trips.index)

    departures = pd.concat([trips, departures], axis=1)


def test_tdc_explicit_error_terms_parity(model_spec, trips):
    setup_dirs()
    model_settings = tdc.TripDepartureChoiceSettings()

    # Increase population for statistical convergence
    large_trips = pd.concat([trips] * 500).reset_index(drop=True)
    large_trips.index.name = "trip_id"
    # Ensure tour_ids are distinct for the expanded set
    large_trips["tour_id"] = (
        large_trips.groupby("tour_id").cumcount() * 1000 + large_trips["tour_id"]
    )

    # Trip departure choice uses tour_leg_id as the random channel index
    tour_legs = tdc.get_tour_legs(large_trips)

    # Run without explicit error terms
    state_no_eet = add_canonical_dirs("configs_test_misc").default_settings()
    state_no_eet.settings.use_explicit_error_terms = False
    state_no_eet.rng().set_base_seed(42)
    state_no_eet.rng().begin_step("test_no_eet")
    state_no_eet.rng().add_channel("trip_id", large_trips)
    state_no_eet.rng().add_channel("tour_leg_id", tour_legs)

    departures_no_eet = tdc.apply_stage_two_model(
        state_no_eet,
        model_spec,
        large_trips,
        0,
        "TEST Trip Departure No EET",
        model_settings=model_settings,
    )

    # Run with explicit error terms
    state_eet = add_canonical_dirs("configs_test_misc").default_settings()
    state_eet.settings.use_explicit_error_terms = True
    state_eet.rng().set_base_seed(42)
    state_eet.rng().begin_step("test_eet")
    state_eet.rng().add_channel("trip_id", large_trips)
    state_eet.rng().add_channel("tour_leg_id", tour_legs)

    departures_eet = tdc.apply_stage_two_model(
        state_eet,
        model_spec,
        large_trips,
        0,
        "TEST Trip Departure EET",
        model_settings=model_settings,
    )

    # Compare distributions
    dist_no_eet = departures_no_eet.value_counts(normalize=True).sort_index()
    dist_eet = departures_eet.value_counts(normalize=True).sort_index()

    # Check that they are reasonably close (within 5% for this sample size)
    pd.testing.assert_series_equal(dist_no_eet, dist_eet, atol=0.05, check_names=False)
