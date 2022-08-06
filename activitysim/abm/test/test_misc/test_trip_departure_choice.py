import numpy as np
import pandas as pd
import pytest

import activitysim.abm.models.trip_departure_choice as tdc
from activitysim.abm.models.util.trip import get_time_windows
from activitysim.core import los

from .setup_utils import setup_dirs


@pytest.fixture(scope="module")
def trips():
    outbound_array = [True, True, False, False, False, True, True, False, False, True]

    trips = pd.DataFrame(
        data={
            "tour_id": [1, 1, 2, 2, 2, 2, 2, 3, 3, 4],
            "trip_duration": [2, 2, 7, 7, 7, 12, 12, 4, 4, 5],
            "inbound_duration": [0, 0, 7, 7, 7, 0, 0, 4, 4, 5],
            "main_leg_duration": [4, 4, 2, 2, 2, 2, 2, 1, 1, 2],
            "outbound_duration": [2, 2, 0, 0, 0, 12, 12, 0, 0, 5],
            "trip_count": [2, 2, 3, 3, 3, 2, 2, 2, 2, 1],
            "trip_num": [1, 2, 1, 2, 3, 1, 2, 1, 2, 1],
            "outbound": outbound_array,
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
    departures = tdc.apply_stage_two_model(model_spec, trips, 0, "TEST Trip Departure")
    assert len(departures) == len(trips)
    pd.testing.assert_index_equal(departures.index, trips.index)

    departures = pd.concat([trips, departures], axis=1)
