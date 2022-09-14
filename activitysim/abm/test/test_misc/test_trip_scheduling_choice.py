import numpy as np
import pandas as pd
import pytest

from activitysim.abm.models import trip_scheduling_choice as tsc
from activitysim.abm.tables.skims import skim_dict
from activitysim.core import los

from .setup_utils import setup_dirs


@pytest.fixture(scope="module")
def tours():
    tours = pd.DataFrame(
        data={
            "duration": [2, 44, 32, 12, 11, 16],
            "num_outbound_stops": [2, 4, 0, 0, 1, 3],
            "num_inbound_stops": [1, 0, 0, 2, 1, 2],
            "tour_type": ["othdisc"] * 2 + ["eatout"] * 4,
            "origin": [3, 10, 15, 23, 5, 8],
            "destination": [5, 9, 12, 24, 20, 17],
            tsc.LAST_OB_STOP: [1, 3, 0, 0, 12, 14],
            tsc.FIRST_IB_STOP: [2, 0, 0, 4, 6, 20],
        },
        index=range(6),
    )

    tours.index.name = "tour_id"

    tours[tsc.HAS_OB_STOPS] = tours[tsc.NUM_OB_STOPS] >= 1
    tours[tsc.HAS_IB_STOPS] = tours[tsc.NUM_IB_STOPS] >= 1

    return tours


@pytest.fixture(scope="module")
def settings():
    return {"skims_file": "skims.omx", "skim_time_periods": {"labels": ["MD"]}}


@pytest.fixture(scope="module")
def model_spec():
    index = [
        "@(df['main_leg_duration']>df['duration']).astype(int)",
        "@(df['main_leg_duration'] == 0)&(df['tour_type']=='othdiscr')",
        "@(df['main_leg_duration'] == 1)&(df['tour_type']=='othdiscr')",
        "@(df['main_leg_duration'] == 2)&(df['tour_type']=='othdiscr')",
        "@(df['main_leg_duration'] == 3)&(df['tour_type']=='othdiscr')",
        "@(df['main_leg_duration'] == 4)&(df['tour_type']=='othdiscr')",
        "@df['tour_type']=='othdiscr'",
        "@df['tour_type']=='eatout'",
        "@df['tour_type']=='eatout'",
    ]

    values = [
        -999,
        -6.5884,
        -5.0326,
        -2.0526,
        -1.0313,
        -0.46489,
        0.060382,
        -0.7508,
        0.53247,
    ]

    return pd.DataFrame(index=index, data=values, columns=["stage_one"])


@pytest.fixture(scope="module")
def skims(settings):
    setup_dirs()
    nw_los = los.Network_LOS()
    nw_los.load_data()
    skim_d = skim_dict(nw_los)

    od_skim_stack_wrapper = skim_d.wrap("origin", "destination")
    do_skim_stack_wrapper = skim_d.wrap("destination", "origin")
    obib_skim_stack_wrapper = skim_d.wrap(tsc.LAST_OB_STOP, tsc.FIRST_IB_STOP)

    skims = [od_skim_stack_wrapper, do_skim_stack_wrapper, obib_skim_stack_wrapper]

    return skims


@pytest.fixture(scope="module")
def locals_dict(skims):
    return {"od_skims": skims[0], "do_skims": skims[1], "obib_skims": skims[2]}


def test_generate_schedule_alternatives(tours):
    windows = tsc.generate_schedule_alternatives(tours)
    assert windows.shape[0] == 296
    assert windows.shape[1] == 4

    output_columns = [
        tsc.SCHEDULE_ID,
        tsc.MAIN_LEG_DURATION,
        tsc.OB_DURATION,
        tsc.IB_DURATION,
    ]

    assert set(output_columns).issubset(windows.columns)


def test_no_stops_patterns(tours):
    no_stops = tours[
        (tours["num_outbound_stops"] == 0) & (tours["num_inbound_stops"] == 0)
    ].copy()
    windows = tsc.no_stops_patterns(no_stops)

    assert windows.shape[0] == 1
    assert windows.shape[1] == 3

    output_columns = [tsc.MAIN_LEG_DURATION, tsc.OB_DURATION, tsc.IB_DURATION]

    assert set(output_columns).issubset(windows.columns)

    pd.testing.assert_series_equal(
        windows[tsc.MAIN_LEG_DURATION],
        no_stops["duration"],
        check_names=False,
        check_dtype=False,
    )
    assert windows[windows[tsc.IB_DURATION] > 0].empty
    assert windows[windows[tsc.OB_DURATION] > 0].empty


def test_one_way_stop_patterns(tours):
    one_way_stops = tours[
        (
            (tours["num_outbound_stops"] > 0).astype(int)
            + (tours["num_inbound_stops"] > 0).astype(int)
        )
        == 1
    ].copy()
    windows = tsc.stop_one_way_only_patterns(one_way_stops)

    assert windows.shape[0] == 58
    assert windows.shape[1] == 3

    output_columns = [tsc.MAIN_LEG_DURATION, tsc.OB_DURATION, tsc.IB_DURATION]

    assert set(output_columns).issubset(windows.columns)

    inbound_options = windows[(windows[tsc.IB_DURATION] > 0)]
    outbound_options = windows[windows[tsc.OB_DURATION] > 0]
    assert np.unique(inbound_options.index).shape[0] == 1
    assert np.unique(outbound_options.index).shape[0] == 1


def test_two_way_stop_patterns(tours):
    two_way_stops = tours[
        (
            (tours["num_outbound_stops"] > 0).astype(int)
            + (tours["num_inbound_stops"] > 0).astype(int)
        )
        == 2
    ].copy()
    windows = tsc.stop_two_way_only_patterns(two_way_stops)

    assert windows.shape[0] == 237
    assert windows.shape[1] == 3

    output_columns = [tsc.MAIN_LEG_DURATION, tsc.OB_DURATION, tsc.IB_DURATION]

    assert set(output_columns).issubset(windows.columns)


def test_run_trip_scheduling_choice(model_spec, tours, skims, locals_dict):
    """
    Test run the model.
    """

    out_tours = tsc.run_trip_scheduling_choice(
        model_spec, tours, skims, locals_dict, 2, None, "PyTest Trip Scheduling"
    )

    assert len(tours) == len(out_tours)
    pd.testing.assert_index_equal(
        tours.sort_index().index, out_tours.sort_index().index
    )

    output_columns = [tsc.MAIN_LEG_DURATION, tsc.OB_DURATION, tsc.IB_DURATION]

    assert set(output_columns).issubset(out_tours.columns)

    assert len(
        out_tours[
            out_tours[output_columns].sum(axis=1) == out_tours[tsc.TOUR_DURATION_COLUMN]
        ]
    ) == len(tours)
