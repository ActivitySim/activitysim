import numpy as np
import pandas as pd
import pytest
import os
from pathlib import Path


from activitysim.abm.models import trip_scheduling_choice as tsc
from activitysim.abm.tables.skims import skim_dict
from activitysim.core import los, workflow

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

    return pd.DataFrame(index=index, data=values, columns=["stage_one"]).rename_axis(
        "Expression"
    )


def add_canonical_dirs(configs_dir_name):
    state = workflow.State()
    configs_dir = os.path.join(os.path.dirname(__file__), f"{configs_dir_name}")
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    state.initialize_filesystem(
        working_dir=os.path.dirname(__file__),
        configs_dir=(configs_dir,),
        output_dir=output_dir,
        data_dir=(data_dir,),
    )
    return state


@pytest.fixture(scope="module")
def skims():
    state = add_canonical_dirs("configs_test_misc").default_settings()
    nw_los = los.Network_LOS(state, los_settings_file_name="settings_60_min.yaml")
    nw_los.load_data()
    skim_d = skim_dict(state, nw_los)

    od_skim_stack_wrapper = skim_d.wrap("origin", "destination")
    do_skim_stack_wrapper = skim_d.wrap("destination", "origin")
    obib_skim_stack_wrapper = skim_d.wrap(tsc.LAST_OB_STOP, tsc.FIRST_IB_STOP)

    skims = [od_skim_stack_wrapper, do_skim_stack_wrapper, obib_skim_stack_wrapper]

    return skims


@pytest.fixture(scope="module")
def locals_dict(skims):
    return {"od_skims": skims[0], "do_skims": skims[1], "obib_skims": skims[2]}


@pytest.fixture(scope="module")
def base_dir() -> Path:
    """
    A pytest fixture that returns the data folder location.
    :return: folder location for any necessary data to initialize the tests
    """
    return Path(__file__).parent


@pytest.fixture(scope="module")
def module() -> str:
    """
    A pytest fixture that returns the data folder location.
    :return: folder location for any necessary data to initialize the tests
    """
    return "summarize"


# Used by conftest.py initialize_pipeline method
@pytest.fixture(scope="module")
def tables() -> dict[str, str]:
    """
    A pytest fixture that returns the "mock" tables to build pipeline dataframes. The
    key-value pair is the name of the table and the index column.
    :return: dict
    """
    return {
        "land_use": "zone_id",
        "tours": "tour_id",
        "trips": "trip_id",
        "persons": "person_id",
        "households": "household_id",
    }


# Used by conftest.py initialize_pipeline method
# Set to true if you need to read skims into the pipeline
@pytest.fixture(scope="module")
def initialize_network_los() -> bool:
    """
    A pytest boolean fixture indicating whether network skims should be read from the
    fixtures test data folder.
    :return: bool
    """
    return True


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
    # create a temporary workflow state with no content
    state = workflow.State.make_temp()

    # Define model settings for this test.
    # The settings for this model requires a filename for the spec, but in this test we
    # are passing the spec dataframe directly, so the filename is just a placeholder.
    # In non-testing use cases, the SPEC would actually be read from the yaml file
    # instead of being passed directly as a dataframe.
    model_settings = tsc.TripSchedulingChoiceSettings(
        **{
            "SPEC": "placeholder.csv",
            "compute_settings": {
                "protect_columns": ["origin", "destination", "schedule_id"]
            },
        }
    )

    # As is common in ActivitySim the component will modify the input dataframe in-place.
    # For testing we make a copy of the input tours to compare against after running the model.
    in_tours = tours.copy(deep=True)

    # run the trip scheduling choice model
    out_tours = tsc.run_trip_scheduling_choice(
        state,
        model_spec,
        tours,
        skims,
        locals_dict,
        trace_label="PyTest Trip Scheduling",
        model_settings=model_settings,
    )

    # check that the number of tours is unchanged
    assert len(in_tours) == len(out_tours)
    pd.testing.assert_index_equal(
        in_tours.sort_index().index, out_tours.sort_index().index
    )

    # check that the expected output columns are not present in input tours
    output_columns = [tsc.MAIN_LEG_DURATION, tsc.OB_DURATION, tsc.IB_DURATION]
    for col in output_columns:
        assert col not in in_tours.columns

    # check that the expected output columns *are* present in output tours
    assert set(output_columns).issubset(out_tours.columns)

    # check that the sum of the output durations equals the tour duration
    assert len(
        out_tours[
            out_tours[output_columns].sum(axis=1) == out_tours[tsc.TOUR_DURATION_COLUMN]
        ]
    ) == len(in_tours)

    # check that tours with no outbound stops have zero outbound duration
    assert out_tours[tsc.OB_DURATION].mask(in_tours[tsc.HAS_OB_STOPS], 0).sum() == 0

    # check that tours with no inbound stops have zero inbound duration
    assert out_tours[tsc.IB_DURATION].mask(in_tours[tsc.HAS_IB_STOPS], 0).sum() == 0
