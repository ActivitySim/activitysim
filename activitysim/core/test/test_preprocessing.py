# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging
import logging.config
import os.path

import numpy as np
import pandas as pd
import pytest

from activitysim.core import workflow, expressions, los
from activitysim.core.configuration.base import PreprocessorSettings


def add_canonical_dirs(configs_dir_name):
    state = workflow.State()
    los_configs_dir = os.path.join(os.path.dirname(__file__), f"los/{configs_dir_name}")
    configs_dir = os.path.join(os.path.dirname(__file__), "configs")
    data_dir = os.path.join(os.path.dirname(__file__), f"los/data")
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    state.initialize_filesystem(
        working_dir=os.path.dirname(__file__),
        configs_dir=(los_configs_dir, configs_dir),
        output_dir=output_dir,
        data_dir=(data_dir,),
    )
    return state


@pytest.fixture
def state() -> workflow.State:
    state = add_canonical_dirs("configs_1z").load_settings()
    network_los = los.Network_LOS(state)
    network_los.load_data()
    state.set("skim_dict", network_los.get_default_skim_dict())
    return state


@pytest.fixture(scope="module")
def households():
    return pd.DataFrame(
        {
            "household_id": [1, 2, 3],
            "home_zone_id": [1, 2, 3],
            "income": [50000, 60000, 70000],
        }
    ).set_index("household_id")


@pytest.fixture(scope="module")
def persons():
    return pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4, 5],
            "household_id": [1, 1, 2, 2, 3],
            "age": [25, 30, 22, 28, 35],
        }
    ).set_index("person_id")


@pytest.fixture(scope="module")
def tours():
    return pd.DataFrame(
        {
            "tour_id": [1, 2, 3],
            "household_id": [1, 2, 3],
            "person_id": [1, 2, 3],
            "tour_type": ["work", "shopping", "othmaint"],
            "origin": [1, 2, 3],
            "destination": [2, 3, 1],
            "period": ["AM", "PM", "AM"],
        }
    ).set_index("tour_id")


def check_outputs(tours):
    """
    Check that the tours DataFrame has the expected new columns and values
    according to the preprocessor / annotator expressions.
    """
    new_cols = [
        "is_high_income",
        "num_persons",
        "od_distance",
        "od_distance_wrapper",
        "od_sov_time",
        "constant_test",
    ]

    # check all new columns are added
    assert all(
        col in tours.columns for col in new_cols
    ), f"Missing columns: {set(new_cols) - set(tours.columns)}"

    # column with _ shouldn't be in the columns
    assert (
        "_hh_income" not in tours.columns
    ), f"Unexpected column found: _hh_income in {tours.columns}"

    # check the values in the new columns
    exppected_output = pd.DataFrame(
        {
            "tour_id": [1, 2, 3],
            "is_high_income": [False, True, True],
            "num_persons": [2, 2, 1],
            "od_distance": [0.24, 0.28, 0.57],
            "od_distance_wrapper": [0.24, 0.28, 0.57],
            "od_sov_time": [0.78, 0.89, 1.76],
            "constant_test": [21, 21, 21],
        }
    ).set_index("tour_id")
    pd.testing.assert_frame_equal(tours[new_cols], exppected_output, check_dtype=False)


def setup_skims(state: workflow.State):
    """Creates a set of skim wrappers to test in expressions."""
    skim_dict = state.get("skim_dict")
    skims3d = skim_dict.wrap_3d(
        orig_key="origin", dest_key="destination", dim3_key="period"
    )
    skims2d = skim_dict.wrap("origin", "destination")
    return {"skims3d": skims3d, "skims2d": skims2d}


def test_preprocessor(state: workflow.State, households, persons, tours):
    # adding dataframes to state so they can be accessed in preprocessor
    state.add_table("households", households)
    state.add_table("persons", persons)
    original_tours = tours.copy()
    state.add_table("tours", original_tours)

    # defining preprocessor
    preprocessor_settings = PreprocessorSettings(
        SPEC="preprocessor.csv",
        DF="tours",
        TABLES=["persons", "households"],
    )
    model_settings = {"preprocessor": preprocessor_settings}

    # annotating preprocessors
    expressions.annotate_preprocessors(
        state,
        df=tours,
        locals_dict={"test_constant": 42},
        skims=setup_skims(state),
        model_settings=model_settings,
        trace_label="ci_test_preprocessor",
    )

    check_outputs(tours)

    state_tours = state.get_table("tours")
    # check that the state table is not modified
    pd.testing.assert_frame_equal(state_tours, original_tours)


def test_annotator(state, households, persons, tours):
    # adding dataframes to state so they can be accessed in annotator
    state.add_table("households", households)
    state.add_table("persons", persons)
    original_tours = tours.copy()
    state.add_table("tours", original_tours)

    # defining annotator
    annotator_settings = PreprocessorSettings(
        SPEC="preprocessor.csv",
        DF="tours",
        TABLES=["persons", "households"],
    )
    model_settings = {"annotate_tours": annotator_settings}

    # annotating preprocessors
    expressions.annotate_tables(
        state,
        model_settings=model_settings,
        trace_label="ci_test_annotator",
        skims=setup_skims(state),
        locals_dict={"test_constant": 42},
    )

    # outputs now put directly into the state object
    check_outputs(state.get_table("tours"))

    # test what happens if we try to annotate a table that does not exist
    model_settings = {"annotate_trips": annotator_settings}

    with pytest.raises(ValueError) as excinfo:
        # this should raise an error because "trips" table does not exist in state
        expressions.annotate_tables(
            state,
            model_settings=model_settings,
            trace_label="ci_test_annotator",
            skims=None,
            locals_dict={"test_constant": 42},
        )
