import logging
import os

import pandas as pd
import pytest

# import models is necessary to initalize the model steps with orca
from activitysim.abm import models
from activitysim.core import config, pipeline


# Used by conftest.py initialize_pipeline method
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


def test_summarize(initialize_pipeline: pipeline.Pipeline, caplog):
    # Run summarize model
    caplog.set_level(logging.DEBUG)
    pipeline.run(models=["summarize"])

    # Retrieve output tables to check contents
    model_settings = config.read_model_settings("summarize.yaml")
    output_location = (
        model_settings["OUTPUT"] if "OUTPUT" in model_settings else "summaries"
    )
    output_dir = config.output_file_path(output_location)

    # Check that households are counted correctly
    households_count = pd.read_csv(
        config.output_file_path(os.path.join(output_location, f"households_count.csv"))
    )
    households = pd.read_csv(config.data_file_path("households.csv"))
    assert int(households_count.iloc[0]) == len(households)

    # Check that bike trips are counted correctly
    trips_by_mode_count = pd.read_csv(
        config.output_file_path(
            os.path.join(output_location, f"trips_by_mode_count.csv")
        )
    )
    trips = pd.read_csv(config.data_file_path("trips.csv"))
    assert int(trips_by_mode_count.BIKE.iloc[0]) == len(
        trips[trips.trip_mode == "BIKE"]
    )
