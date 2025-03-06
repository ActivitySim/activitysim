from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from numpy import dot
from numpy.linalg import norm

# import models is necessary to initalize the model steps
from activitysim.abm import models
from activitysim.core import config, simulate, tracing, workflow
from activitysim.core.util import read_csv, to_csv

logger = logging.getLogger(__name__)


# Used by conftest.py initialize_pipeline method
@pytest.fixture(scope="module")
def module() -> str:
    """
    A pytest fixture that returns the data folder location.
    :return: folder location for any necessary data to initialize the tests
    """
    return "parking_location"


# Used by conftest.py initialize_pipeline method
@pytest.fixture(scope="module")
def tables(prepare_module_inputs) -> dict[str, str]:
    """
    A pytest fixture that returns the "mock" tables to build pipeline dataframes. The
    key-value pair is the name of the table and the index column.
    :return: dict
    """
    return {
        "land_use": "maz",
        "persons": "person_id",
        "households": "household_id",
        "accessibility": "maz",
        "tours": "tour_id",
        "trips": "trip_id",
    }


# Used by conftest.py initialize_pipeline method
# Set to true if you need to read skims into the pipeline
@pytest.fixture(scope="module")
def initialize_network_los() -> bool:
    """
    A pytest boolean fixture indicating whether network skims should be read from the
    fixtures test data folder.
    :return: boolcls
    """
    return True


@pytest.fixture(scope="module")
def load_checkpoint() -> bool:
    """
    checkpoint to be loaded from the pipeline when reconnecting.
    """
    return "initialize_households"


# make a reconnect_pipeline internal to test module
@pytest.mark.skipif(
    os.path.isfile("test/parking_location/output/pipeline.h5"),
    reason="no need to recreate pipeline store if already exist",
)
def test_prepare_input_pipeline(initialize_pipeline: workflow.State, caplog):
    # Run summarize model
    caplog.set_level(logging.INFO)

    state = initialize_pipeline

    # run model step
    state.run(models=["initialize_landuse", "initialize_households"])

    # save the updated pipeline tables
    person_df = state.get_table("persons")
    to_csv(person_df, state.filesystem.get_output_file_path("person.csv"))

    household_df = state.get_table("households")
    to_csv(household_df, state.filesystem.get_output_file_path("household.csv"))

    land_use_df = state.get_table("land_use")
    to_csv(land_use_df, state.filesystem.get_output_file_path("land_use.csv"))

    accessibility_df = state.get_table("accessibility")
    to_csv(accessibility_df, state.filesystem.get_output_file_path("accessibility.csv"))

    tours_df = state.get_table("tours")
    to_csv(tours_df, state.filesystem.get_output_file_path("tours.csv"))

    trips_df = state.get_table("trips")
    to_csv(trips_df, state.filesystem.get_output_file_path("trips.csv"))

    state.close_pipeline()


# @pytest.mark.skip
def test_parking_location(reconnect_pipeline: workflow.State, caplog):
    # Run summarize model
    caplog.set_level(logging.INFO)

    state = reconnect_pipeline

    # run model step
    state.run(models=["parking_location"], resume_after="initialize_households")

    # get the updated trips data
    trips_df = state.get_table("trips")
    to_csv(trips_df, "test/parking_location/output/trips_after_parking_choice.csv")


# fetch/prepare existing files for model inputs
# e.g. read accessibilities.csv from ctramp result, rename columns, write out to accessibility.csv which is the input to activitysim
@pytest.fixture(scope="module")
def prepare_module_inputs(tmp_path_module: Path) -> Path:
    """

    copy input files from sharepoint into test folder

    create unique person id in person file

    :return: None
    """
    # https://wsponlinenam.sharepoint.com/sites/US-TM2ConversionProject/Shared%20Documents/Forms/
    # AllItems.aspx?id=%2Fsites%2FUS%2DTM2ConversionProject%2FShared%20Documents%2FTask%203%20ActivitySim&viewid=7a1eaca7%2D3999%2D4d45%2D9701%2D9943cc3d6ab1
    test_dir = os.path.join("test", "parking_location", "data")

    accessibility_file = os.path.join(
        "test", "parking_location", "data", "accessibilities.csv"
    )
    household_file = os.path.join(test_dir, "popsyn", "households.csv")
    person_file = os.path.join(test_dir, "popsyn", "persons.csv")
    landuse_file = os.path.join(test_dir, "landuse", "maz_data_withDensity.csv")

    shutil.copy(accessibility_file, os.path.join(test_dir, "accessibility.csv"))
    shutil.copy(household_file, os.path.join(test_dir, "households.csv"))
    shutil.copy(person_file, os.path.join(test_dir, "persons.csv"))
    shutil.copy(landuse_file, os.path.join(test_dir, "land_use.csv"))

    # add original maz id to accessibility table
    land_use_df = read_csv(os.path.join(test_dir, "land_use.csv"))

    land_use_df.rename(
        columns={"MAZ": "maz", "MAZ_ORIGINAL": "maz_county_based"}, inplace=True
    )

    land_use_df.to_csv(os.path.join(test_dir, "land_use.csv"), index=False)

    accessibility_df = read_csv(os.path.join(test_dir, "accessibility.csv"))

    accessibility_df["maz"] = accessibility_df["mgra"]

    to_csv(accessibility_df, tmp_path.joinpath("accessibility.csv"), index=False)

    # currently household file has to have these two columns, even before annotation
    # because annotate person happens before household and uses these two columns
    # TODO find a way to get around this
    ####
    household_df = read_csv(os.path.join(test_dir, "households.csv"))

    household_columns_dict = {
        "HHID": "household_id",
        "TAZ": "taz",
        "MAZ": "maz_county_based",
    }

    household_df.rename(columns=household_columns_dict, inplace=True)

    tm2_simulated_household_df = read_csv(
        os.path.join(test_dir, "tm2_outputs", "householdData_1.csv")
    )
    tm2_simulated_household_df.rename(columns={"hh_id": "household_id"}, inplace=True)

    household_df = pd.merge(
        household_df,
        tm2_simulated_household_df[
            [
                "household_id",
                "autos",
                "automated_vehicles",
                "transponder",
                "cdap_pattern",
                "jtf_choice",
            ]
        ],
        how="inner",  # tm2 is not 100% sample run
        on="household_id",
    )

    to_csv(household_df, tmp_path.joinpath("households.csv"), index=False)

    person_df = read_csv(os.path.join(test_dir, "persons.csv"))

    person_columns_dict = {"HHID": "household_id", "PERID": "person_id"}

    person_df.rename(columns=person_columns_dict, inplace=True)

    tm2_simulated_person_df = read_csv(
        os.path.join(test_dir, "tm2_outputs", "personData_1.csv")
    )
    tm2_simulated_person_df.rename(columns={"hh_id": "household_id"}, inplace=True)

    person_df = pd.merge(
        person_df,
        tm2_simulated_person_df[
            [
                "household_id",
                "person_id",
                "person_num",
                "type",
                "value_of_time",
                "activity_pattern",
                "imf_choice",
                "inmf_choice",
                "fp_choice",
                "reimb_pct",
                "workDCLogsum",
                "schoolDCLogsum",
            ]
        ],
        how="inner",  # tm2 is not 100% sample run
        on=["household_id", "person_id"],
    )

    # get tm2 simulated workplace and school location results
    tm2_simulated_wsloc_df = read_csv(
        ext_examp_dir.joinpath("tm2_outputs", "wsLocResults_3.csv.gz")
    )
    tm2_simulated_wsloc_df.rename(
        columns={"HHID": "household_id", "PersonID": "person_id"}, inplace=True
    )

    person_df = pd.merge(
        person_df,
        tm2_simulated_wsloc_df[
            [
                "household_id",
                "person_id",
                "WorkLocation",
                "WorkLocationLogsum",  # this is the same as `workDCLogsum` in tm2 person output
                "SchoolLocation",
                "SchoolLocationLogsum",  # this is the same as `schoolDCLogsum` in tm2 person output
            ]
        ],
        how="inner",  # ctramp might not be 100% sample run
        on=["household_id", "person_id"],
    )

    person_df.to_csv(os.path.join(test_dir, "persons.csv"), index=False)

    ## get tour data from tm2 output

    tm2_simulated_indiv_tour_df = read_csv(
        ext_examp_dir.joinpath("tm2_outputs", "indivTourData_1.csv.gz")
    )
    tm2_simulated_joint_tour_df = read_csv(
        os.path.join(test_dir, "tm2_outputs", "jointTourData_1.csv")
    )

    tm2_simulated_tour_df = pd.concat(
        [tm2_simulated_indiv_tour_df, tm2_simulated_joint_tour_df],
        sort=False,
        ignore_index=True,
    )

    tm2_simulated_tour_df.rename(columns={"hh_id": "household_id"}, inplace=True)

    tm2_simulated_tour_df["unique_tour_id"] = range(1, len(tm2_simulated_tour_df) + 1)

    ## get trip data from tm2 output
    tm2_simulated_indiv_trip_df = read_csv(
        os.path.join(test_dir, "tm2_outputs", "indivTripData_1.csv")
    )
    tm2_simulated_joint_trip_df = read_csv(
        os.path.join(test_dir, "tm2_outputs", "jointTripData_1.csv")
    )

    tm2_simulated_trip_df = pd.concat(
        [tm2_simulated_indiv_trip_df, tm2_simulated_joint_trip_df],
        sort=False,
        ignore_index=True,
    )

    tm2_simulated_trip_df.rename(columns={"hh_id": "household_id"}, inplace=True)

    tm2_simulated_trip_df["trip_id"] = range(1, len(tm2_simulated_trip_df) + 1)

    tm2_simulated_trip_df = pd.merge(
        tm2_simulated_trip_df,
        tm2_simulated_tour_df[
            [
                "household_id",
                "person_id",
                "tour_id",
                "tour_purpose",
                "unique_tour_id",
                "start_period",
                "end_period",
            ]
        ],
        how="left",
        on=["household_id", "person_id", "tour_id", "tour_purpose"],
    )
    # drop tour id and rename unique_tour_id to tour_id
    tm2_simulated_tour_df.drop(["tour_id"], axis=1, inplace=True)
    tm2_simulated_tour_df.rename(columns={"unique_tour_id": "tour_id"}, inplace=True)

    tm2_simulated_trip_df.drop(["tour_id"], axis=1, inplace=True)
    tm2_simulated_trip_df.rename(
        columns={
            "unique_tour_id": "tour_id",
            "orig_mgra": "origin",
            "dest_mgra": "destination",
            "start_period": "tour_start_period",
            "end_period": "tour_end_period",
        },
        inplace=True,
    )

    tm2_simulated_trip_df["purpose"] = tm2_simulated_trip_df["dest_purpose"].str.lower()

    period_map_df = read_csv(os.path.join(test_dir, "period_mapping_mtc.csv"))

    tm2_simulated_trip_df.sort_values(
        by=["household_id", "person_id", "person_num", "stop_period", "tour_id"],
        inplace=True,
    )

    tm2_simulated_trip_df = pd.merge(
        tm2_simulated_trip_df,
        period_map_df,
        left_on="stop_period",
        right_on="period",
        how="left",
    )

    tm2_simulated_trip_df["trip_departure_time"] = tm2_simulated_trip_df[
        "minutes_reference"
    ]
    tm2_simulated_trip_df["trip_arrival_time"] = (
        tm2_simulated_trip_df["trip_departure_time"]
        + tm2_simulated_trip_df["TRIP_TIME"]
    )
    tm2_simulated_trip_df["next_trip_departure_time"] = tm2_simulated_trip_df.groupby(
        ["household_id", "person_id", "person_num"]
    )["trip_arrival_time"].shift(-1)

    # activity duration is start of next trip - end of current trip
    tm2_simulated_trip_df["activity_duration"] = (
        tm2_simulated_trip_df["next_trip_departure_time"]
        - tm2_simulated_trip_df["trip_arrival_time"]
    )

    # set default activity duration to 5 minutes
    tm2_simulated_trip_df["activity_duration"] = np.where(
        tm2_simulated_trip_df["activity_duration"] > 0,
        tm2_simulated_trip_df["activity_duration"],
        5,
    )

    # activity duration in hours
    tm2_simulated_trip_df["activity_duration_in_hours"] = (
        tm2_simulated_trip_df["activity_duration"] / 60
    )

    tm2_simulated_trip_df.drop(
        [
            "period",
            "start_time",
            "end_time",
            "minutes_reference",
            "trip_departure_time",
            "trip_arrival_time",
            "next_trip_departure_time",
            "activity_duration",
        ],
        inplace=True,
        axis=1,
    )

    # randomly select 100,000 trips to run the parking location choice model
    # memory constraint on `logit.interaction_dataset` for large trips table
    # tm2_simulated_trip_df = tm2_simulated_trip_df.sample(100000, random_state=1)

    # select only if target parking maz is > 0
    tm2_simulated_trip_df = tm2_simulated_trip_df[
        tm2_simulated_trip_df["parking_mgra"] > 0
    ]

    tm2_simulated_tour_df = tm2_simulated_tour_df[
        tm2_simulated_tour_df.household_id.isin(tm2_simulated_trip_df.household_id)
    ]
    to_csv(tm2_simulated_tour_df, os.path.join(test_dir, "tours.csv"), index=False)

    to_csv(tm2_simulated_trip_df, os.path.join(test_dir, "trips.csv"), index=False)


def create_summary(input_df, key, out_col="Share") -> pd.DataFrame:
    """
    Create summary for the input data.
    1. group input data by the "key" column
    2. calculate the percent of input data records in each "key" category.

    :return: pd.DataFrame
    """

    out_df = input_df.groupby(key).size().reset_index(name="Count")
    out_df[out_col] = round(out_df["Count"] / out_df["Count"].sum(), 4)

    return out_df[[key, out_col]]


def cosine_similarity(a, b):
    """
    Computes cosine similarity between two vectors.

    Cosine similarity is used here as a metric to measure similarity between two sequence of numbers.
    Two sequence of numbers are represented as vectors (in a multi-dimensional space) and cosine similiarity is defined as the cosine of the angle between them
    i.e., dot products of the vectors divided by the product of their lengths.

    :return:
    """

    return dot(a, b) / (norm(a) * norm(b))


def compare_simulated_against_target(
    target_df: pd.DataFrame,
    simulated_df: pd.DataFrame,
    target_key: str,
    simulated_key: str,
) -> bool:
    """
    compares the simulated and target results by computing the cosine similarity between them.

    :return:
    """

    merged_df = pd.merge(
        target_df, simulated_df, left_on=target_key, right_on=simulated_key, how="outer"
    )
    merged_df = merged_df.fillna(0)

    logger.info("simulated vs target share:\n%s" % merged_df)

    similarity_value = cosine_similarity(
        merged_df["Target_Share"].tolist(), merged_df["Simulated_Share"].tolist()
    )

    logger.info("cosine similarity:\n%s" % similarity_value)

    return similarity_value
