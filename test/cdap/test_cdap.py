from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy import dot
from numpy.linalg import norm

# import models is necessary to initalize the model steps
from activitysim.abm import models  # noqa: F401
from activitysim.core import config, tracing, workflow
from activitysim.core.util import read_csv, to_csv

logger = logging.getLogger(__name__)


# Used by conftest.py initialize_pipeline method
@pytest.fixture(scope="module")
def module() -> str:
    """
    A pytest fixture that returns the data folder location.
    :return: folder location for any necessary data to initialize the tests
    """
    return "cdap"


# Used by conftest.py initialize_pipeline method
@pytest.fixture(scope="module")
def tables(prepare_module_inputs) -> dict[str, str]:
    """
    A pytest fixture that returns the "mock" tables to build pipeline dataframes. The
    key-value pair is the name of the table and the index column.
    :return: dict
    """
    return {
        "land_use": "MAZ_ORIGINAL",
        "persons": "person_id",
        "households": "household_id",
        "accessibility": "MAZ_ORIGINAL",
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
    return False


@pytest.fixture(scope="module")
def load_checkpoint() -> bool:
    """
    checkpoint to be loaded from the pipeline when reconnecting.
    """
    return "initialize_households"


# make a reconnect_pipeline internal to cdap
@pytest.mark.skipif(
    os.path.isfile("test/cdap/output/pipeline.h5"),
    reason="no need to recreate pipeline store if alreayd exist",
)
def test_prepare_input_pipeline(initialize_pipeline: workflow.State, caplog):
    # Run summarize model
    caplog.set_level(logging.INFO)

    state = initialize_pipeline
    # run model step
    state.run(models=["initialize_landuse", "initialize_households"])

    state.close_pipeline()


def test_cdap_from_pipeline(reconnect_pipeline: workflow.State, caplog):
    # Run summarize model
    caplog.set_level(logging.INFO)
    state = reconnect_pipeline

    # run model step
    state.run(models=["cdap_simulate"], resume_after="initialize_households")

    # get the updated pipeline data
    person_df = state.get_table("persons")

    # get the updated pipeline data
    household_df = state.get_table("households")

    ############################
    # person cdap pattern validation
    ############################
    # cdap person result from the model
    logger.info("person cdap pattern validation")

    target_key = "activity_pattern"
    simulated_key = "cdap_activity"
    similarity_threshold = 0.99

    simulated_df = create_summary(
        person_df, key=simulated_key, out_col="Simulated_Share"
    )

    # cdap person result from the TM2 run
    target_df = create_summary(person_df, key=target_key, out_col="Target_Share")

    # compare simulated and target results
    similarity_value = compare_simulated_against_target(
        target_df, simulated_df, target_key, simulated_key
    )

    # if the cosine_similarity >= threshold then the simulated and target results are "similar"
    assert similarity_value >= similarity_threshold

    ############################
    # household joint pattern validation
    ############################
    # cdap household result from the model
    logger.info("household joint vs non-joint tour validation")

    target_key = "has_joint_tour_tm2"
    simulated_key = "has_joint_tour"
    similarity_threshold = 0.99

    simulated_df = create_summary(
        household_df, key=simulated_key, out_col="Simulated_Share"
    )

    # cdap household result from the TM2 run
    household_df["has_joint_tour_tm2"] = np.where(
        household_df["cdap_pattern"].str.endswith("j"), 1, 0
    )
    target_df = create_summary(household_df, key=target_key, out_col="Target_Share")

    # compare simulated and target results
    similarity_value = compare_simulated_against_target(
        target_df, simulated_df, target_key, simulated_key
    )

    # if the cosine_similarity >= threshold then the simulated and target results are "similar"
    assert similarity_value >= similarity_threshold

    ############################
    # one person household cdap pattern validation
    ############################
    # cdap household result from the model
    logger.info("one-person household cdap pattern validation")

    target_key = "cdap_pattern"
    simulated_key = "cdap_activity"
    similarity_threshold = 0.99

    household_df["cdap_pattern"] = household_df["cdap_pattern"].apply(
        lambda x: x[:-1].upper() if x.endswith("0") else x.upper()
    )
    household_df["cdap_activity"] = household_df.apply(
        lambda x: x["cdap_activity"] + "J"
        if x["has_joint_tour"] == 1
        else x["cdap_activity"],
        axis=1,
    )

    hh1_df = household_df[household_df.hhsize == 1].copy()

    simulated_df = create_summary(hh1_df, key=simulated_key, out_col="Simulated_Share")
    target_df = create_summary(hh1_df, key=target_key, out_col="Target_Share")

    # compare simulated and target results
    similarity_value = compare_simulated_against_target(
        target_df, simulated_df, target_key, simulated_key
    )

    # if the cosine_similarity >= threshold then the simulated and target results are "similar"
    assert similarity_value >= similarity_threshold

    ############################
    # two person household cdap pattern validation
    ############################
    # cdap household result from the model
    logger.info("two-person household cdap pattern validation")

    target_key = "cdap_pattern"
    simulated_key = "cdap_activity"
    similarity_threshold = 0.99

    hh2_df = household_df[household_df.hhsize == 2].copy()

    simulated_df = create_summary(hh2_df, key=simulated_key, out_col="Simulated_Share")
    target_df = create_summary(hh2_df, key=target_key, out_col="Target_Share")

    # compare simulated and target results
    similarity_value = compare_simulated_against_target(
        target_df, simulated_df, target_key, simulated_key
    )

    # if the cosine_similarity >= threshold then the simulated and target results are "similar"
    assert similarity_value >= similarity_threshold

    ############################
    # three person household cdap pattern validation
    ############################
    # cdap household result from the model
    logger.info("three-person household cdap pattern validation")

    target_key = "cdap_pattern"
    simulated_key = "cdap_activity"
    similarity_threshold = 0.99

    hh3_df = household_df[household_df.hhsize == 3].copy()

    simulated_df = create_summary(hh3_df, key=simulated_key, out_col="Simulated_Share")
    target_df = create_summary(hh3_df, key=target_key, out_col="Target_Share")

    # compare simulated and target results
    similarity_value = compare_simulated_against_target(
        target_df, simulated_df, target_key, simulated_key
    )

    # if the cosine_similarity >= threshold then the simulated and target results are "similar"
    assert similarity_value >= similarity_threshold


# fetch/prepare existing files for model inputs
# e.g. read accessibilities.csv from ctramp result, rename columns, write out to accessibility.csv which is the input to activitysim
@pytest.fixture(scope="module")
def prepare_module_inputs(tmp_path_module: Path) -> Path:
    """
    copy input files from sharepoint into test folder

    create unique person id in person file

    :return: None
    """
    tmp_path = tmp_path_module
    tmp_path.mkdir(parents=True, exist_ok=True)

    from activitysim.examples.external import registered_external_example

    ext_examp_dir = registered_external_example("legacy_mtc", tmp_path)

    # https://wsponlinenam.sharepoint.com/sites/US-TM2ConversionProject/Shared%20Documents/Forms/
    # AllItems.aspx?id=%2Fsites%2FUS%2DTM2ConversionProject%2FShared%20Documents%2FTask%203%20ActivitySim&viewid=7a1eaca7%2D3999%2D4d45%2D9701%2D9943cc3d6ab1
    # accessibility_file = os.path.join(
    #     "test", "cdap", "data", "tm2_outputs", "accessibilities.csv"
    # )
    # household_file = os.path.join("test", "cdap", "data", "popsyn", "households.csv")
    # person_file = os.path.join("test", "cdap", "data", "popsyn", "persons.csv")
    # landuse_file = os.path.join(
    #     "test", "cdap", "data", "landuse", "maz_data_withDensity.csv"
    # )
    #
    # test_dir = os.path.join("test", "cdap", "data")
    #
    # shutil.copy(accessibility_file, os.path.join(test_dir, "accessibility.csv"))
    # shutil.copy(household_file, os.path.join(test_dir, "households.csv"))
    # shutil.copy(person_file, os.path.join(test_dir, "persons.csv"))
    # shutil.copy(landuse_file, os.path.join(test_dir, "land_use.csv"))

    # add original maz id to accessibility table
    land_use_df = pd.read_csv(
        ext_examp_dir.joinpath("landuse", "maz_data_withDensity.csv.gz")
    )
    accessibility_df = pd.read_csv(
        ext_examp_dir.joinpath("tm2_outputs", "accessibilities.csv.gz")
    )

    accessibility_df = pd.merge(
        accessibility_df,
        land_use_df[["MAZ", "MAZ_ORIGINAL"]].rename(columns={"MAZ": "mgra"}),
        how="left",
        on="mgra",
    )

    to_csv(accessibility_df, tmp_path.joinpath("accessibility.csv"), index=False)

    # currently household file has to have these two columns, even before annotation
    # because annotate person happens before household and uses these two columns
    # TODO find a way to get around this
    ####
    household_df = read_csv(ext_examp_dir.joinpath("popsyn", "households.csv.gz"))

    household_columns_dict = {"HHID": "household_id", "MAZ": "home_zone_id"}

    household_df.rename(columns=household_columns_dict, inplace=True)

    tm2_simulated_household_df = read_csv(
        ext_examp_dir.joinpath("tm2_outputs", "householdData_3.csv.gz")
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

    person_df = read_csv(ext_examp_dir.joinpath("popsyn", "persons.csv.gz"))

    person_columns_dict = {"HHID": "household_id", "PERID": "person_id"}

    person_df.rename(columns=person_columns_dict, inplace=True)

    tm2_simulated_person_df = read_csv(
        ext_examp_dir.joinpath("tm2_outputs", "personData_3.csv.gz")
    )
    tm2_simulated_person_df.rename(columns={"hh_id": "household_id"}, inplace=True)

    person_df = pd.merge(
        person_df,
        tm2_simulated_person_df[
            [
                "household_id",
                "person_id",
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

    to_csv(person_df, tmp_path.joinpath("persons.csv"), index=False)
    return tmp_path


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
