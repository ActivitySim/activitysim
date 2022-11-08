import logging
import pytest
import os
import shutil
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm

# import models is necessary to initalize the model steps with orca
from activitysim.abm import models
from activitysim.core import pipeline, config
from activitysim.core import tracing

logger = logging.getLogger(__name__)

# Used by conftest.py initialize_pipeline method
@pytest.fixture(scope="module")
def module() -> str:
    """
    A pytest fixture that returns the data folder location.
    :return: folder location for any necessary data to initialize the tests
    """
    return "non_mandatory_tour_frequency"


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
        "tours": "tour_id",
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


# make a reconnect_pipeline internal to test module
@pytest.mark.skipif(
    os.path.isfile("test/non_mandatory_tour_frequency/output/pipeline.h5"),
    reason="no need to recreate pipeline store if alreayd exist",
)
def test_prepare_input_pipeline(initialize_pipeline: pipeline.Pipeline, caplog):
    # Run summarize model
    caplog.set_level(logging.INFO)

    # run model step
    pipeline.run(models=["initialize_landuse", "initialize_households"])

    # get the updated pipeline data
    person_df = pipeline.get_table("persons")
    person_df.to_csv("test/non_mandatory_tour_frequency/output/person.csv", index=False)

    # get the updated pipeline data
    household_df = pipeline.get_table("households")
    household_df.to_csv(
        "test/non_mandatory_tour_frequency/output/household.csv", index=False
    )

    pipeline.close_pipeline()


def test_nmtf_from_pipeline(reconnect_pipeline: pipeline.Pipeline, caplog):
    # Run summarize model
    caplog.set_level(logging.INFO)

    # run model step
    pipeline.run(
        models=["non_mandatory_tour_frequency"], resume_after="initialize_households"
    )

    # get the updated pipeline data
    person_df = pipeline.get_table("persons")

    # get the updated pipeline data
    household_df = pipeline.get_table("households")

    ############################
    # person nmtf validation
    ############################
    # nmtf person result from the model
    logger.info("person nmtf pattern validation")

    target_key = "inmf_choice"
    simulated_key = "non_mandatory_tour_frequency"
    similarity_threshold = 0.99

    simulated_df = create_summary(
        person_df, key=simulated_key, out_col="Simulated_Share"
    )

    # result from the TM2 run
    target_df = create_summary(person_df, key=target_key, out_col="Target_Share")

    # compare simulated and target results
    similarity_value = compare_simulated_against_target(
        target_df, simulated_df, target_key, simulated_key
    )

    # if the cosine_similarity >= threshold then the simulated and target results are "similar"
    assert similarity_value >= similarity_threshold


# fetch/prepare existing files for model inputs
# e.g. read accessibilities.csv from ctramp result, rename columns, write out to accessibility.csv which is the input to activitysim
@pytest.fixture(scope="module")
def prepare_module_inputs() -> None:
    """
    copy input files from sharepoint into test folder

    create unique person id in person file

    :return: None
    """
    # https://wsponlinenam.sharepoint.com/sites/US-TM2ConversionProject/Shared%20Documents/Forms/
    # AllItems.aspx?id=%2Fsites%2FUS%2DTM2ConversionProject%2FShared%20Documents%2FTask%203%20ActivitySim&viewid=7a1eaca7%2D3999%2D4d45%2D9701%2D9943cc3d6ab1
    test_dir = os.path.join("test", "non_mandatory_tour_frequency", "data")

    accessibility_file = os.path.join(test_dir, "tm2_outputs", "accessibilities.csv")
    household_file = os.path.join(test_dir, "popsyn", "households.csv")
    person_file = os.path.join(test_dir, "popsyn", "persons.csv")
    landuse_file = os.path.join(test_dir, "landuse", "maz_data_withDensity.csv")

    shutil.copy(accessibility_file, os.path.join(test_dir, "accessibility.csv"))
    shutil.copy(household_file, os.path.join(test_dir, "households.csv"))
    shutil.copy(person_file, os.path.join(test_dir, "persons.csv"))
    shutil.copy(landuse_file, os.path.join(test_dir, "land_use.csv"))

    # add original maz id to accessibility table
    land_use_df = pd.read_csv(os.path.join(test_dir, "land_use.csv"))
    accessibility_df = pd.read_csv(os.path.join(test_dir, "accessibility.csv"))

    accessibility_df = pd.merge(
        accessibility_df,
        land_use_df[["MAZ", "MAZ_ORIGINAL"]].rename(columns={"MAZ": "mgra"}),
        how="left",
        on="mgra",
    )

    accessibility_df.to_csv(os.path.join(test_dir, "accessibility.csv"), index=False)

    # currently household file has to have these two columns, even before annotation
    # because annotate person happens before household and uses these two columns
    # TODO find a way to get around this
    ####
    household_df = pd.read_csv(os.path.join(test_dir, "households.csv"))

    household_columns_dict = {"HHID": "household_id", "MAZ": "home_zone_id"}

    household_df.rename(columns=household_columns_dict, inplace=True)

    tm2_simulated_household_df = pd.read_csv(
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

    household_df.to_csv(os.path.join(test_dir, "households.csv"), index=False)

    person_df = pd.read_csv(os.path.join(test_dir, "persons.csv"))

    person_columns_dict = {"HHID": "household_id", "PERID": "person_id"}

    person_df.rename(columns=person_columns_dict, inplace=True)

    tm2_simulated_person_df = pd.read_csv(
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
    tm2_simulated_wsloc_df = pd.read_csv(
        os.path.join(test_dir, "tm2_outputs", "wsLocResults_1.csv")
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

    ## get tour data from tn2 output

    tm2_simulated_indiv_tour_df = pd.read_csv(
        os.path.join(test_dir, "tm2_outputs", "indivTourData_1.csv")
    )
    tm2_simulated_joint_tour_df = pd.read_csv(
        os.path.join(test_dir, "tm2_outputs", "jointTourData_1.csv")
    )

    tm2_simulated_tour_df = pd.concat(
        [tm2_simulated_indiv_tour_df, tm2_simulated_joint_tour_df],
        sort=False,
        ignore_index=True,
    )

    tm2_simulated_tour_df.rename(columns={"hh_id": "household_id"}).to_csv(
        os.path.join(test_dir, "tours.csv"), index=False
    )

    ####


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
