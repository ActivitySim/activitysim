import logging
import pytest
import os
import shutil
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
import orca
from numpy.random import randint
import scipy.stats as stats

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
    return "auto_ownership"


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


# Used by conftest.py reconnect_pipeline method
@pytest.fixture(scope="module")
def load_checkpoint() -> bool:
    """
    checkpoint to be loaded from the pipeline when reconnecting.
    """
    return "initialize_households"


@pytest.mark.skipif(
    os.path.isfile("test/auto_ownership/output/pipeline.h5"),
    reason="no need to recreate pipeline store if alreayd exist",
)
def test_prepare_input_pipeline(initialize_pipeline: pipeline.Pipeline, caplog):
    # Run summarize model
    caplog.set_level(logging.INFO)

    # run model step
    pipeline.run(models=["initialize_landuse", "initialize_households"])

    pipeline.close_pipeline()


def test_auto_ownership(reconnect_pipeline: pipeline.Pipeline, caplog):

    caplog.set_level(logging.INFO)

    # run model step
    pipeline.run(
        models=["auto_ownership_simulate"], resume_after="initialize_households"
    )

    # get the updated pipeline data
    household_df = pipeline.get_table("households")
    # logger.info("household_df columns: ", household_df.columns.tolist())

    # target_col = "autos"
    target_col = "pre_autos"
    choice_col = "auto_ownership"
    simulated_col = "autos_model"

    similarity_threshold = 0.99

    ao_alternatives_df = pd.DataFrame.from_dict(
        {
            choice_col: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "auto_choice_label": [
                "0_CARS",
                "1_CAR_1CV",
                "1_CAR_1AV",
                "2_CARS_2CV",
                "2_CARS_2AV",
                "2_CARS_1CV1AV",
                "3_CARS_3CV",
                "3_CARS_3AV",
                "3_CARS_2CV1AV",
                "3_CARS_1CV2AV",
                "4_CARS_4CV",
            ],
            simulated_col: [0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4],
        }
    )

    household_df = pd.merge(household_df, ao_alternatives_df, how="left", on=choice_col)

    # AO summary from the model
    simulated_df = create_summary(
        household_df, key=simulated_col, out_col="Simulated_Share"
    )

    # AO summary from the results/target
    target_df = create_summary(household_df, key=target_col, out_col="Target_Share")

    merged_df = pd.merge(
        target_df, simulated_df, left_on=target_col, right_on=simulated_col, how="outer"
    )
    merged_df = merged_df.fillna(0)

    # compare simulated and target results by computing the cosine similarity between them
    similarity_value = cosine_similarity(
        merged_df["Target_Share"].tolist(), merged_df["Simulated_Share"].tolist()
    )

    # save the results to disk
    merged_df.to_csv(
        os.path.join("test", "auto_ownership", "output", "ao_test_results.csv"),
        index=False,
    )

    # if the cosine_similarity >= threshold then the simulated and target results are "similar"
    assert similarity_value >= similarity_threshold


@pytest.mark.skip
def test_auto_ownership_variation(reconnect_pipeline: pipeline.Pipeline, caplog):

    caplog.set_level(logging.INFO)

    output_file = os.path.join(
        "test", "auto_ownership", "output", "ao_results_variation.csv"
    )

    if os.path.isfile(output_file):
        out_df = pd.read_csv(output_file)

        pipeline.open_pipeline(resume_after="initialize_households")

        household_df = pipeline.get_table("households")

    else:
        target_col = "pre_autos"
        choice_col = "auto_ownership"
        simulated_col = "autos_model"

        ao_alternatives_df = pd.DataFrame.from_dict(
            {
                choice_col: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "auto_choice_label": [
                    "0_CARS",
                    "1_CAR_1CV",
                    "1_CAR_1AV",
                    "2_CARS_2CV",
                    "2_CARS_2AV",
                    "2_CARS_1CV1AV",
                    "3_CARS_3CV",
                    "3_CARS_3AV",
                    "3_CARS_2CV1AV",
                    "3_CARS_1CV2AV",
                    "4_CARS_4CV",
                ],
                simulated_col: [0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4],
            }
        )

        NUM_SEEDS = 100

        for i in range(1, NUM_SEEDS + 1):
            base_seed = randint(1, 99999)
            orca.add_injectable("rng_base_seed", base_seed)

            # run model step
            pipeline.run(
                models=["auto_ownership_simulate"], resume_after="initialize_households"
            )

            # get the updated pipeline data
            household_df = pipeline.get_table("households")

            household_df = pd.merge(
                household_df, ao_alternatives_df, how="left", on=choice_col
            )

            # AO summary from the model
            simulated_share_col_name = "simulation_" + str(i)
            simulated_df = create_summary(
                household_df, key=simulated_col, out_col=simulated_share_col_name
            )

            if i == 1:
                out_df = create_summary(household_df, key=target_col, out_col="target")

            out_df = pd.concat(
                [out_df, simulated_df[[simulated_share_col_name]]], axis=1
            )

            # since model_name is used as checkpoint name, the same model can not be run more than once.
            # have to close the pipeline before running the same model again.
            pipeline.close_pipeline()

        out_df["simulation_min"] = out_df.filter(like="simulation_").min(axis=1)
        out_df["simulation_max"] = out_df.filter(like="simulation_").max(axis=1)
        out_df["simulation_mean"] = out_df.filter(like="simulation_").mean(axis=1)

        # reorder columns
        cols = [
            "pre_autos",
            "target",
            "simulation_mean",
            "simulation_min",
            "simulation_max",
        ]
        cols = cols + [x for x in out_df.columns if x not in cols]
        out_df = out_df[cols]

        out_df.to_csv(output_file, index=False)

    # chi-square test
    alpha = 0.05
    observed_prob = out_df["simulation_mean"]
    expected_prob = out_df["target"]

    num_hh = len(household_df)

    observed = [p * num_hh for p in observed_prob]
    expected = [p * num_hh for p in expected_prob]

    observed_sum = sum(observed)
    expected_sum = sum(expected)

    # the sum of the observed and expected frequencies must be the same for the test
    observed = [f * expected_sum / observed_sum for f in observed]

    chi_square, prob = stats.chisquare(observed, expected)
    p_value = 1 - prob

    # Ho: Target and Simulated results are from same distribution and there is no significant difference b/w the two.
    # Ha: Target and Simulated results are statistically different.

    conclusion = "Failed to reject the null hypothesis."
    if p_value >= alpha:
        conclusion = "Null Hypothesis is rejected. Obersved and Simulated results are statistically different."

    logger.info(conclusion)

    # if p-value is less than alpha, then difference between simulated and target results
    # are statistically greater than the random variation of the model results
    assert p_value < alpha


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
    accessibility_file = os.path.join(
        "test", "auto_ownership", "data", "accessibilities.csv"
    )
    household_file = os.path.join(
        "test", "auto_ownership", "data", "popsyn", "households.csv"
    )
    person_file = os.path.join(
        "test", "auto_ownership", "data", "popsyn", "persons.csv"
    )
    landuse_file = os.path.join(
        "test", "auto_ownership", "data", "landuse", "maz_data_withDensity.csv"
    )

    test_dir = os.path.join("test", "auto_ownership", "data")

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

    # household file from populationsim
    household_df = pd.read_csv(os.path.join(test_dir, "households.csv"))

    household_columns_dict = {"HHID": "household_id", "MAZ": "home_zone_id"}

    household_df.rename(columns=household_columns_dict, inplace=True)

    # get columns from ctramp output
    tm2_simulated_household_df = pd.read_csv(
        os.path.join(test_dir, "tm2_outputs", "householdData_3.csv")
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

    tm2_pre_ao_results_df = pd.read_csv(
        os.path.join(test_dir, "tm2_outputs", "aoResults_pre.csv")
    )
    tm2_pre_ao_results_df.rename(
        columns={"HHID": "household_id", "AO": "pre_autos"}, inplace=True
    )

    household_df = pd.merge(
        household_df, tm2_pre_ao_results_df, how="inner", on="household_id"
    )

    household_df.to_csv(os.path.join(test_dir, "households.csv"), index=False)

    # person file from populationsim
    person_df = pd.read_csv(os.path.join(test_dir, "persons.csv"))

    person_columns_dict = {"HHID": "household_id", "PERID": "person_id"}

    person_df.rename(columns=person_columns_dict, inplace=True)

    # get columns from ctramp result
    tm2_simulated_person_df = pd.read_csv(
        os.path.join(test_dir, "tm2_outputs", "personData_3.csv")
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
        how="inner",  # ctramp might not be 100% sample run
        on=["household_id", "person_id"],
    )

    person_df.to_csv(os.path.join(test_dir, "persons.csv"), index=False)


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
