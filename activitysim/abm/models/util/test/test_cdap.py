# ActivitySim
# See full license in LICENSE.txt.

import os.path

import pandas as pd
import pandas.testing as pdt
import pytest
import yaml

from activitysim.core import chunk, config, inject, simulate

from .. import cdap


@pytest.fixture(scope="module")
def data_dir():
    return os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture(scope="module")
def people(data_dir):
    return pd.read_csv(os.path.join(data_dir, "people.csv"), index_col="id")


def teardown_function(func):
    inject.clear_cache()
    inject.reinject_decorated_tables()


@pytest.fixture(scope="module")
def model_settings(configs_dir):
    yml_file = os.path.join(configs_dir, "cdap.yaml")
    with open(yml_file) as f:
        model_settings = yaml.load(f, Loader=yaml.loader.SafeLoader)
    return model_settings


@pytest.fixture(scope="module")
def configs_dir():
    return os.path.join(os.path.dirname(__file__), "configs")


def setup_function():
    configs_dir = os.path.join(os.path.dirname(__file__), "configs")
    inject.add_injectable("configs_dir", configs_dir)
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    inject.add_injectable("output_dir", output_dir)


def test_bad_coefficients():

    coefficients = pd.read_csv(
        config.config_file_path("cdap_interaction_coefficients.csv"), comment="#"
    )
    coefficients = cdap.preprocess_interaction_coefficients(coefficients)

    coefficients.loc[2, "activity"] = "AA"

    with pytest.raises(RuntimeError) as excinfo:
        coefficients = cdap.preprocess_interaction_coefficients(coefficients)
    assert "Expect only M, N, or H" in str(excinfo.value)


def test_assign_cdap_rank(people, model_settings):

    person_type_map = model_settings.get("PERSON_TYPE_MAP", {})

    with chunk.chunk_log("test_assign_cdap_rank", base=True):
        cdap.assign_cdap_rank(people, person_type_map)

    expected = pd.Series(
        [1, 1, 1, 2, 2, 1, 3, 1, 2, 1, 3, 2, 1, 3, 2, 4, 1, 3, 4, 2], index=people.index
    )

    pdt.assert_series_equal(
        people["cdap_rank"], expected, check_dtype=False, check_names=False
    )


def test_individual_utilities(people, model_settings):

    cdap_indiv_and_hhsize1 = simulate.read_model_spec(
        file_name="cdap_indiv_and_hhsize1.csv"
    )

    person_type_map = model_settings.get("PERSON_TYPE_MAP", {})

    with chunk.chunk_log("test_individual_utilities", base=True):
        cdap.assign_cdap_rank(people, person_type_map)
        individual_utils = cdap.individual_utilities(
            people, cdap_indiv_and_hhsize1, locals_d=None
        )

    individual_utils = individual_utils[["M", "N", "H"]]

    expected = pd.DataFrame(
        [
            [2, 0, 0],  # person 1
            [0, 0, 1],  # person 2
            [3, 0, 0],  # person 3
            [3, 0, 0],  # person 4
            [0, 1, 0],  # person 5
            [1, 0, 0],  # person 6
            [1, 0, 0],  # person 7
            [0, 2, 0],  # person 8
            [0, 0, 1],  # person 9
            [2, 0, 0],  # person 10
            [0, 0, 3],  # person 11
            [0, 0, 2],  # person 12
            [3, 0, 0],  # person 13
            [1, 0, 0],  # person 14
            [0, 4, 0],  # person 15
            [0, 4, 0],  # person 16
            [0, 0, 4],  # person 17
            [0, 0, 5],  # person 18
            [50, 0, 4],  # person 19
            [2, 0, 0],  # person 20
        ],
        index=people.index,
        columns=cdap_indiv_and_hhsize1.columns,
    )

    pdt.assert_frame_equal(
        individual_utils, expected, check_dtype=False, check_names=False
    )


def test_build_cdap_spec_hhsize2(people, model_settings):

    hhsize = 2
    cdap_indiv_and_hhsize1 = simulate.read_model_spec(
        file_name="cdap_indiv_and_hhsize1.csv"
    )

    interaction_coefficients = pd.read_csv(
        config.config_file_path("cdap_interaction_coefficients.csv"), comment="#"
    )
    interaction_coefficients = cdap.preprocess_interaction_coefficients(
        interaction_coefficients
    )

    person_type_map = model_settings.get("PERSON_TYPE_MAP", {})

    with chunk.chunk_log("test_build_cdap_spec_hhsize2", base=True):
        cdap.assign_cdap_rank(people, person_type_map)
        indiv_utils = cdap.individual_utilities(
            people, cdap_indiv_and_hhsize1, locals_d=None
        )

        choosers = cdap.hh_choosers(indiv_utils, hhsize=hhsize)

        spec = cdap.build_cdap_spec(
            interaction_coefficients, hhsize=hhsize, cache=False
        )

        # pandas.dot depends on column names of expression_values matching spec index values
        # expressions should have been uniquified when spec was read
        assert spec.index.is_unique

        vars = simulate.eval_variables(spec.index, choosers)
        assert (spec.index.values == vars.columns.values).all()

    # spec = spec.astype(np.float64)

    utils = vars.dot(spec)

    expected = pd.DataFrame(
        [
            [0, 3, 0, 3, 7, 3, 0, 3, 0],  # household 3
            [0, 0, 1, 1, 1, 2, 0, 0, 2],  # household 4
        ],
        index=[3, 4],
        columns=["HH", "HM", "HN", "MH", "MM", "MN", "NH", "NM", "NN"],
    ).astype("float")

    pdt.assert_frame_equal(utils, expected, check_names=False)
