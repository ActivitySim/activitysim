# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import os.path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest

from activitysim.core import chunk, simulate, workflow


@pytest.fixture
def data_dir():
    return os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture
def spec_name(data_dir):
    return "sample_spec.csv"


@pytest.fixture
def state(data_dir) -> workflow.State:
    state = workflow.State()
    state.initialize_filesystem(
        working_dir=os.path.dirname(__file__), data_dir=(data_dir,)
    ).default_settings()
    return state


@pytest.fixture
def spec(state, spec_name):
    return state.filesystem.read_model_spec(file_name=spec_name)


@pytest.fixture
def data(data_dir):
    return pd.read_csv(os.path.join(data_dir, "data.csv"))


@pytest.fixture
def nest_spec():
    nest_spec = {
        "name": "root",
        "coefficient": 1.0,
        "alternatives": [
            {"name": "alt0", "coefficient": 0.5, "alternatives": ["alt0.0", "alt0.1"]},
            "alt1",
        ],
    }
    return nest_spec


def test_read_model_spec(state, spec_name):
    spec = state.filesystem.read_model_spec(file_name=spec_name)

    assert len(spec) == 4
    assert spec.index.name == "Expression"
    assert list(spec.columns) == ["alt0", "alt1"]
    npt.assert_array_equal(spec.values, [[1.1, 11], [2.2, 22], [3.3, 33], [4.4, 44]])


def test_eval_variables(state, spec, data):
    result = simulate.eval_variables(state, spec.index, data)

    expected = pd.DataFrame(
        [[1, 0, 4, 1], [0, 1, 4, 1], [0, 1, 5, 1]], index=data.index, columns=spec.index
    )

    expected[expected.columns[0]] = expected[expected.columns[0]].astype(np.int8)
    expected[expected.columns[1]] = expected[expected.columns[1]].astype(np.int8)
    expected[expected.columns[2]] = expected[expected.columns[2]].astype(np.int64)
    expected[expected.columns[3]] = expected[expected.columns[3]].astype(int)

    print("\nexpected\n%s" % expected.dtypes)
    print("\nresult\n%s" % result.dtypes)

    pdt.assert_frame_equal(result, expected, check_names=False)


def test_simple_simulate(state, data, spec):
    state.settings.check_for_variability = False

    choices = simulate.simple_simulate(state, choosers=data, spec=spec, nest_spec=None)
    expected = pd.Series([1, 1, 1], index=data.index)
    pdt.assert_series_equal(choices, expected, check_dtype=False)


def test_simple_simulate_chunked(state, data, spec):
    state.settings.check_for_variability = False
    state.settings.chunk_size = 2
    choices = simulate.simple_simulate(
        state,
        choosers=data,
        spec=spec,
        nest_spec=None,
    )
    expected = pd.Series([1, 1, 1], index=data.index)
    pdt.assert_series_equal(choices, expected, check_dtype=False)


def test_eval_mnl_eet(state):
    # Check that the same counts are returned by eval_mnl when using EET and when not.

    num_choosers = 100_000

    np.random.seed(42)
    data2 = pd.DataFrame(
        {
            "chooser_attr": np.random.rand(num_choosers),
        },
        index=pd.Index(range(num_choosers), name="person_id"),
    )

    spec2 = pd.DataFrame(
        {"alt0": [1.0], "alt1": [2.0]},
        index=pd.Index(["chooser_attr"], name="Expression"),
    )

    # Set up a state with EET enabled
    state.settings.use_explicit_error_terms = True
    state.rng().set_base_seed(42)
    state.rng().add_channel("person_id", data2)
    state.rng().begin_step("test_step_mnl")

    chunk_sizer = chunk.ChunkSizer(state, "", "", num_choosers)

    # run eval_mnl with EET enabled
    choices_eet = simulate.eval_mnl(
        state=state,
        choosers=data2,
        spec=spec2,
        locals_d=None,
        custom_chooser=None,
        estimator=None,
        chunk_sizer=chunk_sizer,
    )

    # Reset the state, without EET enabled
    state.settings.use_explicit_error_terms = False

    state.rng().end_step("test_step_mnl")
    state.rng().begin_step("test_step_mnl")

    choices_mnl = simulate.eval_mnl(
        state=state,
        choosers=data2,
        spec=spec2,
        locals_d=None,
        custom_chooser=None,
        estimator=None,
        chunk_sizer=chunk_sizer,
    )

    # Compare counts
    mnl_counts = choices_mnl.value_counts(normalize=True)
    explicit_counts = choices_eet.value_counts(normalize=True)
    assert np.allclose(mnl_counts, explicit_counts, atol=0.01)


def test_eval_nl_eet(state, nest_spec):
    # Check that the same counts are returned by eval_nl when using EET and when not.

    num_choosers = 100_000

    np.random.seed(42)
    data2 = pd.DataFrame(
        {
            "chooser_attr": np.random.rand(num_choosers),
        },
        index=pd.Index(range(num_choosers), name="person_id"),
    )

    spec2 = pd.DataFrame(
        {"alt1": [2.0], "alt0.0": [0.5], "alt0.1": [0.2]},
        index=pd.Index(["chooser_attr"], name="Expression"),
    )

    # Set up a state with EET enabled
    state.settings.use_explicit_error_terms = True
    state.rng().set_base_seed(42)
    state.rng().add_channel("person_id", data2)
    state.rng().begin_step("test_step_mnl")

    chunk_sizer = chunk.ChunkSizer(state, "", "", num_choosers)

    # run eval_nl with EET enabled
    choices_eet = simulate.eval_nl(
        state=state,
        choosers=data2,
        spec=spec2,
        nest_spec=nest_spec,
        locals_d={},
        custom_chooser=None,
        estimator=None,
        trace_label="test",
        chunk_sizer=chunk_sizer,
    )

    # Reset the state, without EET enabled
    state.settings.use_explicit_error_terms = False

    state.rng().end_step("test_step_mnl")
    state.rng().begin_step("test_step_mnl")

    choices_mnl = simulate.eval_nl(
        state=state,
        choosers=data2,
        spec=spec2,
        nest_spec=nest_spec,
        locals_d={},
        custom_chooser=None,
        trace_label="test",
        estimator=None,
        chunk_sizer=chunk_sizer,
    )

    # Compare counts
    mnl_counts = choices_mnl.value_counts(normalize=True)
    explicit_counts = choices_eet.value_counts(normalize=True)
    assert np.allclose(mnl_counts, explicit_counts, atol=0.01)


def test_compute_nested_utilities(nest_spec):
    # computes nested utilities manually and using the function and checks that
    # the utilities are the same

    num_choosers = 2
    raw_utilities = pd.DataFrame(
        {"alt1": [1, 10], "alt0.0": [2, 3], "alt0.1": [4, 5]},
        index=pd.Index(range(num_choosers)),
    )

    nested_utilities = simulate.compute_nested_utilities(raw_utilities, nest_spec)

    # these are from the definition of nest_spec
    alt0_nest_coefficient = nest_spec["alternatives"][0]["coefficient"]
    alt0_leaf_product_of_coefficients = nest_spec["coefficient"] * alt0_nest_coefficient
    assert alt0_leaf_product_of_coefficients == 0.5  # 1 * 0.5

    product_of_coefficientss = pd.DataFrame(
        {
            "alt1": [nest_spec["coefficient"]],
            "alt0.0": [alt0_leaf_product_of_coefficients],
            "alt0.1": [alt0_leaf_product_of_coefficients],
        },
        index=[0],
    )
    leaf_utilities = raw_utilities / product_of_coefficientss.iloc[0]

    constructed_nested_utilities = pd.DataFrame(index=raw_utilities.index)

    constructed_nested_utilities[leaf_utilities.columns] = leaf_utilities
    constructed_nested_utilities["alt0"] = alt0_nest_coefficient * np.log(
        np.exp(leaf_utilities[["alt0.0", "alt0.1"]]).sum(axis=1)
    )
    constructed_nested_utilities["root"] = nest_spec["coefficient"] * np.log(
        np.exp(constructed_nested_utilities[["alt1", "alt0"]]).sum(axis=1)
    )

    assert np.allclose(
        nested_utilities, constructed_nested_utilities[nested_utilities.columns]
    ), "Mismatch in nested utilities"


def test_eval_nl_logsums_eet_vs_non_eet(state, nest_spec):
    """eval_nl with want_logsums=True must produce identical logsums under
    EET and non-EET modes"""

    num_choosers = 100

    np.random.seed(42)
    data2 = pd.DataFrame(
        {"chooser_attr": np.random.rand(num_choosers)},
        index=pd.Index(range(num_choosers), name="person_id"),
    )

    spec2 = pd.DataFrame(
        {"alt1": [2.0], "alt0.0": [0.5], "alt0.1": [0.2]},
        index=pd.Index(["chooser_attr"], name="Expression"),
    )

    chunk_sizer = chunk.ChunkSizer(state, "", "", num_choosers)

    state.settings.use_explicit_error_terms = True
    state.rng().set_base_seed(42)
    state.rng().add_channel("person_id", data2)
    state.rng().begin_step("test_step_logsums")

    result_eet = simulate.eval_nl(
        state=state,
        choosers=data2,
        spec=spec2,
        nest_spec=nest_spec,
        locals_d={},
        custom_chooser=None,
        estimator=None,
        want_logsums=True,
        trace_label="test",
        chunk_sizer=chunk_sizer,
    )

    state.rng().end_step("test_step_logsums")

    state.settings.use_explicit_error_terms = False
    state.rng().begin_step("test_step_logsums")

    result_non_eet = simulate.eval_nl(
        state=state,
        choosers=data2,
        spec=spec2,
        nest_spec=nest_spec,
        locals_d={},
        custom_chooser=None,
        estimator=None,
        want_logsums=True,
        trace_label="test",
        chunk_sizer=chunk_sizer,
    )

    state.rng().end_step("test_step_logsums")

    # Both paths should return a DataFrame with 'choice' and 'logsum' columns
    assert "logsum" in result_eet.columns, "EET result missing logsum column"
    assert "logsum" in result_non_eet.columns, "non-EET result missing logsum column"

    # Logsums are deterministic — they must be identical across paths
    assert np.allclose(
        result_eet["logsum"].values, result_non_eet["logsum"].values, rtol=1e-10
    )
