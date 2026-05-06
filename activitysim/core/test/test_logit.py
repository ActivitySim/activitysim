# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import os.path
import re

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from activitysim.core import logit, random, simulate, workflow
from activitysim.core.exceptions import InvalidTravelError
from activitysim.core.logit import AltsContext, add_ev1_random
from activitysim.core.simulate import eval_variables


@pytest.fixture(scope="module")
def data_dir():
    return os.path.join(os.path.dirname(__file__), "data")


# this is lifted straight from urbansim's test_mnl.py
@pytest.fixture(
    scope="module",
    params=[
        (
            "fish.csv",
            "fish_choosers.csv",
            pd.DataFrame(
                [[-0.02047652], [0.95309824]], index=["price", "catch"], columns=["Alt"]
            ),
            pd.DataFrame(
                [
                    [0.2849598, 0.2742482, 0.1605457, 0.2802463],
                    [0.1498991, 0.4542377, 0.2600969, 0.1357664],
                ],
                columns=["beach", "boat", "charter", "pier"],
            ),
        )
    ],
)
def test_data(request):
    data, choosers, spec, probabilities = request.param
    return {
        "data": data,
        "choosers": choosers,
        "spec": spec,
        "probabilities": probabilities,
    }


@pytest.fixture
def choosers(test_data, data_dir):
    filen = os.path.join(data_dir, test_data["choosers"])
    return pd.read_csv(filen)


@pytest.fixture
def spec(test_data):
    return test_data["spec"]


@pytest.fixture
def utilities(choosers, spec, test_data):
    state = workflow.State().default_settings()
    vars = eval_variables(state, spec.index, choosers)
    utils = vars.dot(spec).astype("float")
    return pd.DataFrame(
        utils.values.reshape(test_data["probabilities"].shape),
        columns=test_data["probabilities"].columns,
    )


@pytest.fixture(scope="module")
def interaction_choosers():
    return pd.DataFrame({"attr": ["a", "b", "c", "b"]}, index=["w", "x", "y", "z"])


@pytest.fixture(scope="module")
def interaction_alts():
    return pd.DataFrame({"prop": [10, 20, 30, 40]}, index=[1, 2, 3, 4])


#
# Utility Validation Tests
#
def test_validate_utils_replaces_unavailable_values():
    state = workflow.State().default_settings()
    utils = pd.DataFrame([[0.0, logit.UTIL_MIN - 1.0], [1.0, 2.0]])

    validated = logit.validate_utils(state, utils, allow_zero_probs=False)

    assert validated.iloc[0, 0] == pytest.approx(0.0)
    assert validated.iloc[0, 1] == pytest.approx(logit.UTIL_UNAVAILABLE)
    assert validated.iloc[1, 0] == pytest.approx(1.0)
    assert validated.iloc[1, 1] == pytest.approx(2.0)


def test_validate_utils_raises_when_all_unavailable():
    state = workflow.State().default_settings()
    utils = pd.DataFrame([[logit.UTIL_MIN - 1.0, logit.UTIL_MIN - 2.0]])

    with pytest.raises(InvalidTravelError) as excinfo:
        logit.validate_utils(state, utils, allow_zero_probs=False)

    assert "all probabilities are zero" in str(excinfo.value)


def test_validate_utils_allows_zero_probs():
    state = workflow.State().default_settings()
    utils = pd.DataFrame([[0.5, logit.UTIL_MIN - 1.0]])

    validated = logit.validate_utils(state, utils, allow_zero_probs=True)

    assert validated.iloc[0, 0] == 0.5
    assert validated.iloc[0, 1] == logit.UTIL_UNAVAILABLE


#
# `utils_to_probs` Tests
#
def test_utils_to_probs_logsums_with_overflow_protection():
    state = workflow.State().default_settings()
    utils = pd.DataFrame(
        [[1000.0, 1001.0, 999.0], [-1000.0, -1001.0, -999.0]],
        columns=["a", "b", "c"],
    )
    original_utils = utils.copy()

    probs, logsums = logit.utils_to_probs(
        state,
        utils,
        trace_label=None,
        overflow_protection=True,
        return_logsums=True,
    )

    utils_np = original_utils.to_numpy()
    row_max = utils_np.max(axis=1, keepdims=True)
    exp_shifted = np.exp(utils_np - row_max)
    expected_probs = exp_shifted / exp_shifted.sum(axis=1, keepdims=True)
    expected_logsums = pd.Series(
        np.log(exp_shifted.sum(axis=1)) + row_max.squeeze(),
        index=utils.index,
    )

    pdt.assert_frame_equal(
        probs,
        pd.DataFrame(expected_probs, index=utils.index, columns=utils.columns),
        rtol=1.0e-7,
        atol=0.0,
    )
    pdt.assert_series_equal(logsums, expected_logsums, rtol=1.0e-7, atol=0.0)


def test_utils_to_probs_warns_on_zero_probs_overflow():
    state = workflow.State().default_settings()
    utils = pd.DataFrame(
        [[logit.UTIL_MIN - 1.0, logit.UTIL_MIN - 2.0], [0.0, 0.0]],
        columns=["a", "b"],
    )

    with pytest.warns(UserWarning, match="cannot set overflow_protection"):
        probs = logit.utils_to_probs(
            state,
            utils,
            trace_label=None,
            allow_zero_probs=True,
            overflow_protection=True,
        )

    assert (probs.iloc[0] == 0.0).all()
    assert probs.iloc[1].sum() == pytest.approx(1.0)
    assert probs.iloc[1].iloc[0] == pytest.approx(0.5)
    assert probs.iloc[1].iloc[1] == pytest.approx(0.5)


def test_utils_to_probs_raises_on_float32_zero_probs_overflow():
    state = workflow.State().default_settings()
    utils = pd.DataFrame(np.array([[90.0, 0.0]], dtype=np.float32))

    with pytest.raises(ValueError, match="cannot prevent expected overflow"):
        logit.utils_to_probs(
            state,
            utils,
            trace_label=None,
            allow_zero_probs=True,
            overflow_protection=True,
        )


def test_utils_to_probs(utilities, test_data):
    state = workflow.State().default_settings()
    probs = logit.utils_to_probs(state, utilities, trace_label=None)
    pdt.assert_frame_equal(probs, test_data["probabilities"])


def test_utils_to_probs_raises():
    state = workflow.State().default_settings()
    idx = pd.Index(name="household_id", data=[1])
    with pytest.raises(RuntimeError) as excinfo:
        logit.utils_to_probs(
            state,
            pd.DataFrame([[1, 2, np.inf, 3]], index=idx),
            trace_label=None,
            overflow_protection=False,
        )
    assert "infinite exponentiated utilities" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        logit.utils_to_probs(
            state,
            pd.DataFrame([[1, 2, 9999, 3]], index=idx),
            trace_label=None,
            overflow_protection=False,
        )
    assert "infinite exponentiated utilities" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        logit.utils_to_probs(
            state,
            pd.DataFrame([[-999, -999, -999, -999]], index=idx),
            trace_label=None,
            overflow_protection=False,
        )
    assert "all probabilities are zero" in str(excinfo.value)

    # test that overflow protection works
    z = logit.utils_to_probs(
        state,
        pd.DataFrame([[1, 2, 9999, 3]], index=idx),
        trace_label=None,
        overflow_protection=True,
    )
    assert np.asarray(z).ravel() == pytest.approx(np.asarray([0.0, 0.0, 1.0, 0.0]))


#
# `make_choices` Tests
#
def test_make_choices_only_one():
    state = workflow.State().default_settings()
    probs = pd.DataFrame(
        [[1, 0, 0], [0, 1, 0]], columns=["a", "b", "c"], index=["x", "y"]
    )
    choices, rands = logit.make_choices(state, probs)

    pdt.assert_series_equal(
        choices, pd.Series([0, 1], index=["x", "y"]), check_dtype=False
    )


def test_make_choices_real_probs(utilities):
    state = workflow.State().default_settings()
    probs = logit.utils_to_probs(state, utilities, trace_label=None)
    choices, rands = logit.make_choices(state, probs)

    pdt.assert_series_equal(
        choices,
        pd.Series([1, 2], index=[0, 1]),
        check_dtype=False,
    )


def test_different_order_make_choices():
    # check if, when we shuffle utilities, make_choices chooses the same alternatives
    state = workflow.State().default_settings()

    # increase number of choosers and alternatives for realism
    n_choosers = 100
    n_alts = 50
    data = np.random.rand(n_choosers, n_alts)
    chooser_ids = np.arange(n_choosers)
    alt_ids = [f"alt_{i}" for i in range(n_alts)]

    utilities = pd.DataFrame(
        data,
        index=pd.Index(chooser_ids, name="chooser_id"),
        columns=alt_ids,
    )

    # We need a stable RNG that gives the same random numbers for the same chooser_id
    # regardless of row order. ActivitySim's random.Random does this.
    state.get_rn_generator().add_channel("chooser_id", utilities)
    state.get_rn_generator().begin_step("test_step")

    probs = logit.utils_to_probs(state, utilities, trace_label=None)
    choices, rands = logit.make_choices(state, probs)

    # shuffle utilities (rows) and make_choices again
    # We must reset the step offset so the RNG produces the same sequence for the same IDs
    state.get_rn_generator().end_step("test_step")
    state.get_rn_generator().begin_step("test_step")
    utilities_shuffled = utilities.sample(frac=1, random_state=42)
    probs_shuffled = logit.utils_to_probs(state, utilities_shuffled, trace_label=None)
    choices_shuffled, rands_shuffled = logit.make_choices(state, probs_shuffled)

    # sorting both to ensure comparison is on the same index order
    pdt.assert_series_equal(
        choices.sort_index(), choices_shuffled.sort_index(), check_dtype=False
    )


def test_make_choices_matches_random_draws():
    class DummyRNG:
        def random_for_df(self, df, n=1):
            assert n == 1
            return np.array([[0.05], [0.6], [0.95]])

    class DummyState:
        @staticmethod
        def get_rn_generator():
            return DummyRNG()

    state = DummyState()
    probs = pd.DataFrame(
        [[0.1, 0.2, 0.7], [0.4, 0.4, 0.2], [0.05, 0.9, 0.05]],
        index=["a", "b", "c"],
        columns=["x", "y", "z"],
    )
    choices, rands = logit.make_choices(state, probs)

    expected_rands = np.array([0.05, 0.6, 0.95])
    expected_choices = np.array([0, 1, 1])

    pdt.assert_series_equal(
        rands,
        pd.Series(expected_rands, index=probs.index),
        check_names=False,
    )
    pdt.assert_series_equal(
        choices,
        pd.Series(expected_choices, index=probs.index),
        check_dtype=False,
    )


#
# EV1 Random Tests
#
def test_add_ev1_random():
    class DummyRNG:
        def gumbel_for_df(self, df, n):
            # Deterministic, non-constant draws make it easy to verify
            # correct per-row/per-column addition behavior.
            row_component = df.index.to_numpy(dtype=float).reshape(-1, 1) / 10.0
            col_component = np.arange(n, dtype=float).reshape(1, -1)
            return row_component + col_component

    rng = DummyRNG()

    class DummyState:
        @staticmethod
        def get_rn_generator():
            return rng

    utilities = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=[10, 11],
        columns=["a", "b"],
    )

    randomized = logit.add_ev1_random(DummyState(), utilities)

    expected = pd.DataFrame(
        [[2.0, 4.0], [4.1, 6.1]],
        index=[10, 11],
        columns=["a", "b"],
    )

    # check that the random component was added correctly, and that the original utilities were not mutated
    pdt.assert_frame_equal(randomized, expected)
    pdt.assert_index_equal(randomized.index, utilities.index)
    pdt.assert_index_equal(randomized.columns, utilities.columns)
    pdt.assert_frame_equal(
        utilities,
        pd.DataFrame(
            [[1.0, 2.0], [3.0, 4.0]],
            index=[10, 11],
            columns=["a", "b"],
        ),
    )


def test_add_ev1_random_requires_paired_alt_context_args():
    class DummyRNG:
        def gumbel_for_df(self, df, n):
            return np.zeros((len(df), n))

    class DummyState:
        @staticmethod
        def get_rn_generator():
            return DummyRNG()

    utilities = pd.DataFrame([[1.0, 2.0]], index=[1], columns=["a", "b"])

    with pytest.raises(
        AssertionError,
        match="alt_info and alt_nrs_df must both be provided or omitted together",
    ):
        logit.add_ev1_random(
            DummyState(),
            utilities,
            alt_info=AltsContext.from_num_alts(2),
            alt_nrs_df=None,
        )

#
# EET Choice Behavior Tests
#
def test_make_choices_eet_mnl(monkeypatch):
    def fake_add_ev1_random(_state, _df, alt_info=None, alt_nrs_df=None):
        return pd.DataFrame(
            [[1.0, 3.0], [4.0, 2.0]],
            index=[100, 101],
            columns=["a", "b"],
        )

    monkeypatch.setattr(logit, "add_ev1_random", fake_add_ev1_random)

    choices = logit.make_choices_explicit_error_term_mnl(
        workflow.State().default_settings(),
        pd.DataFrame([[0.0, 0.0], [0.0, 0.0]], index=[100, 101], columns=["a", "b"]),
        trace_label=None,
    )

    pdt.assert_series_equal(choices, pd.Series([1, 0], index=[100, 101]))


def test_make_choices_eet_nl(monkeypatch):
    def fake_sample_nested_logit_exact_leaf_error_terms(_state, df, nest_spec):
        assert nest_spec["name"] == "root"
        assert list(df.columns) == ["walk", "car", "bus"]

        error_terms = pd.DataFrame(0.0, index=df.index, columns=df.columns)
        error_terms.loc[10, ["walk", "car", "bus"]] = [1.0, 5.0, 3.0]
        error_terms.loc[11, ["walk", "car", "bus"]] = [4.0, 2.0, 3.0]
        return error_terms

    monkeypatch.setattr(
        logit,
        "sample_nested_logit_exact_leaf_error_terms",
        fake_sample_nested_logit_exact_leaf_error_terms,
    )

    nest_spec = {
        "name": "root",
        "coefficient": 1.0,
        "alternatives": [
            {"name": "motorized", "coefficient": 0.7, "alternatives": ["car", "bus"]},
            "walk",
        ],
    }

    state = workflow.State().default_settings()
    monkeypatch.setattr(state.tracing, "trace_df", lambda *args, **kwargs: None)

    choices = logit.make_choices_explicit_error_term_nl(
        state,
        pd.DataFrame(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            index=[10, 11],
            columns=["walk", "car", "bus"],
        ),
        nest_spec,
        trace_label="test",
    )

    pdt.assert_series_equal(choices, pd.Series([1, 0], index=[10, 11]))


def test_sample_nested_logit_exact_leaf_error_terms_accumulates_node_and_leaf_terms(
    monkeypatch,
):
    stable_draws = np.array([0.4, -0.2], dtype=np.float64)

    def fake_log_positive_stable_for_df(_state, df, alpha):
        assert alpha == pytest.approx(0.5)
        assert list(df.columns) == ["car", "bus", "walk"]
        return stable_draws

    monkeypatch.setattr(
        logit, "_log_positive_stable_for_df", fake_log_positive_stable_for_df
    )

    class DummyRNG:
        @staticmethod
        def gumbel_for_df(df, n):
            assert n == df.shape[1]
            return np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)

    class DummyState:
        @staticmethod
        def get_rn_generator():
            return DummyRNG()

    nest_spec = {
        "name": "root",
        "coefficient": 1.0,
        "alternatives": [
            {"name": "motorized", "coefficient": 0.5, "alternatives": ["car", "bus"]},
            "walk",
        ],
    }
    alt_utilities = pd.DataFrame(
        0.0,
        index=pd.Index([10, 11], name="chooser_id"),
        columns=["car", "bus", "walk"],
        dtype=np.float64,
    )

    error_terms = logit.sample_nested_logit_exact_leaf_error_terms(
        DummyState(), alt_utilities, nest_spec
    )

    expected = pd.DataFrame(
        {
            "car": [0.7, 1.9],
            "bus": [1.2, 2.4],
            "walk": [3.0, 6.0],
        },
        index=alt_utilities.index,
        dtype=np.float64,
    )

    pdt.assert_frame_equal(error_terms, expected)


def test_make_choices_utility_based_sets_zero_rands(monkeypatch):
    def fake_add_ev1_random(_state, df, alt_info=None, alt_nrs_df=None):
        return pd.DataFrame(
            [[2.0, 1.0], [0.5, 2.5]],
            index=df.index,
            columns=df.columns,
        )

    monkeypatch.setattr(logit, "add_ev1_random", fake_add_ev1_random)

    utilities = pd.DataFrame([[3.0, 2.0], [1.0, 4.0]], index=[11, 12])
    choices, rands = logit.make_choices_utility_based(
        workflow.State().default_settings(),
        utilities,
        nest_spec=None,
        trace_label=None,
    )

    expected_choices = pd.Series([0, 1], index=[11, 12])
    pdt.assert_series_equal(choices, expected_choices)
    pdt.assert_series_equal(rands, pd.Series([0, 0], index=[11, 12]))


#
# EET vs non-EET Choice Behavior Tests
#
def test_make_choices_vs_eet_same_distribution():
    """With many draws, make_choices (probability-based) and
    make_choices_explicit_error_term_mnl (EET) should produce roughly the
    same empirical choice-frequency distribution for the same utilities."""
    n_draws = 1_000_000
    a_tol = 0.001
    r_tol = 0.01
    utils_values = [5.0, 6.0, 7.0, 8.0, 9.0]
    n_alts = len(utils_values)
    columns = ["a", "b", "c", "d", "e"]

    utils = pd.DataFrame([utils_values] * n_draws, columns=columns)

    # Probability-based (Monte Carlo) path — independent RNG
    mc_rng = np.random.default_rng(42)

    class MCDummyRNG:
        def random_for_df(self, df, n=1):
            return mc_rng.random((len(df), n))

    class MCDummyState:
        @staticmethod
        def get_rn_generator():
            return MCDummyRNG()

    probs = logit.utils_to_probs(
        MCDummyState(), utils, trace_label=None, overflow_protection=True
    )
    choices_mc, _ = logit.make_choices(MCDummyState(), probs, trace_label=None)

    # Explicit-error-term (EET) path — independent RNG
    eet_rng = np.random.default_rng(123)

    class EETDummyRNG:
        def random_for_df(self, df, n=1):
            return eet_rng.random((len(df), n))

        def gumbel_for_df(self, df, n):
            return eet_rng.gumbel(size=(len(df), n))

    class EETDummyState:
        @staticmethod
        def get_rn_generator():
            return EETDummyRNG()

    choices_eet = logit.make_choices_explicit_error_term_mnl(
        EETDummyState(), utils, trace_label=None
    )

    mc_fracs = np.bincount(choices_mc.values.astype(int), minlength=n_alts) / n_draws
    eet_fracs = np.bincount(choices_eet.values.astype(int), minlength=n_alts) / n_draws

    np.testing.assert_allclose(mc_fracs, eet_fracs, atol=a_tol, rtol=r_tol)
    np.testing.assert_allclose(
        mc_fracs, probs.iloc[0].to_numpy(), atol=a_tol, rtol=r_tol
    )
    np.testing.assert_allclose(
        eet_fracs, probs.iloc[0].to_numpy(), atol=a_tol, rtol=r_tol
    )


def test_make_choices_vs_eet_nl_same_distribution():
    """With many draws, nested logit choices via probabilities and
    nested logit choices via EET should produce the same empirical distribution."""
    n_draws = 100_000
    a_tol = 0.01

    nest_spec = {
        "name": "root",
        "coefficient": 1.0,
        "alternatives": [
            {"name": "motorized", "coefficient": 0.5, "alternatives": ["car", "bus"]},
            "walk",
        ],
    }
    # Utilities for car, bus, walk
    # For NL, we need utilities for all nodes in the tree for EET,
    # but for probability-based choice we usually use the flattened/logsummed probabilities.
    # To compare them fairly, we use the same base utilities.
    # car=0.5, bus=0.2, walk=0.4
    leaf_utilities = pd.DataFrame(
        [[0.5, 0.2, 0.4]],
        columns=["car", "bus", "walk"],
    )
    utils_df = pd.concat([leaf_utilities] * n_draws, ignore_index=True)

    # 1. Probability-based Nested Logit choices
    mc_rng = np.random.default_rng(42)

    class MCDummyRNG:
        def random_for_df(self, df, n=1):
            return mc_rng.random((len(df), n))

    class MCDummyState:
        @staticmethod
        def get_rn_generator():
            return MCDummyRNG()

        def default_settings(self):
            return self

    # Compute probabilities for NL using simulation logic
    nested_exp_utilities = simulate.compute_nested_exp_utilities(utils_df, nest_spec)
    nested_probabilities = simulate.compute_nested_probabilities(
        MCDummyState(), nested_exp_utilities, nest_spec, trace_label=None
    )
    probs = simulate.compute_base_probabilities(
        nested_probabilities, nest_spec, utils_df
    )
    choices_mc, _ = logit.make_choices(MCDummyState(), probs, trace_label=None)

    # 2. EET-based Nested Logit choices
    eet_rng = np.random.default_rng(123)

    class EETDummyRNG:
        def random_for_df(self, df, n=1):
            return eet_rng.random((len(df), n))

        def gumbel_for_df(self, df, n):
            return eet_rng.gumbel(size=(len(df), n))

    class EETDummyState:
        @staticmethod
        def get_rn_generator():
            return EETDummyRNG()

        def default_settings(self):
            return self

        @property
        def tracing(self):
            import activitysim.core.tracing as tracing

            return tracing

    choices_eet = logit.make_choices_explicit_error_term_nl(
        EETDummyState(),
        utils_df,
        nest_spec,
        trace_label=None,
    )

    mc_fracs = np.bincount(choices_mc.values.astype(int), minlength=3) / n_draws
    eet_fracs = np.bincount(choices_eet.values.astype(int), minlength=3) / n_draws

    # They should be close
    np.testing.assert_allclose(mc_fracs, eet_fracs, atol=a_tol)


def _repeated_utility_df(raw_utilities: pd.Series, n_draws: int) -> pd.DataFrame:
    raw_utilities = pd.Series(raw_utilities, dtype=float)
    return pd.DataFrame(
        np.repeat(raw_utilities.to_numpy()[None, :], n_draws, axis=0),
        columns=raw_utilities.index,
        index=pd.RangeIndex(n_draws, name="chooser_id"),
    )


def _make_rng_state(
    df: pd.DataFrame,
    seed: int,
    step_name: str,
) -> workflow.State:
    state = workflow.State().default_settings()
    rng = state.get_rn_generator()
    rng.set_base_seed(seed)
    rng.add_channel(df.index.name, df)
    rng.begin_step(step_name)
    return state


def _finish_rng_state(state: workflow.State, step_name: str) -> None:
    state.get_rn_generator().end_step(step_name)


def _choice_shares(choices: pd.Series, alt_names) -> pd.Series:
    alt_names = pd.Index(alt_names)
    counts = np.bincount(choices.to_numpy(dtype=int), minlength=len(alt_names))
    return pd.Series(counts / counts.sum(), index=alt_names)


def _expected_nested_logit_shares(
    raw_utilities: pd.Series,
    nest_spec: dict,
    seed: int = 42,
) -> pd.Series:
    raw_df = _repeated_utility_df(raw_utilities, n_draws=1)
    step_name = f"expected_nested_logit_{len(raw_utilities)}_seed_{seed}"
    state = _make_rng_state(raw_df, seed=seed, step_name=step_name)
    try:
        nested_exp_utilities = simulate.compute_nested_exp_utilities(raw_df, nest_spec)
        nested_probabilities = simulate.compute_nested_probabilities(
            state, nested_exp_utilities, nest_spec, trace_label=None
        )
        base_probabilities = simulate.compute_base_probabilities(
            nested_probabilities, nest_spec, raw_df
        )
    finally:
        _finish_rng_state(state, step_name)

    return base_probabilities.iloc[0]


def _nested_logit_eet_shares(
    raw_utilities: pd.Series,
    nest_spec: dict,
    n_draws: int,
    seed: int = 42,
) -> pd.Series:
    raw_df = _repeated_utility_df(raw_utilities, n_draws=n_draws)
    step_name = f"nested_eet_exact_leaf_{n_draws}_{len(raw_utilities)}"
    state = _make_rng_state(raw_df, seed=seed, step_name=step_name)
    try:
        choices = logit.make_choices_explicit_error_term_nl(
            state,
            raw_df,
            nest_spec,
            trace_label=None,
        )
    finally:
        _finish_rng_state(state, step_name)

    return _choice_shares(choices, raw_df.columns)


def _nested_logit_mc_shares(
    raw_utilities: pd.Series,
    nest_spec: dict,
    n_draws: int,
    seed: int = 42,
) -> pd.Series:
    raw_df = _repeated_utility_df(raw_utilities, n_draws=n_draws)
    step_name = f"nested_mc_{n_draws}_{len(raw_utilities)}"
    state = _make_rng_state(raw_df, seed=seed, step_name=step_name)
    try:
        nested_exp_utilities = simulate.compute_nested_exp_utilities(raw_df, nest_spec)
        nested_probabilities = simulate.compute_nested_probabilities(
            state, nested_exp_utilities, nest_spec, trace_label=None
        )
        base_probabilities = simulate.compute_base_probabilities(
            nested_probabilities, nest_spec, raw_df
        )
        choices, _ = logit.make_choices(state, base_probabilities, trace_label=None)
    finally:
        _finish_rng_state(state, step_name)

    return _choice_shares(choices, raw_df.columns)


def _assert_empirical_shares_close(
    observed: pd.Series,
    expected: pd.Series,
    n_draws: int,
    sigma_multiplier: float = 6.0,
    variance_floor: float = 0.02,
) -> None:
    expected = expected.reindex(observed.index)
    tolerances = sigma_multiplier * np.sqrt(
        np.maximum(expected * (1.0 - expected), variance_floor) / n_draws
    )
    differences = (observed - expected).abs()
    assert (differences <= tolerances).all(), pd.DataFrame(
        {
            "observed": observed,
            "expected": expected,
            "abs_diff": differences,
            "tolerance": tolerances,
        }
    ).to_string()


def _nested_logit_method_share_matrix(
    raw_utilities: pd.Series,
    nest_spec: dict,
    method: str,
    n_draws: int,
    seeds: list[int],
) -> np.ndarray:
    share_samples = []
    for seed in seeds:
        if method == "mc":
            shares = _nested_logit_mc_shares(
                raw_utilities,
                nest_spec,
                n_draws=n_draws,
                seed=seed,
            )
        elif method == "exact_leaf":
            shares = _nested_logit_eet_shares(
                raw_utilities,
                nest_spec,
                n_draws=n_draws,
                seed=seed,
            )
        else:
            raise ValueError(f"unknown nested-logit share method: {method}")
        share_samples.append(shares.to_numpy())

    return np.vstack(share_samples)


def _assert_average_empirical_shares_close(
    observed_matrix: np.ndarray,
    expected: pd.Series,
    n_draws: int,
    sigma_multiplier: float = 6.0,
    variance_floor: float = 0.02,
) -> None:
    expected = expected.astype(float)
    mean_observed = pd.Series(observed_matrix.mean(axis=0), index=expected.index)
    effective_draws = n_draws * observed_matrix.shape[0]
    tolerances = sigma_multiplier * np.sqrt(
        np.maximum(expected * (1.0 - expected), variance_floor) / effective_draws
    )
    differences = (mean_observed - expected).abs()
    assert (differences <= tolerances).all(), pd.DataFrame(
        {
            "mean_observed": mean_observed,
            "expected": expected,
            "abs_diff": differences,
            "tolerance": tolerances,
        }
    ).to_string()


def _assert_average_share_deltas_close(
    baseline_matrix: np.ndarray,
    perturbed_matrix: np.ndarray,
    baseline_expected: pd.Series,
    perturbed_expected: pd.Series,
    n_draws: int,
    sigma_multiplier: float = 6.0,
    variance_floor: float = 0.02,
) -> None:
    observed_delta = pd.Series(
        perturbed_matrix.mean(axis=0) - baseline_matrix.mean(axis=0),
        index=baseline_expected.index,
    )
    expected_delta = perturbed_expected - baseline_expected
    effective_draws = n_draws * baseline_matrix.shape[0]
    variances = (
        np.maximum(baseline_expected * (1.0 - baseline_expected), variance_floor)
        + np.maximum(perturbed_expected * (1.0 - perturbed_expected), variance_floor)
    ) / effective_draws
    tolerances = sigma_multiplier * np.sqrt(variances)
    differences = (observed_delta - expected_delta).abs()
    assert (differences <= tolerances).all(), pd.DataFrame(
        {
            "observed_delta": observed_delta,
            "expected_delta": expected_delta,
            "abs_diff": differences,
            "tolerance": tolerances,
        }
    ).to_string()


def _assert_nested_logit_methods_match_expected_across_seeds(
    raw_utilities: pd.Series,
    nest_spec: dict,
    n_draws: int,
    seeds: list[int],
    methods: tuple[str, ...] = ("mc", "exact_leaf"),
) -> dict[str, np.ndarray]:
    expected = _expected_nested_logit_shares(raw_utilities, nest_spec)
    share_matrices: dict[str, np.ndarray] = {}
    for method in methods:
        share_matrix = _nested_logit_method_share_matrix(
            raw_utilities,
            nest_spec,
            method=method,
            n_draws=n_draws,
            seeds=seeds,
        )
        _assert_average_empirical_shares_close(share_matrix, expected, n_draws=n_draws)
        share_matrices[method] = share_matrix

    for i, left_method in enumerate(methods):
        for right_method in methods[i + 1 :]:
            left_mean = pd.Series(
                share_matrices[left_method].mean(axis=0),
                index=raw_utilities.index.to_numpy(),
            )
            right_mean = pd.Series(
                share_matrices[right_method].mean(axis=0),
                index=raw_utilities.index.to_numpy(),
            )
            tolerances = 8.0 * np.sqrt(
                2.0
                * np.maximum(expected * (1.0 - expected), 0.02)
                / (n_draws * len(seeds))
            )
            differences = (left_mean - right_mean).abs()
            assert (differences <= tolerances).all(), pd.DataFrame(
                {
                    "left_method": left_method,
                    "right_method": right_method,
                    "left_mean": left_mean,
                    "right_mean": right_mean,
                    "abs_diff": differences,
                    "tolerance": tolerances,
                }
            ).to_string()

    return share_matrices


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _estimate_power_law_slope(draw_counts: np.ndarray, errors: np.ndarray) -> float:
    clipped_errors = np.clip(errors.astype(float), np.finfo(float).eps, None)
    slope, _intercept = np.polyfit(
        np.log(draw_counts.astype(float)), np.log(clipped_errors), deg=1
    )
    return float(slope)


def _assert_three_level_nested_logit_methods_follow_power_law(
    draw_counts: np.ndarray,
    seeds: list[int],
    slope_lower: float = -0.8,
    slope_upper: float = -0.2,
    pair_slope_lower: float | None = None,
    pair_slope_upper: float | None = None,
    max_final_method_error: float | None = None,
    max_final_pair_error: float | None = None,
) -> None:
    if pair_slope_lower is None:
        pair_slope_lower = slope_lower
    if pair_slope_upper is None:
        pair_slope_upper = slope_upper

    method_names = ["mc", "exact_leaf"]
    pair_names = [("mc", "exact_leaf")]

    nest_spec = {
        "name": "root",
        "coefficient": 1.0,
        "alternatives": [
            {
                "name": "AUTO",
                "coefficient": 0.72,
                "alternatives": [
                    {
                        "name": "DRIVEALONE",
                        "coefficient": 0.35,
                        "alternatives": ["DA_FREE", "DA_PAY"],
                    }
                ],
            },
            {
                "name": "TRANSIT",
                "coefficient": 0.72,
                "alternatives": [
                    {
                        "name": "WALKACCESS",
                        "coefficient": 0.50,
                        "alternatives": ["WALK_LOC", "WALK_EXP"],
                    }
                ],
            },
            {
                "name": "NONMOTORIZED",
                "coefficient": 0.72,
                "alternatives": ["WALK"],
            },
        ],
    }
    raw_utilities = pd.Series(
        {
            "DA_FREE": 1.4,
            "DA_PAY": 0.9,
            "WALK_LOC": 0.5,
            "WALK_EXP": 0.2,
            "WALK": 0.0,
        }
    )

    expected = _expected_nested_logit_shares(raw_utilities, nest_spec)
    method_errors = {name: [] for name in method_names}
    pair_errors = {pair: [] for pair in pair_names}

    for n_draws in draw_counts:
        shares_by_method = {name: [] for name in method_names}

        for seed in seeds:
            shares_by_method["mc"].append(
                _nested_logit_mc_shares(
                    raw_utilities,
                    nest_spec,
                    n_draws=int(n_draws),
                    seed=seed,
                )
            )
            shares_by_method["exact_leaf"].append(
                _nested_logit_eet_shares(
                    raw_utilities,
                    nest_spec,
                    n_draws=int(n_draws),
                    seed=seed,
                )
            )

        for method_name, share_samples in shares_by_method.items():
            share_matrix = np.vstack([share.to_numpy() for share in share_samples])
            centered = share_matrix - expected.to_numpy()
            method_errors[method_name].append(_rmse(centered))

        for left_name, right_name in pair_names:
            left_matrix = np.vstack(
                [share.to_numpy() for share in shares_by_method[left_name]]
            )
            right_matrix = np.vstack(
                [share.to_numpy() for share in shares_by_method[right_name]]
            )
            pair_errors[(left_name, right_name)].append(
                _rmse(left_matrix - right_matrix)
            )

    for method_name, errors in method_errors.items():
        errors = np.asarray(errors, dtype=float)
        slope = _estimate_power_law_slope(draw_counts, errors)
        assert (
            slope_lower <= slope <= slope_upper
        ), f"{method_name} slope {slope:.3f} outside [{slope_lower}, {slope_upper}]"
        assert (
            errors[-1] < errors[0]
        ), f"{method_name} errors did not decrease: {errors}"
        if max_final_method_error is not None:
            assert (
                errors[-1] <= max_final_method_error
            ), f"{method_name} final error {errors[-1]:.6f} exceeds {max_final_method_error:.6f}"

    for left_name, right_name in pair_names:
        errors = np.asarray(pair_errors[(left_name, right_name)], dtype=float)
        slope = _estimate_power_law_slope(draw_counts, errors)
        assert (
            pair_slope_lower <= slope <= pair_slope_upper
        ), f"{left_name} vs {right_name} slope {slope:.3f} outside [{pair_slope_lower}, {pair_slope_upper}]"
        assert (
            errors[-1] < errors[0]
        ), f"{left_name} vs {right_name} errors did not decrease: {errors}"
        if max_final_pair_error is not None:
            assert (
                errors[-1] <= max_final_pair_error
            ), f"{left_name} vs {right_name} final error {errors[-1]:.6f} exceeds {max_final_pair_error:.6f}"


NESTED_LOGIT_EXACT_PARITY_CASES = [
    pytest.param(
        {
            "name": "root",
            "coefficient": 1.0,
            "alternatives": [
                {
                    "name": "AUTO",
                    "coefficient": 0.72,
                    "alternatives": ["DA_FREE", "DA_PAY"],
                },
                {"name": "NONMOTORIZED", "coefficient": 0.80, "alternatives": ["WALK"]},
            ],
        },
        pd.Series({"DA_FREE": 1.2, "DA_PAY": 0.7, "WALK": 0.1}),
        np.array(["DA_FREE", "DA_PAY", "WALK"]),
        id="two_level_single_leaf_nest",
    ),
    pytest.param(
        {
            "name": "root",
            "coefficient": 1.0,
            "alternatives": [
                {
                    "name": "AUTO",
                    "coefficient": 0.72,
                    "alternatives": [
                        {
                            "name": "DRIVEALONE",
                            "coefficient": 0.35,
                            "alternatives": ["DA_FREE", "DA_PAY"],
                        }
                    ],
                },
                {
                    "name": "TRANSIT",
                    "coefficient": 0.72,
                    "alternatives": [
                        {
                            "name": "WALKACCESS",
                            "coefficient": 0.50,
                            "alternatives": ["WALK_LOC", "WALK_EXP"],
                        }
                    ],
                },
                {
                    "name": "NONMOTORIZED",
                    "coefficient": 0.72,
                    "alternatives": ["WALK"],
                },
            ],
        },
        pd.Series(
            {
                "DA_FREE": 1.4,
                "DA_PAY": 0.9,
                "WALK_LOC": 0.5,
                "WALK_EXP": 0.2,
                "WALK": 0.0,
            }
        ),
        np.array(["DA_FREE", "DA_PAY", "WALK_LOC", "WALK_EXP", "WALK"]),
        id="three_level_single_leaf_chains",
    ),
    pytest.param(
        {
            "name": "root",
            "coefficient": 1.0,
            "alternatives": [
                {
                    "name": "MOTORIZED",
                    "coefficient": 0.78,
                    "alternatives": [
                        {
                            "name": "AUTO",
                            "coefficient": 0.62,
                            "alternatives": ["DA_FREE", "DA_PAY"],
                        },
                        {
                            "name": "RIDEHAIL",
                            "coefficient": 0.58,
                            "alternatives": ["RH_SHARED", "RH_SOLO"],
                        },
                    ],
                },
                {
                    "name": "ACTIVE",
                    "coefficient": 0.85,
                    "alternatives": ["BIKE", "WALK"],
                },
            ],
        },
        pd.Series(
            {
                "DA_FREE": 1.1,
                "DA_PAY": 0.8,
                "RH_SHARED": 0.7,
                "RH_SOLO": 0.9,
                "BIKE": 0.2,
                "WALK": 0.0,
            }
        ),
        np.array(["DA_FREE", "DA_PAY", "RH_SHARED", "RH_SOLO", "BIKE", "WALK"]),
        id="three_level_balanced",
    ),
    pytest.param(
        {
            "name": "root",
            "coefficient": 1.0,
            "alternatives": [
                {
                    "name": "AUTO",
                    "coefficient": 0.72,
                    "alternatives": [
                        {
                            "name": "DRIVE",
                            "coefficient": 0.60,
                            "alternatives": [
                                {
                                    "name": "SOLO",
                                    "coefficient": 0.45,
                                    "alternatives": ["DA_FREE", "DA_PAY"],
                                }
                            ],
                        }
                    ],
                },
                {
                    "name": "TRANSIT",
                    "coefficient": 0.75,
                    "alternatives": [
                        {
                            "name": "ACCESS",
                            "coefficient": 0.55,
                            "alternatives": [
                                {
                                    "name": "LOCAL",
                                    "coefficient": 0.50,
                                    "alternatives": ["WALK_LOC", "WALK_EXP"],
                                }
                            ],
                        }
                    ],
                },
                {"name": "ACTIVE", "coefficient": 0.82, "alternatives": ["WALK"]},
            ],
        },
        pd.Series(
            {
                "DA_FREE": 1.5,
                "DA_PAY": 1.0,
                "WALK_LOC": 0.7,
                "WALK_EXP": 0.4,
                "WALK": 0.1,
            }
        ),
        np.array(["DA_FREE", "DA_PAY", "WALK_LOC", "WALK_EXP", "WALK"]),
        id="four_level_single_leaf_chains",
    ),
    pytest.param(
        {
            "name": "root",
            "coefficient": 1.0,
            "alternatives": [
                {
                    "name": "MOTORIZED",
                    "coefficient": 0.80,
                    "alternatives": [
                        {
                            "name": "AUTO",
                            "coefficient": 0.68,
                            "alternatives": [
                                {
                                    "name": "SOLO",
                                    "coefficient": 0.48,
                                    "alternatives": ["DA_FREE", "DA_PAY"],
                                },
                                {
                                    "name": "SHARED",
                                    "coefficient": 0.52,
                                    "alternatives": ["SR2", "SR3"],
                                },
                            ],
                        },
                        {
                            "name": "TRANSIT",
                            "coefficient": 0.72,
                            "alternatives": [
                                {
                                    "name": "WALKACCESS",
                                    "coefficient": 0.55,
                                    "alternatives": ["WALK_LOC", "WALK_EXP"],
                                }
                            ],
                        },
                    ],
                },
                {"name": "ACTIVE", "coefficient": 0.88, "alternatives": ["BIKE"]},
            ],
        },
        pd.Series(
            {
                "DA_FREE": 1.4,
                "DA_PAY": 1.0,
                "SR2": 0.8,
                "SR3": 0.6,
                "WALK_LOC": 0.7,
                "WALK_EXP": 0.3,
                "BIKE": 0.1,
            }
        ),
        np.array(["DA_FREE", "DA_PAY", "SR2", "SR3", "WALK_LOC", "WALK_EXP", "BIKE"]),
        id="four_level_mixed_structure",
    ),
]


REALISTIC_NESTED_LOGIT_FAST_CASES = [
    {
        "id": "mtc_extended_tour_mode_choice_style",
        "nest_spec": {
            "name": "root",
            "coefficient": 1.0,
            "alternatives": [
                {
                    "name": "AUTO",
                    "coefficient": 0.72,
                    "alternatives": [
                        {
                            "name": "DRIVEALONE",
                            "coefficient": 0.35,
                            "alternatives": ["DRIVEALONEFREE", "DRIVEALONEPAY"],
                        },
                        {
                            "name": "SHAREDRIDE2",
                            "coefficient": 0.35,
                            "alternatives": ["SHARED2FREE", "SHARED2PAY"],
                        },
                        {
                            "name": "SHAREDRIDE3",
                            "coefficient": 0.40,
                            "alternatives": ["SHARED3FREE", "SHARED3PAY"],
                        },
                    ],
                },
                {
                    "name": "NONMOTORIZED",
                    "coefficient": 0.80,
                    "alternatives": ["WALK", "BIKE"],
                },
                {
                    "name": "TRANSIT",
                    "coefficient": 0.60,
                    "alternatives": [
                        {
                            "name": "WALKACCESS",
                            "coefficient": 0.50,
                            "alternatives": [
                                "WALK_LOC",
                                "WALK_LRF",
                                "WALK_EXP",
                                "WALK_HVY",
                                "WALK_COM",
                            ],
                        },
                        {
                            "name": "DRIVEACCESS",
                            "coefficient": 0.45,
                            "alternatives": [
                                "DRIVE_LOC",
                                "DRIVE_LRF",
                                "DRIVE_EXP",
                                "DRIVE_HVY",
                                "DRIVE_COM",
                            ],
                        },
                    ],
                },
                {
                    "name": "RIDEHAIL",
                    "coefficient": 0.65,
                    "alternatives": ["TAXI", "TNC_SINGLE", "TNC_SHARED"],
                },
            ],
        },
        "raw_utilities": pd.Series(
            {
                "DRIVEALONEFREE": 1.60,
                "DRIVEALONEPAY": 1.10,
                "SHARED2FREE": 1.05,
                "SHARED2PAY": 0.82,
                "SHARED3FREE": 0.70,
                "SHARED3PAY": 0.48,
                "WALK": -0.20,
                "BIKE": 0.05,
                "WALK_LOC": 0.15,
                "WALK_LRF": 0.05,
                "WALK_EXP": 0.22,
                "WALK_HVY": 0.10,
                "WALK_COM": -0.03,
                "DRIVE_LOC": 0.42,
                "DRIVE_LRF": 0.34,
                "DRIVE_EXP": 0.54,
                "DRIVE_HVY": 0.38,
                "DRIVE_COM": 0.26,
                "TAXI": 0.30,
                "TNC_SINGLE": 0.45,
                "TNC_SHARED": 0.18,
            }
        ),
    },
    {
        "id": "semcog_tour_mode_choice_style",
        "nest_spec": {
            "name": "root",
            "coefficient": 1.0,
            "alternatives": [
                {
                    "name": "AUTO",
                    "coefficient": 0.78,
                    "alternatives": ["DRIVEALONE", "SHARED2", "SHARED3"],
                },
                {
                    "name": "NONMOTORIZED",
                    "coefficient": 0.85,
                    "alternatives": ["WALK", "BIKE"],
                },
                {
                    "name": "TRANSIT",
                    "coefficient": 0.64,
                    "alternatives": [
                        {
                            "name": "WALKACCESS",
                            "coefficient": 0.56,
                            "alternatives": ["WALK_LOC", "WALK_PRM", "WALK_MIX"],
                        },
                        {
                            "name": "PNRACCESS",
                            "coefficient": 0.52,
                            "alternatives": ["PNR_LOC", "PNR_PRM", "PNR_MIX"],
                        },
                        {
                            "name": "KNRACCESS",
                            "coefficient": 0.50,
                            "alternatives": ["KNR_LOC", "KNR_PRM", "KNR_MIX"],
                        },
                    ],
                },
                {
                    "name": "SCHOOL_BUS",
                    "coefficient": 0.92,
                    "alternatives": ["SCHOOLBUS"],
                },
                {
                    "name": "RIDEHAIL",
                    "coefficient": 0.68,
                    "alternatives": ["TAXI", "TNC_SINGLE", "TNC_SHARED"],
                },
            ],
        },
        "raw_utilities": pd.Series(
            {
                "DRIVEALONE": 1.45,
                "SHARED2": 1.08,
                "SHARED3": 0.76,
                "WALK": -0.10,
                "BIKE": 0.12,
                "WALK_LOC": 0.10,
                "WALK_PRM": 0.18,
                "WALK_MIX": 0.06,
                "PNR_LOC": 0.30,
                "PNR_PRM": 0.36,
                "PNR_MIX": 0.26,
                "KNR_LOC": 0.27,
                "KNR_PRM": 0.32,
                "KNR_MIX": 0.21,
                "SCHOOLBUS": 0.24,
                "TAXI": 0.22,
                "TNC_SINGLE": 0.40,
                "TNC_SHARED": 0.16,
            }
        ),
    },
]


@pytest.mark.parametrize(
    "nest_spec,raw_utilities,_alt_order_array",
    NESTED_LOGIT_EXACT_PARITY_CASES,
)
def test_make_choices_vs_eet_nl_exact_leaf_parity_across_structures(
    nest_spec, raw_utilities, _alt_order_array
):
    n_draws = 100_000
    expected = _expected_nested_logit_shares(raw_utilities, nest_spec)
    observed = _nested_logit_eet_shares(
        raw_utilities,
        nest_spec,
        n_draws=n_draws,
    )

    _assert_empirical_shares_close(observed, expected, n_draws=n_draws)


# def test_exact_leaf_error_terms_use_float64_with_float32_nested_utilities():
#     nest_spec = {
#         "name": "root",
#         "coefficient": 1.0,
#         "alternatives": [
#             {"name": "motorized", "coefficient": 0.5, "alternatives": ["car", "bus"]},
#             "walk",
#         ],
#     }
#     raw_utilities = pd.DataFrame(
#         np.array([[0.5, 0.2, 0.4]], dtype=np.float32),
#         index=pd.RangeIndex(1, name="chooser_id"),
#         columns=["car", "bus", "walk"],
#     )
#     # nested_utilities = simulate.compute_nested_utilities(
#     #     raw_utilities, nest_spec
#     # ).astype(np.float32)
#     # alt_order_array = np.array(["car", "bus", "walk"])
#     state = _make_rng_state(
#         raw_utilities,
#         seed=17,
#         step_name="exact_leaf_float64_dtype",
#     )

#     try:
#         error_terms = logit.sample_nested_logit_exact_leaf_error_terms(
#             state,
#             raw_utilities,
#             nest_spec,
#         )
#     finally:
#         _finish_rng_state(state, "exact_leaf_float64_dtype")

#     assert all(dtype == np.float64 for dtype in error_terms.dtypes)

def test_make_choices_utility_based_routes_nested_logit_to_nl_eet(monkeypatch):
    sentinel = pd.Series([1, 0], index=pd.Index([100, 101], name="chooser_id"))

    def fake_make_choices_explicit_error_term_nl(
        state,
        alt_utilities,
        nest_spec,
        trace_label,
        trace_choosers=None,
        alts_context=None,
        alt_nrs_df=None,
    ):
        assert list(alt_utilities.columns) == ["car", "walk"]
        assert trace_label == "test.make_choices_utility_based"
        assert trace_choosers is None
        assert alts_context is None
        assert alt_nrs_df is None
        return sentinel

    monkeypatch.setattr(
        logit,
        "make_choices_explicit_error_term_nl",
        fake_make_choices_explicit_error_term_nl,
    )

    state = workflow.State().default_settings()
    utilities = pd.DataFrame(
        [[0.0, 0.0], [0.0, 0.0]],
        index=pd.Index([100, 101], name="chooser_id"),
        columns=["car", "walk"],
    )
    nest_spec = {
        "name": "root",
        "coefficient": 1.0,
        "alternatives": [
            {"name": "motorized", "coefficient": 0.7, "alternatives": ["car"]},
            "walk",
        ],
    }

    choices, rands = logit.make_choices_utility_based(
        state,
        utilities,
        nest_spec=nest_spec,
        trace_label="test",
    )

    pdt.assert_series_equal(choices, sentinel)
    pdt.assert_series_equal(
        rands,
        pd.Series([0, 0], index=pd.Index([100, 101], name="chooser_id")),
    )


@pytest.mark.parametrize(
    "case",
    REALISTIC_NESTED_LOGIT_FAST_CASES,
    ids=[case["id"] for case in REALISTIC_NESTED_LOGIT_FAST_CASES],
)
def test_nested_logit_methods_match_expected_shares_for_realistic_tour_mode_choice_nests(
    case,
):
    _assert_nested_logit_methods_match_expected_across_seeds(
        case["raw_utilities"],
        case["nest_spec"],
        n_draws=6_000,
        seeds=[11, 23, 37],
    )


def test_nested_logit_share_response_tracks_utility_perturbations():
    case = REALISTIC_NESTED_LOGIT_FAST_CASES[0]
    base_utilities = case["raw_utilities"]
    perturbed_utilities = base_utilities.copy()
    perturbed_utilities["DRIVE_EXP"] += 0.60
    perturbed_utilities["TNC_SHARED"] -= 0.45

    baseline_expected = _expected_nested_logit_shares(base_utilities, case["nest_spec"])
    perturbed_expected = _expected_nested_logit_shares(
        perturbed_utilities, case["nest_spec"]
    )

    expected_delta = perturbed_expected - baseline_expected
    assert expected_delta["DRIVE_EXP"] > 0
    assert expected_delta["TNC_SHARED"] < 0

    for method in ("mc", "exact_leaf"):
        baseline_matrix = _nested_logit_method_share_matrix(
            base_utilities,
            case["nest_spec"],
            method=method,
            n_draws=8_000,
            seeds=[11, 23, 37],
        )
        perturbed_matrix = _nested_logit_method_share_matrix(
            perturbed_utilities,
            case["nest_spec"],
            method=method,
            n_draws=8_000,
            seeds=[11, 23, 37],
        )
        _assert_average_empirical_shares_close(
            baseline_matrix,
            baseline_expected,
            n_draws=8_000,
        )
        _assert_average_empirical_shares_close(
            perturbed_matrix,
            perturbed_expected,
            n_draws=8_000,
        )
        _assert_average_share_deltas_close(
            baseline_matrix,
            perturbed_matrix,
            baseline_expected,
            perturbed_expected,
            n_draws=8_000,
        )


def test_three_level_nested_logit_methods_follow_monte_carlo_power_law():
    _assert_three_level_nested_logit_methods_follow_power_law(
        draw_counts=np.array([2_000, 8_000, 32_000]),
        seeds=[17, 29, 43],
    )


# # @pytest.mark.slow
# def test_three_level_nested_logit_methods_follow_monte_carlo_power_law_large_draws():
#     _assert_three_level_nested_logit_methods_follow_power_law(
#         draw_counts=np.array([8_000, 32_000, 128_000]),
#         seeds=[17, 29, 43],
#         slope_lower=-0.7,
#         slope_upper=-0.3,
#         pair_slope_lower=-1.0,
#         pair_slope_upper=-0.2,
#         max_final_method_error=0.0015,
#         max_final_pair_error=0.0020,
#     )


#
# Interaction Dataset Tests
#
def test_interaction_dataset_no_sample(interaction_choosers, interaction_alts):
    expected = pd.DataFrame(
        {
            "attr": ["a"] * 4 + ["b"] * 4 + ["c"] * 4 + ["b"] * 4,
            "prop": [10, 20, 30, 40] * 4,
        },
        index=[1, 2, 3, 4] * 4,
    )

    interacted = logit.interaction_dataset(
        workflow.State().default_settings(), interaction_choosers, interaction_alts
    )

    interacted, expected = interacted.align(expected, axis=1)
    pdt.assert_frame_equal(interacted, expected)


def test_interaction_dataset_sampled(interaction_choosers, interaction_alts):
    expected = pd.DataFrame(
        {
            "attr": ["a"] * 2 + ["b"] * 2 + ["c"] * 2 + ["b"] * 2,
            "prop": [30, 40, 10, 30, 40, 10, 20, 10],
        },
        index=[3, 4, 1, 3, 4, 1, 2, 1],
    )

    interacted = logit.interaction_dataset(
        workflow.State().default_settings(),
        interaction_choosers,
        interaction_alts,
        sample_size=2,
    )

    interacted, expected = interacted.align(expected, axis=1)
    pdt.assert_frame_equal(interacted, expected)


def reset_step(state, name="test_step"):
    state.get_rn_generator().end_step(name)
    state.get_rn_generator().begin_step(name)


def test_make_choices_utility_based_sampled_alts():
    """Test the situation of making choices from a sampled choice set"""
    # TODO should these tests go in test_random?
    state = workflow.State().default_settings()
    # Make explicit that there's two indexing schemes - the raw alts, and the 0 based internals
    utils_project_raw = pd.DataFrame(
        {"a": 10.582999, "b": 10.680792, "c": 10.710443},
        index=pd.Index([0], name="person_id"),
    )
    # zero based indexes
    utils_project = utils_project_raw.rename(columns={"a": 0, "b": 1, "c": 2})
    utils_base = utils_project_raw[["a", "c"]].rename(columns={"a": 0, "c": 1})

    assert utils_project.index.name == "person_id"
    state.get_rn_generator().add_channel("persons", utils_project)
    state.get_rn_generator().begin_step("test_step")
    # mock base case, where alt 1 is omitted (it was improved in the project)
    # this situation is quite common with poisson sampling with a variable choice set size,
    # but it can also happen in with-replacement EET sampling e.g. if alt 2 had a pick_count of 2 in the base case.
    # In principle, it can also be problematic for non-sampled choices where there is a base project difference in the
    # availability of alternatives .e.g a new mode was introduced in the project case

    utils_project_with_rands = add_ev1_random(state, utils_project)
    rands_project = utils_project_with_rands - utils_project
    reset_step(state)
    utils_base_with_rands = add_ev1_random(state, utils_base)
    rands_base = utils_base_with_rands - utils_base
    rands_base_labeled = rands_base.rename(columns={0: "a", 1: "c"})
    rands_project_labeled = rands_project.rename(columns={0: "a", 1: "b", 2: "c"})
    with pytest.raises(
        AssertionError, match=re.escape('(column name="c") are different')
    ):
        # TODO this should pass
        pdt.assert_frame_equal(
            rands_base_labeled, rands_project_labeled.loc[:, rands_base_labeled.columns]
        )
    # document incorrect invariant - first two columns have the same random numbers:
    pdt.assert_frame_equal(rands_base, rands_project.iloc[:, :2])

    # revised approach
    reset_step(state)
    alt_nrs_df = pd.DataFrame({0: 0, 1: 1, 2: 2}, index=utils_project_raw.index)
    alt_info = AltsContext.from_num_alts(3, zero_based=True)
    utils_project_with_rands = add_ev1_random(
        state, utils_project, alt_info=alt_info, alt_nrs_df=alt_nrs_df
    )
    rands_project = utils_project_with_rands - utils_project
    reset_step(state)

    # alt "b" is missing from the sampled choice set, alt_nrs_df is set to reflect that
    alt_nrs_df = pd.DataFrame({0: 0, 1: 2}, index=utils_project_raw.index)
    utils_base_with_rands = add_ev1_random(
        state, utils_base, alt_info=alt_info, alt_nrs_df=alt_nrs_df
    )
    rands_base = utils_base_with_rands - utils_base
    rands_base_labeled = rands_base.rename(columns={0: "a", 1: "c"})
    rands_project_labeled = rands_project.rename(columns={0: "a", 1: "b", 2: "c"})

    # Corrected invariant holds true
    pdt.assert_frame_equal(
        rands_base_labeled, rands_project_labeled.loc[:, rands_base_labeled.columns]
    )


def test_alts_context_from_series_and_properties():
    ctx = AltsContext.from_series(pd.Index([3, 5, 9, 4]))

    assert ctx.min_alt_id == 3
    assert ctx.max_alt_id == 9
    assert ctx.n_alts_to_cover_max_id == 10
    assert ctx.n_rands_to_sample == 10


@pytest.mark.parametrize(
    "num_alts,zero_based,expected_min,expected_max,expected_n_cover",
    [
        (5, True, 0, 4, 5),
        (5, False, 1, 5, 6),
    ],
)
def test_alts_context_from_num_alts(
    num_alts, zero_based, expected_min, expected_max, expected_n_cover
):
    ctx = AltsContext.from_num_alts(num_alts=num_alts, zero_based=zero_based)

    assert ctx.min_alt_id == expected_min
    assert ctx.max_alt_id == expected_max
    assert ctx.n_alts_to_cover_max_id == expected_n_cover
    assert ctx.n_rands_to_sample == expected_n_cover
