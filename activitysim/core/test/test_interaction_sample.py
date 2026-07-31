# ActivitySim
# See full license in LICENSE.txt.

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from activitysim.core import interaction_sample, workflow
from activitysim.core.configuration.base import ComputeSettings


@pytest.fixture
def state() -> workflow.State:
    state = workflow.State().default_settings()
    state.settings.check_for_variability = False
    return state


def test_interaction_sample_ignores_stable_positions_without_global_eet(
    state, monkeypatch
):
    # Do not support stable alt positions or tracking total alts when running with MC sampling
    # to not introduce any additional changes while adding eet simulation support to ensure no
    # regressions. We can add these features later if desired.
    captured = {}

    def fake_interaction_sample(_state, _choosers, _alternatives, **kwargs):
        captured["stable_alt_positions"] = kwargs["stable_alt_positions"]
        captured["n_total_alts"] = kwargs["n_total_alts"]
        return pd.DataFrame(
            {"alt_id": [10, 11], "prob": [1.0, 1.0], "pick_count": [1, 1]},
            index=pd.Index([1, 2], name="person_id"),
        )

    monkeypatch.setattr(
        interaction_sample, "_interaction_sample", fake_interaction_sample
    )

    state.settings.use_explicit_error_terms = False
    choosers = pd.DataFrame(index=pd.Index([1, 2], name="person_id"))
    alternatives = pd.DataFrame(index=pd.Index([10, 11, 12], name="alt_id"))
    spec = pd.DataFrame(
        {"coefficient": [1.0]},
        index=pd.Index(["1"], name="Expression"),
    )

    interaction_sample.interaction_sample(
        state,
        choosers,
        alternatives,
        spec,
        sample_size=1,
        alt_col_name="alt_id",
        stable_alt_positions=np.array([0, 2], dtype=np.int64),
        n_total_alts=3,
    )

    assert captured["stable_alt_positions"] is None
    assert captured["n_total_alts"] is None


def test_interaction_sample_preserves_stable_positions_with_global_eet(
    state, monkeypatch
):
    captured = {}

    def fake_interaction_sample(_state, _choosers, _alternatives, **kwargs):
        captured["stable_alt_positions"] = kwargs["stable_alt_positions"]
        captured["n_total_alts"] = kwargs["n_total_alts"]
        return pd.DataFrame(
            {"alt_id": [10, 11], "prob": [1.0, 1.0], "pick_count": [1, 1]},
            index=pd.Index([1, 2], name="person_id"),
        )

    monkeypatch.setattr(
        interaction_sample, "_interaction_sample", fake_interaction_sample
    )

    state.settings.use_explicit_error_terms = True
    choosers = pd.DataFrame(index=pd.Index([1, 2], name="person_id"))
    alternatives = pd.DataFrame(index=pd.Index([10, 11, 12], name="alt_id"))
    spec = pd.DataFrame(
        {"coefficient": [1.0]},
        index=pd.Index(["1"], name="Expression"),
    )
    stable_alt_positions = np.array([0, 2], dtype=np.int64)

    interaction_sample.interaction_sample(
        state,
        choosers,
        alternatives,
        spec,
        sample_size=1,
        alt_col_name="alt_id",
        stable_alt_positions=stable_alt_positions,
        n_total_alts=3,
        compute_settings=ComputeSettings(sample_method="eet"),
    )

    np.testing.assert_array_equal(
        captured["stable_alt_positions"],
        stable_alt_positions,
    )
    assert captured["n_total_alts"] == 3


def _weighted_shares(df: pd.DataFrame) -> pd.Series:
    counts = df.groupby("alt_id")["pick_count"].sum()
    return (counts / counts.sum()).sort_index()


def test_interaction_sample_parity(state):
    # Run all three sampling methods on a realistic synthetic case and check
    # that their aggregate sampled shares stay close.

    num_choosers = 100_000
    num_alts = 100
    sample_size = 10

    # Create random choosers and alternatives
    rng = np.random.default_rng(42)
    choosers = pd.DataFrame(
        {"chooser_attr": rng.random(num_choosers)},
        index=pd.Index(range(num_choosers), name="person_id"),
    )

    alternatives = pd.DataFrame(
        {"alt_attr": rng.random(num_alts)},
        index=pd.Index(range(num_alts), name="alt_id"),
    )

    # Simple spec: utility = chooser_attr * alt_attr
    spec = pd.DataFrame(
        {"coefficient": [1.0]},
        index=pd.Index(["chooser_attr * alt_attr"], name="Expression"),
    )

    # Run Monte Carlo with replacement.
    state.settings.use_explicit_error_terms = False
    state.rng().set_base_seed(42)
    state.rng().add_channel("person_id", choosers)
    state.rng().begin_step("test_step_mnl")

    choices_mnl = interaction_sample.interaction_sample(
        state,
        choosers,
        alternatives,
        spec,
        sample_size=sample_size,
        alt_col_name="alt_id",
    )

    # Run Poisson inclusion sampling, which is the default when global EET is enabled.
    state.init_state()  # reset the state to rerun with same seed
    state.settings.use_explicit_error_terms = True
    state.rng().set_base_seed(42)
    state.rng().add_channel("person_id", choosers)
    state.rng().begin_step("test_step_poisson")

    choices_poisson = interaction_sample.interaction_sample(
        state,
        choosers,
        alternatives,
        spec,
        sample_size=sample_size,
        alt_col_name="alt_id",
    )

    # Run EET-with-replacement with the same global EET setting.
    state.init_state()
    state.settings.use_explicit_error_terms = True
    state.rng().set_base_seed(42)
    state.rng().add_channel("person_id", choosers)
    state.rng().begin_step("test_step_eet")

    choices_eet = interaction_sample.interaction_sample(
        state,
        choosers,
        alternatives,
        spec,
        sample_size=sample_size,
        alt_col_name="alt_id",
        compute_settings=ComputeSettings(sample_method="eet"),
    )

    assert "alt_id" in choices_mnl.columns
    assert "alt_id" in choices_poisson.columns
    assert "alt_id" in choices_eet.columns
    assert not choices_mnl["alt_id"].isna().any()
    assert not choices_poisson["alt_id"].isna().any()
    assert not choices_eet["alt_id"].isna().any()
    assert choices_mnl["alt_id"].isin(alternatives.index).all()
    assert choices_poisson["alt_id"].isin(alternatives.index).all()
    assert choices_eet["alt_id"].isin(alternatives.index).all()

    shares = {
        "monte_carlo": _weighted_shares(choices_mnl),
        "poisson": _weighted_shares(choices_poisson),
        "eet": _weighted_shares(choices_eet),
    }

    for left, right in [
        ("monte_carlo", "poisson"),
        ("monte_carlo", "eet"),
        ("poisson", "eet"),
    ]:
        all_alts = set(shares[left].index) | set(shares[right].index)
        for alt in all_alts:
            diff = abs(shares[left].get(alt, 0.0) - shares[right].get(alt, 0.0))
            assert diff < 0.01, (
                f"Large discrepancy at alt {alt} between {left} and {right}: "
                f"{left}={shares[left].get(alt, 0.0):.4f}, "
                f"{right}={shares[right].get(alt, 0.0):.4f}, diff={diff:.4f}"
            )


def test_interaction_sample_eet_unavailable_alternatives(state):
    # Test that EET handles unavailable alternatives in sampling
    num_choosers = 100
    num_alts = 10
    sample_size = 2
    choosers = pd.DataFrame(
        {"chooser_attr": np.ones(num_choosers)},
        index=pd.Index(range(num_choosers), name="person_id"),
    )

    # Alt 0-4 are attractive, Alt 5-9 are "unavailable"
    alternatives = pd.DataFrame(
        {"alt_attr": [10.0] * 5 + [-1000.0] * 5},
        index=pd.Index(range(num_alts), name="alt_id"),
    )

    spec = pd.DataFrame(
        {"coefficient": [1.0]},
        index=pd.Index(["alt_attr"], name="Expression"),
    )

    # Run with EET
    state.settings.use_explicit_error_terms = True
    state.rng().set_base_seed(42)
    state.rng().add_channel("person_id", choosers)
    state.rng().begin_step("test_unavailable_eet")

    choices_eet = interaction_sample.interaction_sample(
        state,
        choosers,
        alternatives,
        spec,
        sample_size=sample_size,
        alt_col_name="alt_id",
    )

    # Sampled alternatives should only be from Alt 0-4
    assert choices_eet["alt_id"].isin([0, 1, 2, 3, 4]).all()
    assert not choices_eet["alt_id"].isin([5, 6, 7, 8, 9]).any()


def test_interaction_sample_parity_peaked_utilities_eet_with_replacement(state):
    # Under highly peaked utilities, the EET-with-replacement sampler should still
    # approximate repeated-draw MNL shares because both sample with replacement.
    # This test also documents that per-model compute settings can override the
    # global default: global EET implies Poisson by default, but this model opts
    # into EET-with-replacement explicitly.
    num_choosers = 20_000
    num_alts = 100
    sample_size = 5

    choosers = pd.DataFrame(
        {"chooser_attr": np.ones(num_choosers)},
        index=pd.Index(range(num_choosers), name="person_id"),
    )

    alt_utils = np.array([10.0, 1.0] + [0.0] * (num_alts - 2), dtype=np.float64)
    alternatives = pd.DataFrame(
        {"alt_attr": alt_utils},
        index=pd.Index(range(num_alts), name="alt_id"),
    )

    spec = pd.DataFrame(
        {"coefficient": [1.0]},
        index=pd.Index(["alt_attr"], name="Expression"),
    )

    # Run non-EET path.
    state.settings.use_explicit_error_terms = False
    state.rng().set_base_seed(42)
    state.rng().add_channel("person_id", choosers)
    state.rng().begin_step("test_peaked_mnl")
    choices_mnl = interaction_sample.interaction_sample(
        state,
        choosers,
        alternatives,
        spec,
        sample_size=sample_size,
        alt_col_name="alt_id",
    )

    # Run EET-with-replacement path with the same seed.
    state.init_state()
    state.settings.use_explicit_error_terms = True
    state.rng().set_base_seed(42)
    state.rng().add_channel("person_id", choosers)
    state.rng().begin_step("test_peaked_explicit")
    choices_explicit = interaction_sample.interaction_sample(
        state,
        choosers,
        alternatives,
        spec,
        sample_size=sample_size,
        alt_col_name="alt_id",
        compute_settings=ComputeSettings(sample_method="eet"),
    )

    mnl_shares = _weighted_shares(choices_mnl)
    explicit_shares = _weighted_shares(choices_explicit)

    all_alts = set(mnl_shares.index) | set(explicit_shares.index)
    for alt in all_alts:
        diff = abs(mnl_shares.get(alt, 0.0) - explicit_shares.get(alt, 0.0))
        assert diff < 0.005, (
            f"Peaked utility parity mismatch at alt {alt}: "
            f"mnl={mnl_shares.get(alt, 0.0):.6f}, "
            f"explicit={explicit_shares.get(alt, 0.0):.6f}, diff={diff:.6f}"
        )

    # The dominant alternative should absorb almost all mass in both paths.
    assert mnl_shares.get(0, 0.0) > 0.99
    assert explicit_shares.get(0, 0.0) > 0.99


def test_interaction_sample_peaked_utilities_poisson_matches_inclusion_shares(state):
    # Poisson sampling does not reproduce repeated-draw MNL shares in peaked cases.
    # It samples each alternative independently with inclusion probability
    # 1 - (1 - p)^sample_size, so the dominant alternative's share is flattened
    # relative to MNL once the included set is normalized. This is also the
    # default interaction_sample behavior when global EET is enabled.
    num_choosers = 20_000
    num_alts = 100
    sample_size = 5

    choosers = pd.DataFrame(
        {"chooser_attr": np.ones(num_choosers)},
        index=pd.Index(range(num_choosers), name="person_id"),
    )
    alt_utils = np.array([10.0, 1.0] + [0.0] * (num_alts - 2), dtype=np.float64)
    alternatives = pd.DataFrame(
        {"alt_attr": alt_utils},
        index=pd.Index(range(num_alts), name="alt_id"),
    )
    spec = pd.DataFrame(
        {"coefficient": [1.0]},
        index=pd.Index(["alt_attr"], name="Expression"),
    )

    state.settings.use_explicit_error_terms = False
    state.rng().set_base_seed(42)
    state.rng().add_channel("person_id", choosers)
    state.rng().begin_step("test_peaked_mnl_poisson_compare")
    choices_mnl = interaction_sample.interaction_sample(
        state,
        choosers,
        alternatives,
        spec,
        sample_size=sample_size,
        alt_col_name="alt_id",
    )

    state.init_state()
    state.settings.use_explicit_error_terms = True
    state.rng().set_base_seed(42)
    state.rng().add_channel("person_id", choosers)
    state.rng().begin_step("test_peaked_poisson")
    choices_poisson = interaction_sample.interaction_sample(
        state,
        choosers,
        alternatives,
        spec,
        sample_size=sample_size,
        alt_col_name="alt_id",
    )

    mnl_shares = _weighted_shares(choices_mnl)
    poisson_shares = _weighted_shares(choices_poisson)

    weights = np.exp(alt_utils)
    probs = weights / weights.sum()
    expected_poisson_shares = 1 - np.power(1 - probs, sample_size)
    expected_poisson_shares /= expected_poisson_shares.sum()

    assert mnl_shares.get(0, 0.0) > poisson_shares.get(0, 0.0) + 0.01
    assert abs(poisson_shares.get(0, 0.0) - expected_poisson_shares[0]) < 0.005
    assert abs(poisson_shares.get(1, 0.0) - expected_poisson_shares[1]) < 0.002


def _shares_for_sample(
    state,
    choosers,
    alternatives,
    spec,
    sample_size,
    *,
    use_eet,
    sample_method,
    seed,
    step_name,
):
    state.init_state()
    state.settings.use_explicit_error_terms = use_eet
    state.rng().set_base_seed(seed)
    state.rng().add_channel("person_id", choosers)
    state.rng().begin_step(step_name)
    compute_settings = (
        ComputeSettings(sample_method=sample_method) if sample_method else None
    )
    choices = interaction_sample.interaction_sample(
        state,
        choosers,
        alternatives,
        spec,
        sample_size=sample_size,
        alt_col_name="alt_id",
        compute_settings=compute_settings,
    )
    return choices, _weighted_shares(choices)


def test_interaction_sample_eet_sampling_under_mc_simulation(state):
    # use_eet=False + sample_method="eet" was silently ignored before the
    # sampling/simulation decoupling. The dispatch now keys on sampling_method
    # only, so this combo must produce shares that match use_eet=True + eet.
    num_choosers = 100_000
    num_alts = 100
    sample_size = 10

    rng = np.random.default_rng(42)
    choosers = pd.DataFrame(
        {"chooser_attr": rng.random(num_choosers)},
        index=pd.Index(range(num_choosers), name="person_id"),
    )
    alternatives = pd.DataFrame(
        {"alt_attr": rng.random(num_alts)},
        index=pd.Index(range(num_alts), name="alt_id"),
    )
    spec = pd.DataFrame(
        {"coefficient": [1.0]},
        index=pd.Index(["chooser_attr * alt_attr"], name="Expression"),
    )

    _, shares_mc_sim = _shares_for_sample(
        state,
        choosers,
        alternatives,
        spec,
        sample_size,
        use_eet=False,
        sample_method="eet",
        seed=42,
        step_name="test_eet_under_mc_sim",
    )
    _, shares_eet_sim = _shares_for_sample(
        state,
        choosers,
        alternatives,
        spec,
        sample_size,
        use_eet=True,
        sample_method="eet",
        seed=42,
        step_name="test_eet_under_eet_sim",
    )

    all_alts = set(shares_mc_sim.index) | set(shares_eet_sim.index)
    for alt in all_alts:
        diff = abs(shares_mc_sim.get(alt, 0.0) - shares_eet_sim.get(alt, 0.0))
        assert diff < 0.01, (
            f"EET sampling shares should not depend on simulation mode at alt {alt}: "
            f"mc_sim={shares_mc_sim.get(alt, 0.0):.4f}, "
            f"eet_sim={shares_eet_sim.get(alt, 0.0):.4f}, diff={diff:.4f}"
        )


def test_interaction_sample_poisson_sampling_under_mc_simulation(state):
    # use_eet=False + sample_method="poisson" used to silently fall through to MC
    # sampling and then have pick_count forced to 1, corrupting results. After
    # decoupling, the combo must run the Poisson path and match use_eet=True + poisson.
    num_choosers = 100_000
    num_alts = 100
    sample_size = 10

    rng = np.random.default_rng(42)
    choosers = pd.DataFrame(
        {"chooser_attr": rng.random(num_choosers)},
        index=pd.Index(range(num_choosers), name="person_id"),
    )
    alternatives = pd.DataFrame(
        {"alt_attr": rng.random(num_alts)},
        index=pd.Index(range(num_alts), name="alt_id"),
    )
    spec = pd.DataFrame(
        {"coefficient": [1.0]},
        index=pd.Index(["chooser_attr * alt_attr"], name="Expression"),
    )

    choices_mc_sim, shares_mc_sim = _shares_for_sample(
        state,
        choosers,
        alternatives,
        spec,
        sample_size,
        use_eet=False,
        sample_method="poisson",
        seed=42,
        step_name="test_poisson_under_mc_sim",
    )
    _, shares_eet_sim = _shares_for_sample(
        state,
        choosers,
        alternatives,
        spec,
        sample_size,
        use_eet=True,
        sample_method="poisson",
        seed=42,
        step_name="test_poisson_under_eet_sim",
    )

    # Poisson contract: pick_count must be uniformly 1
    assert (choices_mc_sim["pick_count"] == 1).all(), (
        "Poisson sampling under MC simulation must produce pick_count=1; got "
        f"{choices_mc_sim['pick_count'].value_counts().to_dict()}"
    )

    all_alts = set(shares_mc_sim.index) | set(shares_eet_sim.index)
    for alt in all_alts:
        diff = abs(shares_mc_sim.get(alt, 0.0) - shares_eet_sim.get(alt, 0.0))
        assert diff < 0.01, (
            f"Poisson sampling shares should not depend on simulation mode at alt {alt}: "
            f"mc_sim={shares_mc_sim.get(alt, 0.0):.4f}, "
            f"eet_sim={shares_eet_sim.get(alt, 0.0):.4f}, diff={diff:.4f}"
        )


def test_interaction_sample_mc_sampling_under_eet_simulation(state):
    num_choosers = 100_000
    num_alts = 100
    sample_size = 10

    rng = np.random.default_rng(42)
    choosers = pd.DataFrame(
        {"chooser_attr": rng.random(num_choosers)},
        index=pd.Index(range(num_choosers), name="person_id"),
    )
    alternatives = pd.DataFrame(
        {"alt_attr": rng.random(num_alts)},
        index=pd.Index(range(num_alts), name="alt_id"),
    )
    spec = pd.DataFrame(
        {"coefficient": [1.0]},
        index=pd.Index(["chooser_attr * alt_attr"], name="Expression"),
    )

    _, shares_mc_sim = _shares_for_sample(
        state,
        choosers,
        alternatives,
        spec,
        sample_size,
        use_eet=False,
        sample_method="monte_carlo",
        seed=42,
        step_name="test_mc_under_mc_sim",
    )
    _, shares_eet_sim = _shares_for_sample(
        state,
        choosers,
        alternatives,
        spec,
        sample_size,
        use_eet=True,
        sample_method="monte_carlo",
        seed=42,
        step_name="test_mc_under_eet_sim",
    )

    all_alts = set(shares_mc_sim.index) | set(shares_eet_sim.index)
    for alt in all_alts:
        diff = abs(shares_mc_sim.get(alt, 0.0) - shares_eet_sim.get(alt, 0.0))
        assert diff < 0.01, (
            f"MC sampling shares should not depend on simulation mode at alt {alt}: "
            f"mc_sim={shares_mc_sim.get(alt, 0.0):.4f}, "
            f"eet_sim={shares_eet_sim.get(alt, 0.0):.4f}, diff={diff:.4f}"
        )


class _DummyChunkSizer:
    def log_df(self, *_args, **_kwargs):
        return None


class _DummyState:
    def __init__(self, rng):
        self._rng = rng
        self.settings = SimpleNamespace(skip_failed_choices=False)

    def get_rn_generator(self):
        return self._rng


class _SequentialDummyRng:
    def __init__(self, draws):
        self._draws = list(draws)

    def random_for_df(self, df, n=1):
        draw = self._draws.pop(0)
        assert draw.shape == (len(df), n)
        return draw

    def random_for_df_stable_alt_positions(
        self, df, stable_alt_positions, n_total_alts
    ):
        draw = self._draws.pop(0)
        assert draw.shape == (len(df), n_total_alts)
        return draw[:, stable_alt_positions]


class _DummyRngUtilityBased:
    def __init__(self, rands_3d):
        self.rands_3d = rands_3d

    def gumbel_max_positions_for_df(
        self,
        utilities,
        sample_size,
        stable_alt_positions=None,
        n_total_alts=None,
    ):
        assert sample_size == self.rands_3d.shape[2]
        if stable_alt_positions is None:
            active_rands = self.rands_3d
        else:
            assert n_total_alts == self.rands_3d.shape[1]
            active_rands = self.rands_3d[:, stable_alt_positions, :]
        return np.argmax(
            active_rands + utilities.to_numpy()[:, :, np.newaxis],
            axis=1,
        )


def _reference_poisson_sampled_values(probs_np, draws, sample_size):
    """
    Independent re-derivation of the documented Poisson sampling result, used to check
    the implementation against the formula rather than against itself.

    An alternative ends up in the choice set if its Bernoulli draw succeeded, or if the
    chooser drew nothing at all and the alternative is one of the `sample_size` most
    likely *available* (p > 0) alternatives. Those events are disjoint, so the
    probability of an alternative being in the returned set is
    `q_i + P0 * 1{i in fallback set}` for every chooser and both branches.

    Returns the sparse chooser-by-alternative array of reported probabilities, with
    np.nan for alternatives that are not in the choice set.
    """
    inclusion_probs = 1.0 - np.power(1.0 - probs_np, sample_size)
    empty_sample_probs = np.prod(1.0 - inclusion_probs, axis=1)

    sampled = draws < inclusion_probs
    empty_rows = ~sampled.any(axis=1)

    in_fallback = np.zeros(probs_np.shape, dtype=bool)
    k = min(sample_size, probs_np.shape[1])
    top_k = np.argsort(-probs_np, axis=1, kind="stable")[:, :k]
    np.put_along_axis(in_fallback, top_k, True, axis=1)
    # unavailable alternatives never enter the choice set
    in_fallback &= probs_np > 0

    # the implementation skips the P0 term where it cannot matter; mirror that here so
    # the comparison stays exact (see POISSON_EMPTY_SAMPLE_TOLERANCE)
    material = empty_sample_probs > interaction_sample.POISSON_EMPTY_SAMPLE_TOLERANCE
    reported = inclusion_probs + empty_sample_probs[:, None] * (
        in_fallback & (material | empty_rows)[:, None]
    )

    sampled = sampled | (empty_rows[:, None] & in_fallback)
    return np.where(sampled, reported, np.nan)


def _reference_poisson_choices_df(
    probs, draws, sample_size, alternatives, alt_col_name
):
    """Flatten `_reference_poisson_sampled_values` into the expected choices frame."""
    sampled_values = _reference_poisson_sampled_values(
        probs.to_numpy(), draws, sample_size
    )
    chooser_idx, alt_idx = np.nonzero(~np.isnan(sampled_values))
    return pd.DataFrame(
        {
            probs.index.name: probs.index.to_numpy()[chooser_idx],
            "prob": sampled_values[chooser_idx, alt_idx],
            alt_col_name: alternatives.index.to_numpy()[alt_idx],
        }
    )


def test_poisson_sample_alternatives_inner_returns_inclusion_mask():
    probs = pd.DataFrame(
        [[0.2, 0.4, 0.6], [0.1, 0.3, 0.5]],
        index=pd.Index([11, 17], name="person_id"),
        columns=[0, 1, 2],
    )
    inclusion_probs_values = np.array(
        [[0.36, 0.64, 0.84], [0.19, 0.51, 0.75]],
        dtype=np.float64,
    )
    rng = _SequentialDummyRng(
        [
            np.array(
                [[0.10, 0.80, 0.20], [0.30, 0.50, 0.90]],
                dtype=np.float64,
            )
        ]
    )

    sampled = interaction_sample._poisson_sample_alternatives_inner(
        probs,
        inclusion_probs_values,
        rng,
        trace_label="test_poisson_sample_alternatives_inner_returns_inclusion_mask",
        chunk_sizer=_DummyChunkSizer(),
    )

    expected = np.array(
        [[True, False, True], [False, True, False]],
        dtype=bool,
    )

    np.testing.assert_array_equal(sampled, expected)


def test_poisson_fallback_positions_selects_highest_probability_alternatives():
    probs_values = np.array(
        [[0.20, 0.30, 0.50, 0.00], [0.40, 0.10, 0.30, 0.20]],
        dtype=np.float64,
    )

    positions = interaction_sample._poisson_fallback_positions(probs_values, 2)

    # highest probability first, so [0.50, 0.30] and [0.40, 0.30]
    np.testing.assert_array_equal(positions, np.array([[2, 1], [0, 2]]))


def test_poisson_fallback_positions_breaks_ties_by_column_and_caps_at_alt_count():
    probs_values = np.array([[0.25, 0.25, 0.25, 0.25]], dtype=np.float64)

    # ties resolve to the leading columns, deterministically
    np.testing.assert_array_equal(
        interaction_sample._poisson_fallback_positions(probs_values, 2),
        np.array([[0, 1]]),
    )

    # asking for more alternatives than exist returns all of them
    np.testing.assert_array_equal(
        interaction_sample._poisson_fallback_positions(probs_values, 99),
        np.array([[0, 1, 2, 3]]),
    )


def test_make_sample_choices_poisson_returns_expected_frames():
    probs = pd.DataFrame(
        [
            [0.20, 0.60, 0.10, 0.05],
            [0.40, 0.10, 0.30, 0.20],
            [0.30, 0.20, 0.70, 0.10],
        ],
        index=pd.Index([11, 17, 42], name="person_id"),
        columns=np.arange(4),
    )
    sample_size = 2
    alternatives = pd.DataFrame(index=pd.Index([100, 300, 700, 900], name="alt_id"))
    # the middle chooser samples nothing and takes the fallback set
    draws = np.array(
        [
            [0.10, 0.90, 0.50, 0.90],
            [0.90, 0.90, 0.90, 0.90],
            [0.80, 0.90, 0.20, 0.80],
        ],
        dtype=np.float64,
    )
    state = _DummyState(_SequentialDummyRng([draws]))

    choices_df = interaction_sample.make_sample_choices_poisson(
        chunk_sizer=_DummyChunkSizer(),
        probs=probs,
        alternatives=alternatives,
        sample_size=sample_size,
        alt_col_name="alt_id",
        state=state,
        trace_label="test_make_sample_choices_poisson_returns_expected_frames",
    )

    expected = _reference_poisson_choices_df(
        probs, draws, sample_size, alternatives, "alt_id"
    )
    pd.testing.assert_frame_equal(choices_df, expected)

    # the fallback chooser gets the two most likely alternatives, 100 and 700
    assert choices_df.loc[choices_df.person_id == 17, "alt_id"].tolist() == [100, 700]


def test_make_sample_choices_poisson_consumes_no_extra_randoms_on_empty_draw():
    # the fallback must not draw again: _SequentialDummyRng raises IndexError if the
    # sampler asks for a second block, so a single draw array is the assertion here
    probs = pd.DataFrame(
        [[0.20, 0.30, 0.50]],
        index=pd.Index([11], name="person_id"),
        columns=np.arange(3),
    )
    sample_size = 2
    alternatives = pd.DataFrame(index=pd.Index([100, 300, 700], name="alt_id"))
    fail_draw = np.array([[0.99, 0.99, 0.99]], dtype=np.float64)
    state = _DummyState(_SequentialDummyRng([fail_draw]))

    choices_df = interaction_sample.make_sample_choices_poisson(
        chunk_sizer=_DummyChunkSizer(),
        probs=probs,
        alternatives=alternatives,
        sample_size=sample_size,
        alt_col_name="alt_id",
        state=state,
        trace_label="test_make_sample_choices_poisson_consumes_no_extra_randoms_on_empty_draw",
    )

    # the two most likely alternatives, reported at q_i + P0
    inclusion_probs = 1 - np.power(1 - probs.to_numpy(), sample_size)
    empty_sample_prob = np.prod(1 - inclusion_probs, axis=1)[0]
    expected = pd.DataFrame(
        {
            "person_id": [11, 11],
            "prob": [
                inclusion_probs[0, 1] + empty_sample_prob,
                inclusion_probs[0, 2] + empty_sample_prob,
            ],
            "alt_id": [300, 700],
        }
    )

    pd.testing.assert_frame_equal(choices_df, expected)


def test_make_sample_choices_poisson_fallback_excludes_unavailable_alternatives():
    # a chooser with fewer available (p > 0) alternatives than the fallback window must
    # not have its fallback set padded with unavailable alternatives: those would enter
    # the final choice set carrying a large positive correction term log(1/P0)
    probs = pd.DataFrame(
        [[0.60, 0.40, 0.00, 0.00]],
        index=pd.Index([11], name="person_id"),
        columns=np.arange(4),
    )
    sample_size = 3
    alternatives = pd.DataFrame(index=pd.Index([100, 300, 700, 900], name="alt_id"))
    # both available alternatives fail their inclusion draw, forcing the fallback
    fail_draw = np.array([[0.99, 0.99, 0.99, 0.99]], dtype=np.float64)
    state = _DummyState(_SequentialDummyRng([fail_draw]))

    choices_df = interaction_sample.make_sample_choices_poisson(
        chunk_sizer=_DummyChunkSizer(),
        probs=probs,
        alternatives=alternatives,
        sample_size=sample_size,
        alt_col_name="alt_id",
        state=state,
        trace_label="test_make_sample_choices_poisson_fallback_excludes_unavailable_alternatives",
    )

    # only the two available alternatives are returned, each reported at q_i + P0,
    # even though the fallback window min(sample_size, n_alts) = 3 is wider
    inclusion_probs = 1 - np.power(1 - probs.to_numpy(), sample_size)
    empty_sample_prob = np.prod(1 - inclusion_probs, axis=1)[0]
    expected = pd.DataFrame(
        {
            "person_id": [11, 11],
            "prob": [
                inclusion_probs[0, 0] + empty_sample_prob,
                inclusion_probs[0, 1] + empty_sample_prob,
            ],
            "alt_id": [100, 300],
        }
    )
    pd.testing.assert_frame_equal(choices_df, expected)

    # the reference implementation agrees
    pd.testing.assert_frame_equal(
        choices_df,
        _reference_poisson_choices_df(
            probs, fail_draw, sample_size, alternatives, "alt_id"
        ),
    )


def test_make_sample_choices_poisson_reported_prob_is_total_inclusion_probability():
    # Monte Carlo check that the reported `prob` really is the probability of the
    # alternative ending up in the choice set, counting both the Bernoulli draw and the
    # fallback. Every chooser is identical, so the empirical inclusion rate across
    # choosers estimates that probability directly. sample_size=1 over 6 uniform
    # alternatives makes empty draws frequent (P0 = (5/6)^6 ~ 0.33), which is what puts
    # the fallback term under test.
    n_choosers = 200_000
    n_alts = 6
    sample_size = 1

    probs = pd.DataFrame(
        np.full((n_choosers, n_alts), 1.0 / n_alts),
        index=pd.Index(np.arange(n_choosers), name="person_id"),
        columns=np.arange(n_alts),
    )
    alternatives = pd.DataFrame(index=pd.Index(np.arange(n_alts) * 10, name="alt_id"))

    inclusion_probs = 1 - np.power(1 - probs.to_numpy(), sample_size)
    empty_sample_prob = np.prod(1 - inclusion_probs, axis=1)[0]
    assert empty_sample_prob > 0.3

    draws = np.random.default_rng(20260726).random((n_choosers, n_alts))
    state = _DummyState(_SequentialDummyRng([draws]))

    choices_df = interaction_sample.make_sample_choices_poisson(
        chunk_sizer=_DummyChunkSizer(),
        probs=probs,
        alternatives=alternatives,
        sample_size=sample_size,
        alt_col_name="alt_id",
        state=state,
        trace_label="test_make_sample_choices_poisson_reported_prob_is_total_inclusion_probability",
    )

    # identical choosers must get an identical reported prob per alternative, whether
    # they reached the choice set through the Bernoulli draw or through the fallback
    reported = choices_df.groupby("alt_id")["prob"].agg(["min", "max", "first"])
    np.testing.assert_allclose(reported["min"], reported["max"], rtol=1e-12)

    # ties in the fallback resolve to the first column, so only alternative 0 carries
    # the extra P0 mass
    expected_reported = np.full(n_alts, inclusion_probs[0, 0])
    expected_reported[0] += empty_sample_prob
    np.testing.assert_allclose(reported["first"], expected_reported, rtol=1e-12)

    empirical = choices_df.groupby("alt_id").size() / n_choosers
    np.testing.assert_allclose(empirical, expected_reported, atol=0.005)


def test_repeat_alignment_chooser_heterogeneity():
    # Edge case: utilities are close across alternatives but vary strongly by chooser.
    # This checks that the flattened Poisson result keeps chooser/prob alignment.
    chooser_index = pd.Index([101, 102, 103, 104, 105, 106], name="person_id")
    choosers = pd.DataFrame(index=chooser_index)
    alternatives = pd.DataFrame(index=pd.Index([0, 1, 2, 3], name="alt_id"))

    sample_size = 3

    # Very small alternative differences...
    alt_signal = np.array([0.00, 0.01, 0.02, 0.03], dtype=np.float64)
    # ...but very large chooser sensitivity differences.
    chooser_scale = np.array([-500.0, -200.0, -50.0, 50.0, 200.0, 500.0])

    utilities = pd.DataFrame(
        chooser_scale[:, np.newaxis] * alt_signal[np.newaxis, :],
        index=chooser_index,
    )

    poisson_draws = np.array(
        [
            [0.01, 0.90, 0.90, 0.90],
            [0.80, 0.05, 0.90, 0.90],
            [0.90, 0.10, 0.40, 0.90],
            [0.90, 0.90, 0.10, 0.20],
            [0.90, 0.90, 0.02, 0.10],
            [0.90, 0.90, 0.90, 0.001],
        ],
        dtype=np.float64,
    )
    state = _DummyState(_SequentialDummyRng([poisson_draws]))

    probs = interaction_sample.logit.utils_to_probs(
        state,
        utilities,
        allow_zero_probs=False,
        trace_label="test_repeat_alignment_chooser_heterogeneity",
        overflow_protection=True,
        trace_choosers=choosers,
    )

    out = interaction_sample.make_sample_choices_poisson(
        chunk_sizer=_DummyChunkSizer(),
        probs=probs,
        alternatives=alternatives,
        sample_size=sample_size,
        alt_col_name="alt_id",
        state=state,
        trace_label="test_repeat_alignment_chooser_heterogeneity",
    )

    expected = _reference_poisson_choices_df(
        probs, poisson_draws, sample_size, alternatives, "alt_id"
    )

    pd.testing.assert_frame_equal(out.reset_index(drop=True), expected)


def test_make_sample_choices_poisson_matches_materialized_path():
    chooser_index = pd.Index([201, 202, 203], name="person_id")
    choosers = pd.DataFrame(index=chooser_index)
    alternatives = pd.DataFrame(index=pd.Index([10, 11, 12, 13], name="alt_id"))
    utilities = pd.DataFrame(
        [[0.0, 0.3, -0.2, 0.1], [1.0, 0.2, 0.4, -0.5], [-0.1, 0.0, 0.8, 0.7]],
        index=chooser_index,
    )
    sample_size = 2
    poisson_draws = np.array(
        [
            [0.10, 0.20, 0.50, 0.00],
            [0.60, 0.50, 0.10, 0.40],
            [0.00, 0.30, 0.20, 0.90],
        ],
        dtype=np.float64,
    )
    state = _DummyState(_SequentialDummyRng([poisson_draws]))

    probs = interaction_sample.logit.utils_to_probs(
        state,
        utilities,
        allow_zero_probs=False,
        trace_label="test_fused_rng_matches_materialized",
        overflow_protection=True,
        trace_choosers=choosers,
    )

    out = interaction_sample.make_sample_choices_poisson(
        chunk_sizer=_DummyChunkSizer(),
        probs=probs,
        alternatives=alternatives,
        sample_size=sample_size,
        alt_col_name="alt_id",
        state=state,
        trace_label="test_fused_rng_matches_materialized",
    )

    expected = _reference_poisson_choices_df(
        probs, poisson_draws, sample_size, alternatives, "alt_id"
    )

    pd.testing.assert_frame_equal(out.reset_index(drop=True), expected)


def test_make_sample_choices_eet_matches_materialized_path():
    chooser_index = pd.Index([201, 202, 203], name="person_id")
    choosers = pd.DataFrame(index=chooser_index)
    alternatives = pd.DataFrame(index=pd.Index([10, 11, 12, 13], name="alt_id"))
    utilities = pd.DataFrame(
        [[0.0, 0.3, -0.2, 0.1], [1.0, 0.2, 0.4, -0.5], [-0.1, 0.0, 0.8, 0.7]],
        index=chooser_index,
    )
    sample_size = 2
    rands_3d = np.array(
        [
            [[0.1, -0.3], [0.2, 0.4], [0.5, -0.1], [0.0, 0.2]],
            [[-0.2, 0.3], [0.6, -0.5], [0.1, 0.7], [0.4, 0.2]],
            [[0.0, 0.1], [0.3, -0.4], [0.2, 0.5], [-0.3, 0.2]],
        ],
        dtype=np.float64,
    )
    state = _DummyState(_DummyRngUtilityBased(rands_3d))

    probs = interaction_sample.logit.utils_to_probs(
        state,
        utilities,
        allow_zero_probs=False,
        trace_label="test_make_sample_choices_eet_matches_materialized_path",
        overflow_protection=True,
        trace_choosers=choosers,
    )

    out = interaction_sample.make_sample_choices_eet(
        state=state,
        choosers=choosers,
        utilities=utilities,
        probs=probs,
        alternatives=alternatives,
        sample_size=sample_size,
        alt_col_name="alt_id",
        trace_label="test_make_sample_choices_eet_matches_materialized_path",
        chunk_sizer=_DummyChunkSizer(),
    )

    chosen_positions = np.argmax(
        rands_3d + utilities.to_numpy()[:, :, np.newaxis],
        axis=1,
    )
    chosen_flat = chosen_positions.reshape(-1)
    chooser_idx = np.repeat(np.arange(len(choosers)), sample_size)

    expected = pd.DataFrame(
        {
            "person_id": choosers.index.values[chooser_idx],
            "prob": probs.to_numpy()[chooser_idx, chosen_flat],
            "alt_id": alternatives.index.values[chosen_flat],
        }
    )

    pd.testing.assert_frame_equal(out.reset_index(drop=True), expected)


def test_make_sample_choices_eet_stable_alt_mapping_matches_materialized_path():
    chooser_index = pd.Index([301, 302], name="person_id")
    choosers = pd.DataFrame(index=chooser_index)
    alternatives = pd.DataFrame(index=pd.Index([10, 12, 14], name="alt_id"))
    utilities = pd.DataFrame(
        [[0.0, 0.3, -0.2], [1.0, 0.2, 0.4]],
        index=chooser_index,
    )
    sample_size = 2
    stable_alt_positions = np.array([0, 2, 4], dtype=np.int64)
    n_total_alts = 5
    dense_rands_3d = np.array(
        [
            [[0.1, -0.3], [0.4, 0.2], [0.2, 0.4], [0.3, -0.2], [0.5, -0.1]],
            [[-0.2, 0.3], [0.0, 0.5], [0.6, -0.5], [0.2, 0.1], [0.1, 0.7]],
        ],
        dtype=np.float64,
    )
    state = _DummyState(_DummyRngUtilityBased(dense_rands_3d))

    probs = interaction_sample.logit.utils_to_probs(
        state,
        utilities,
        allow_zero_probs=False,
        trace_label="test_make_sample_choices_eet_stable_alt_mapping_matches_materialized_path",
        overflow_protection=True,
        trace_choosers=choosers,
    )

    out = interaction_sample.make_sample_choices_eet(
        state=state,
        choosers=choosers,
        utilities=utilities,
        probs=probs,
        alternatives=alternatives,
        sample_size=sample_size,
        alt_col_name="alt_id",
        trace_label="test_make_sample_choices_eet_stable_alt_mapping_matches_materialized_path",
        chunk_sizer=_DummyChunkSizer(),
        stable_alt_positions=stable_alt_positions,
        n_total_alts=n_total_alts,
    )

    active_rands = dense_rands_3d[:, stable_alt_positions, :]
    chosen_positions = np.argmax(
        active_rands + utilities.to_numpy()[:, :, np.newaxis],
        axis=1,
    )
    chosen_flat = chosen_positions.reshape(-1)
    chooser_idx = np.repeat(np.arange(len(choosers)), sample_size)

    expected = pd.DataFrame(
        {
            "person_id": choosers.index.values[chooser_idx],
            "prob": probs.to_numpy()[chooser_idx, chosen_flat],
            "alt_id": alternatives.index.values[chosen_flat],
        }
    )

    pd.testing.assert_frame_equal(out.reset_index(drop=True), expected)


def test_make_sample_choices_poisson_stable_alt_mapping_matches_materialized_path():
    chooser_index = pd.Index([311, 312], name="person_id")
    choosers = pd.DataFrame(index=chooser_index)
    alternatives = pd.DataFrame(index=pd.Index([10, 12, 14], name="alt_id"))
    utilities = pd.DataFrame(
        [[0.0, 0.3, -0.2], [1.0, 0.2, 0.4]],
        index=chooser_index,
    )
    sample_size = 2
    stable_alt_positions = np.array([0, 2, 4], dtype=np.int64)
    n_total_alts = 5
    dense_uniforms = np.array(
        [
            [0.05, 0.90, 0.10, 0.80, 0.20],
            [0.90, 0.70, 0.05, 0.60, 0.10],
        ],
        dtype=np.float64,
    )
    state = _DummyState(_SequentialDummyRng([dense_uniforms]))

    probs = interaction_sample.logit.utils_to_probs(
        state,
        utilities,
        allow_zero_probs=False,
        trace_label="test_make_sample_choices_poisson_stable_alt_mapping_matches_materialized_path",
        overflow_protection=True,
        trace_choosers=choosers,
    )

    out = interaction_sample.make_sample_choices_poisson(
        chunk_sizer=_DummyChunkSizer(),
        probs=probs,
        alternatives=alternatives,
        sample_size=sample_size,
        alt_col_name="alt_id",
        state=state,
        trace_label="test_make_sample_choices_poisson_stable_alt_mapping_matches_materialized_path",
        stable_alt_positions=stable_alt_positions,
        n_total_alts=n_total_alts,
    )

    expected = _reference_poisson_choices_df(
        probs,
        dense_uniforms[:, stable_alt_positions],
        sample_size,
        alternatives,
        "alt_id",
    )

    pd.testing.assert_frame_equal(out.reset_index(drop=True), expected)


def test_make_sample_choices_poisson_falls_back_to_most_likely_alternatives():
    chooser_index = pd.Index([301, 302], name="person_id")
    choosers = pd.DataFrame(index=chooser_index)
    alternatives = pd.DataFrame(index=pd.Index([10, 12, 14], name="alt_id"))
    utilities = pd.DataFrame(
        [[0.0, 0.3, -0.2], [1.0, 0.2, 0.4]],
        index=chooser_index,
    )
    sample_size = 2
    fail_draw = np.full((2, 3), 0.99, dtype=np.float64)
    state = _DummyState(_SequentialDummyRng([fail_draw]))

    probs = interaction_sample.logit.utils_to_probs(
        state,
        utilities,
        allow_zero_probs=False,
        trace_label="test_make_sample_choices_poisson_falls_back_to_most_likely_alternatives",
        overflow_protection=True,
        trace_choosers=choosers,
    )

    out = interaction_sample.make_sample_choices_poisson(
        chunk_sizer=_DummyChunkSizer(),
        probs=probs,
        alternatives=alternatives,
        sample_size=sample_size,
        alt_col_name="alt_id",
        state=state,
        trace_label="test_make_sample_choices_poisson_falls_back_to_most_likely_alternatives",
    )

    # neither chooser sampled anything, so both take their two most likely
    # alternatives: utilities [0.0, 0.3, -0.2] -> alts 12, 10 and
    # [1.0, 0.2, 0.4] -> alts 10, 14
    assert out["alt_id"].tolist() == [10, 12, 10, 14]
    assert out["person_id"].tolist() == [301, 301, 302, 302]

    expected = _reference_poisson_choices_df(
        probs, fail_draw, sample_size, alternatives, "alt_id"
    )
    pd.testing.assert_frame_equal(out.reset_index(drop=True), expected)

    # reported prob is q_i + P0, strictly below 1 and strictly above the bare q_i
    inclusion_probs = 1 - np.power(1 - probs.to_numpy(), sample_size)
    empty_sample_probs = np.prod(1 - inclusion_probs, axis=1)
    assert (out["prob"] < 1.0).all()
    assert (empty_sample_probs > 0).all()
