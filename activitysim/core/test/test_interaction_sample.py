# ActivitySim
# See full license in LICENSE.txt.

from __future__ import annotations

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


class _DummyChunkSizer:
    def log_df(self, *_args, **_kwargs):
        return None


class _DummyState:
    def __init__(self, rng):
        self._rng = rng

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


def _expected_choices_df(sampled_alternatives, alternatives, alt_col_name):
    return (
        sampled_alternatives.rename_axis("alt_idx", axis=1)
        .stack()
        .reset_index(name="prob")
        .assign(**{alt_col_name: lambda df: alternatives.index.values[df["alt_idx"]]})
        .drop(columns=["alt_idx"])
    )


def test_poisson_sample_alternatives_inner_returns_masked_inclusion_probs():
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
        trace_label="test_poisson_sample_alternatives_inner_returns_masked_inclusion_probs",
        chunk_sizer=_DummyChunkSizer(),
    )

    expected = np.array(
        [[0.36, np.nan, 0.84], [np.nan, 0.51, np.nan]],
        dtype=np.float64,
    )

    np.testing.assert_allclose(sampled, expected, equal_nan=True)


def test_poisson_fallback_sample_alternatives_selects_distinct_positions_with_prob_one():
    probs = pd.DataFrame(
        [[0.20, 0.30, 0.50, 0.00], [0.40, 0.10, 0.30, 0.20]],
        index=pd.Index([11, 17], name="person_id"),
        columns=np.arange(4),
    )
    rng = _SequentialDummyRng(
        [
            np.array(
                [[0.90, 0.10, 0.40, 0.20], [0.05, 0.70, 0.60, 0.10]],
                dtype=np.float64,
            )
        ]
    )

    sampled = interaction_sample._poisson_fallback_sample_alternatives(
        probs=probs,
        sample_size=2,
        rng=rng,
        trace_label="test_poisson_fallback_sample_alternatives_selects_distinct_positions_with_prob_one",
        chunk_sizer=_DummyChunkSizer(),
    )

    expected = np.array(
        [[np.nan, 1.0, np.nan, 1.0], [1.0, np.nan, np.nan, 1.0]],
        dtype=np.float64,
    )

    np.testing.assert_allclose(sampled, expected, equal_nan=True)


def test_poisson_sample_alternatives_retries_and_returns_expected_frames():
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
    expected_inclusion_probs = 1 - (1 - probs) ** sample_size
    expected_sampled_alternatives = pd.DataFrame(
        [
            [expected_inclusion_probs.iloc[0, 0], np.nan, np.nan, np.nan],
            [
                expected_inclusion_probs.iloc[1, 0],
                expected_inclusion_probs.iloc[1, 1],
                np.nan,
                np.nan,
            ],
            [np.nan, np.nan, expected_inclusion_probs.iloc[2, 2], np.nan],
        ],
        index=probs.index,
        columns=probs.columns,
    )
    state = _DummyState(
        _SequentialDummyRng(
            [
                np.array(
                    [
                        [0.10, 0.90, 0.50, 0.90],
                        [0.90, 0.90, 0.90, 0.90],
                        [0.80, 0.90, 0.20, 0.80],
                    ],
                    dtype=np.float64,
                ),
                np.array([[0.10, 0.05, 0.70, 0.80]], dtype=np.float64),
            ]
        )
    )

    choices_df = interaction_sample._poisson_sample_alternatives(
        chunk_sizer=_DummyChunkSizer(),
        probs=probs,
        alternatives=alternatives,
        sample_size=sample_size,
        alt_col_name="alt_id",
        state=state,
        trace_label="test_poisson_sample_alternatives_retries_and_returns_expected_frames",
    )

    expected_choices_df = _expected_choices_df(
        expected_sampled_alternatives,
        alternatives,
        "alt_id",
    )

    pd.testing.assert_frame_equal(choices_df, expected_choices_df)


def test_poisson_sample_alternatives_falls_back_to_random_sampling_after_ten_retries():
    probs = pd.DataFrame(
        [[0.20, 0.30, 0.50]],
        index=pd.Index([11], name="person_id"),
        columns=np.arange(3),
    )
    sample_size = 2
    alternatives = pd.DataFrame(index=pd.Index([100, 300, 700], name="alt_id"))
    fail_draw = np.array([[0.99, 0.99, 0.99]], dtype=np.float64)
    fallback_draw = np.array([[0.10, 0.80, 0.20]], dtype=np.float64)
    state = _DummyState(_SequentialDummyRng([fail_draw] * 10 + [fallback_draw]))

    choices_df = interaction_sample._poisson_sample_alternatives(
        chunk_sizer=_DummyChunkSizer(),
        probs=probs,
        alternatives=alternatives,
        sample_size=sample_size,
        alt_col_name="alt_id",
        state=state,
        trace_label="test_poisson_sample_alternatives_falls_back_to_random_sampling_after_ten_retries",
    )

    expected_sampled_alternatives = pd.DataFrame(
        [[1.0, np.nan, 1.0]],
        index=probs.index,
        columns=probs.columns,
    )
    expected_choices_df = _expected_choices_df(
        expected_sampled_alternatives,
        alternatives,
        "alt_id",
    )

    pd.testing.assert_frame_equal(choices_df, expected_choices_df)


def test_make_sample_choices_utility_based_repeat_alignment_chooser_dominant_heterogeneity():
    # Edge case: utilities are close across alternatives but vary strongly by chooser.
    # This checks that the flattened Poisson result keeps chooser/prob alignment.
    chooser_index = pd.Index([101, 102, 103, 104, 105, 106], name="person_id")
    choosers = pd.DataFrame(index=chooser_index)
    alternatives = pd.DataFrame(index=pd.Index([0, 1, 2, 3], name="alt_id"))

    n_alts = len(alternatives)
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

    out = interaction_sample.make_sample_choices_utility_based(
        state=state,
        choosers=choosers,
        utilities=utilities,
        alternatives=alternatives,
        sample_size=sample_size,
        alternative_count=n_alts,
        alt_col_name="alt_id",
        allow_zero_probs=False,
        trace_label="test_repeat_alignment_chooser_heterogeneity",
        chunk_sizer=_DummyChunkSizer(),
        sampling_method="poisson",
    )

    probs = interaction_sample.logit.utils_to_probs(
        state,
        utilities,
        allow_zero_probs=False,
        trace_label="test_repeat_alignment_chooser_heterogeneity",
        overflow_protection=True,
        trace_choosers=choosers,
    ).to_numpy()
    inclusion_probs = 1 - np.power(1 - probs, sample_size)
    sampled_values = np.where(poisson_draws < inclusion_probs, inclusion_probs, np.nan)
    chooser_idx, alt_idx = np.nonzero(~np.isnan(sampled_values))

    expected = pd.DataFrame(
        {
            "person_id": chooser_index.to_numpy()[chooser_idx],
            "prob": sampled_values[chooser_idx, alt_idx],
            "alt_id": alternatives.index.to_numpy()[alt_idx],
        }
    )

    pd.testing.assert_frame_equal(out.reset_index(drop=True), expected)


def test_make_sample_choices_utility_based_poisson_retry_matches_materialized_path():
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
    retry_draw = np.array([[0.40, 0.10, 0.90, 0.90]], dtype=np.float64)
    state = _DummyState(_SequentialDummyRng([poisson_draws, retry_draw]))

    out = interaction_sample.make_sample_choices_utility_based(
        state=state,
        choosers=choosers,
        utilities=utilities,
        alternatives=alternatives,
        sample_size=sample_size,
        alternative_count=len(alternatives),
        alt_col_name="alt_id",
        allow_zero_probs=False,
        trace_label="test_fused_rng_matches_materialized",
        chunk_sizer=_DummyChunkSizer(),
        sampling_method="poisson",
    )

    probs = interaction_sample.logit.utils_to_probs(
        state,
        utilities,
        allow_zero_probs=False,
        trace_label="test_fused_rng_matches_materialized",
        overflow_protection=True,
        trace_choosers=choosers,
    ).to_numpy()
    inclusion_probs = 1 - np.power(1 - probs, sample_size)
    sampled_values = np.full(inclusion_probs.shape, np.nan)
    first_pass = np.where(poisson_draws < inclusion_probs, inclusion_probs, np.nan)
    first_pass_empty = np.isnan(first_pass).all(axis=1)
    sampled_values[~first_pass_empty] = first_pass[~first_pass_empty]
    retry_pass = np.where(
        retry_draw < inclusion_probs[first_pass_empty],
        inclusion_probs[first_pass_empty],
        np.nan,
    )
    sampled_values[first_pass_empty] = retry_pass
    chooser_idx, alt_idx = np.nonzero(~np.isnan(sampled_values))

    expected = pd.DataFrame(
        {
            "person_id": choosers.index.values[chooser_idx],
            "prob": sampled_values[chooser_idx, alt_idx],
            "alt_id": alternatives.index.values[alt_idx],
        }
    )

    pd.testing.assert_frame_equal(out.reset_index(drop=True), expected)


def test_make_sample_choices_utility_based_eet_matches_materialized_path():
    chooser_index = pd.Index([201, 202, 203], name="person_id")
    choosers = pd.DataFrame(index=chooser_index)
    alternatives = pd.DataFrame(index=pd.Index([10, 11, 12, 13], name="alt_id"))
    utilities = pd.DataFrame(
        [[0.0, 0.3, -0.2, 0.1], [1.0, 0.2, 0.4, -0.5], [-0.1, 0.0, 0.8, 0.7]],
        index=chooser_index,
    )
    sample_size = 2
    n_alts = len(alternatives)
    rands_3d = np.array(
        [
            [[0.1, -0.3], [0.2, 0.4], [0.5, -0.1], [0.0, 0.2]],
            [[-0.2, 0.3], [0.6, -0.5], [0.1, 0.7], [0.4, 0.2]],
            [[0.0, 0.1], [0.3, -0.4], [0.2, 0.5], [-0.3, 0.2]],
        ],
        dtype=np.float64,
    )
    state = _DummyState(_DummyRngUtilityBased(rands_3d))

    out = interaction_sample.make_sample_choices_utility_based(
        state=state,
        choosers=choosers,
        utilities=utilities,
        alternatives=alternatives,
        sample_size=sample_size,
        alternative_count=n_alts,
        alt_col_name="alt_id",
        allow_zero_probs=False,
        trace_label="test_make_sample_choices_utility_based_eet_matches_materialized_path",
        chunk_sizer=_DummyChunkSizer(),
        sampling_method="eet",
    )

    chosen_positions = np.argmax(
        rands_3d + utilities.to_numpy()[:, :, np.newaxis],
        axis=1,
    )
    chosen_flat = chosen_positions.reshape(-1)
    chooser_idx = np.repeat(np.arange(len(choosers)), sample_size)
    probs = interaction_sample.logit.utils_to_probs(
        state,
        utilities,
        allow_zero_probs=False,
        trace_label="test_make_sample_choices_utility_based_eet_matches_materialized_path",
        overflow_protection=True,
        trace_choosers=choosers,
    ).to_numpy()

    expected = pd.DataFrame(
        {
            "person_id": choosers.index.values[chooser_idx],
            "prob": probs[chooser_idx, chosen_flat],
            "alt_id": alternatives.index.values[chosen_flat],
        }
    )

    pd.testing.assert_frame_equal(out.reset_index(drop=True), expected)


def test_make_sample_choices_utility_based_eet_stable_alt_mapping_matches_materialized_path():
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

    out = interaction_sample.make_sample_choices_utility_based(
        state=state,
        choosers=choosers,
        utilities=utilities,
        alternatives=alternatives,
        sample_size=sample_size,
        alternative_count=len(alternatives),
        alt_col_name="alt_id",
        allow_zero_probs=False,
        trace_label="test_make_sample_choices_utility_based_eet_stable_alt_mapping_matches_materialized_path",
        chunk_sizer=_DummyChunkSizer(),
        sampling_method="eet",
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
    probs = interaction_sample.logit.utils_to_probs(
        state,
        utilities,
        allow_zero_probs=False,
        trace_label="test_make_sample_choices_utility_based_eet_stable_alt_mapping_matches_materialized_path",
        overflow_protection=True,
        trace_choosers=choosers,
    ).to_numpy()

    expected = pd.DataFrame(
        {
            "person_id": choosers.index.values[chooser_idx],
            "prob": probs[chooser_idx, chosen_flat],
            "alt_id": alternatives.index.values[chosen_flat],
        }
    )

    pd.testing.assert_frame_equal(out.reset_index(drop=True), expected)


def test_make_sample_choices_utility_based_poisson_stable_alt_mapping_matches_materialized_path():
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

    out = interaction_sample.make_sample_choices_utility_based(
        state=state,
        choosers=choosers,
        utilities=utilities,
        alternatives=alternatives,
        sample_size=sample_size,
        alternative_count=len(alternatives),
        alt_col_name="alt_id",
        allow_zero_probs=False,
        trace_label="test_make_sample_choices_utility_based_poisson_stable_alt_mapping_matches_materialized_path",
        chunk_sizer=_DummyChunkSizer(),
        sampling_method="poisson",
        stable_alt_positions=stable_alt_positions,
        n_total_alts=n_total_alts,
    )

    probs = interaction_sample.logit.utils_to_probs(
        state,
        utilities,
        allow_zero_probs=False,
        trace_label="test_make_sample_choices_utility_based_poisson_stable_alt_mapping_matches_materialized_path",
        overflow_protection=True,
        trace_choosers=choosers,
    ).to_numpy()
    inclusion_probs = 1 - np.power(1 - probs, sample_size)
    active_uniforms = dense_uniforms[:, stable_alt_positions]
    sampled_values = np.where(
        active_uniforms < inclusion_probs, inclusion_probs, np.nan
    )
    chooser_idx, alt_idx = np.nonzero(~np.isnan(sampled_values))

    expected = pd.DataFrame(
        {
            "person_id": choosers.index.values[chooser_idx],
            "prob": sampled_values[chooser_idx, alt_idx],
            "alt_id": alternatives.index.values[alt_idx],
        }
    )

    pd.testing.assert_frame_equal(out.reset_index(drop=True), expected)


def test_make_sample_choices_utility_based_falls_back_after_retries():
    chooser_index = pd.Index([301, 302], name="person_id")
    choosers = pd.DataFrame(index=chooser_index)
    alternatives = pd.DataFrame(index=pd.Index([10, 12, 14], name="alt_id"))
    utilities = pd.DataFrame(
        [[0.0, 0.3, -0.2], [1.0, 0.2, 0.4]],
        index=chooser_index,
    )
    sample_size = 2
    fail_draw = np.full((2, 3), 0.99, dtype=np.float64)
    fallback_draw = np.array(
        [
            [0.40, 0.10, 0.20],
            [0.30, 0.20, 0.90],
        ],
        dtype=np.float64,
    )
    state = _DummyState(_SequentialDummyRng([fail_draw] * 10 + [fallback_draw]))

    out = interaction_sample.make_sample_choices_utility_based(
        state=state,
        choosers=choosers,
        utilities=utilities,
        alternatives=alternatives,
        sample_size=sample_size,
        alternative_count=len(alternatives),
        alt_col_name="alt_id",
        allow_zero_probs=False,
        trace_label="test_falls_back_after_retries",
        chunk_sizer=_DummyChunkSizer(),
        sampling_method="poisson",
    )

    expected = pd.DataFrame(
        {
            "person_id": [301, 301, 302, 302],
            "prob": [1.0, 1.0, 1.0, 1.0],
            "alt_id": [12, 14, 10, 12],
        }
    )

    pd.testing.assert_frame_equal(out.reset_index(drop=True), expected)
