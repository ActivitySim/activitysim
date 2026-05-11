# ActivitySim
# See full license in LICENSE.txt.

import numpy as np
import pandas as pd
import pytest

from activitysim.core import interaction_sample, workflow


@pytest.fixture
def state() -> workflow.State:
    state = workflow.State().default_settings()
    state.settings.check_for_variability = False
    return state


def test_interaction_sample_parity(state):
    # Run interaction_sample with and without explicit error terms and check that results are similar.

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

    # Run _without_ explicit error terms
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

    # Run _with_ explicit error terms
    state.init_state()  # reset the state to rerun with same seed
    state.settings.use_explicit_error_terms = True
    state.rng().set_base_seed(42)
    state.rng().add_channel("person_id", choosers)
    state.rng().begin_step("test_step_explicit")

    choices_explicit = interaction_sample.interaction_sample(
        state,
        choosers,
        alternatives,
        spec,
        sample_size=sample_size,
        alt_col_name="alt_id",
    )

    assert "alt_id" in choices_mnl.columns
    assert "alt_id" in choices_explicit.columns
    assert not choices_mnl["alt_id"].isna().any()
    assert not choices_explicit["alt_id"].isna().any()
    assert choices_mnl["alt_id"].isin(alternatives.index).all()
    assert choices_explicit["alt_id"].isin(alternatives.index).all()

    # In interaction_sample, choices_explicit and choices_mnl are DataFrames with sampled alternatives.
    # The statistics of chosen alternatives should be similar.
    mnl_counts = choices_mnl["alt_id"].value_counts(normalize=True).sort_index()
    explicit_counts = (
        choices_explicit["alt_id"].value_counts(normalize=True).sort_index()
    )

    # Check top choices overlap significantly or shares are close
    all_alts = set(mnl_counts.index) | set(explicit_counts.index)
    for alt in all_alts:
        share_mnl = mnl_counts.get(alt, 0)
        share_explicit = explicit_counts.get(alt, 0)
        diff = abs(share_mnl - share_explicit)
        assert diff < 0.01, (
            f"Large discrepancy at alt {alt}: "
            f"mnl={share_mnl:.4f}, explicit={share_explicit:.4f}, diff={diff:.4f}"
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


def test_interaction_sample_parity_peaked_utilities(state):
    # Stress parity under a highly peaked utility profile:
    # one dominant alternative, one secondary, and many tiny utilities.
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

    # Run EET path with the same seed.
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
    )

    def weighted_shares(df: pd.DataFrame) -> pd.Series:
        counts = df.groupby("alt_id")["pick_count"].sum()
        return (counts / counts.sum()).sort_index()

    mnl_shares = weighted_shares(choices_mnl)
    explicit_shares = weighted_shares(choices_explicit)

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


def test_make_sample_choices_utility_based_fused_rng_matches_materialized_path():
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
    retry_pass = np.where(retry_draw < inclusion_probs[first_pass_empty], inclusion_probs[first_pass_empty], np.nan)
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
    )

    expected = pd.DataFrame(
        {
            "person_id": [301, 301, 302, 302],
            "prob": [1.0, 1.0, 1.0, 1.0],
            "alt_id": [12, 14, 10, 12],
        }
    )

    pd.testing.assert_frame_equal(out.reset_index(drop=True), expected)
