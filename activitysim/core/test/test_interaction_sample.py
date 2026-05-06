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


def test_make_sample_choices_utility_based_repeat_alignment_chooser_dominant_heterogeneity():
    # Edge case: utilities are close across alternatives but vary strongly by chooser.
    # This is where wrong chooser/sample alignment can hide in aggregate checks.
    chooser_index = pd.Index([101, 102, 103, 104, 105, 106], name="person_id")
    choosers = pd.DataFrame(index=chooser_index)
    alternatives = pd.DataFrame(index=pd.Index([0, 1, 2, 3], name="alt_id"))

    n_choosers = len(choosers)
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

    # No random noise: chosen alternative is deterministic argmax of utilities.
    rands_3d = np.zeros((n_choosers, n_alts, sample_size), dtype=np.float64)
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
        trace_label="test_repeat_alignment_chooser_heterogeneity",
        chunk_sizer=_DummyChunkSizer(),
    )

    # Reconstruct expected indexing behavior.
    chosen_2d = np.argmax(
        rands_3d + utilities.to_numpy()[:, :, np.newaxis],
        axis=1,
    )
    chosen_flat = chosen_2d.reshape(-1)

    chooser_repeat = np.repeat(np.arange(n_choosers), sample_size)
    chooser_tile = np.tile(np.arange(n_choosers), sample_size)

    probs = interaction_sample.logit.utils_to_probs(
        state,
        utilities,
        allow_zero_probs=False,
        trace_label="test_repeat_alignment_chooser_heterogeneity",
        overflow_protection=True,
        trace_choosers=choosers,
    ).to_numpy()

    expected_prob_repeat = probs[chooser_repeat, chosen_flat]
    wrong_prob_tile = probs[chooser_tile, chosen_flat]

    assert np.array_equal(out["prob"].to_numpy(), expected_prob_repeat)
    assert not np.array_equal(out["prob"].to_numpy(), wrong_prob_tile)


def test_make_sample_choices_utility_based_fused_rng_matches_materialized_path():
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
        trace_label="test_fused_rng_matches_materialized",
        chunk_sizer=_DummyChunkSizer(),
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
        trace_label="test_fused_rng_matches_materialized",
        overflow_protection=True,
        trace_choosers=choosers,
    ).to_numpy()

    expected = pd.DataFrame(
        {
            "alt_id": alternatives.index.values[chosen_flat],
            "prob": probs[chooser_idx, chosen_flat],
            "person_id": choosers.index.values[chooser_idx],
        }
    )

    pd.testing.assert_frame_equal(out.reset_index(drop=True), expected)


def test_make_sample_choices_utility_based_stable_alt_mapping_matches_materialized_path():
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
        trace_label="test_stable_alt_mapping",
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
    probs = interaction_sample.logit.utils_to_probs(
        state,
        utilities,
        allow_zero_probs=False,
        trace_label="test_stable_alt_mapping",
        overflow_protection=True,
        trace_choosers=choosers,
    ).to_numpy()

    expected = pd.DataFrame(
        {
            "alt_id": alternatives.index.values[chosen_flat],
            "prob": probs[chooser_idx, chosen_flat],
            "person_id": choosers.index.values[chooser_idx],
        }
    )

    pd.testing.assert_frame_equal(out.reset_index(drop=True), expected)
