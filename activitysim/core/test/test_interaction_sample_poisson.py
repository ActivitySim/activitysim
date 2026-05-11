# ActivitySim
# See full license in LICENSE.txt.

import numpy as np
import pandas as pd
import pytest

from activitysim.core import interaction_sample, workflow


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


@pytest.fixture
def state() -> workflow.State:
    state = workflow.State().default_settings()
    state.settings.check_for_variability = False
    return state


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
            [expected_inclusion_probs.iloc[1, 0], expected_inclusion_probs.iloc[1, 1], np.nan, np.nan],
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
    state = _DummyState(
        _SequentialDummyRng([fail_draw] * 10 + [fallback_draw])
    )

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


def test_make_sample_choices_utility_based_preserves_sparse_choice_order(
    monkeypatch, state
):
    chooser_index = pd.Index([11, 17, 42], name="person_id")
    choosers = pd.DataFrame(index=chooser_index)
    alternatives = pd.DataFrame(index=pd.Index([100, 300, 700, 900], name="alt_id"))
    utilities = pd.DataFrame(
        [[1.0, 0.0, -1.0, 0.5], [0.1, 0.2, 0.3, 0.4], [1.0, 2.0, 3.0, 4.0]],
        index=chooser_index,
        columns=np.arange(len(alternatives)),
    )

    sampled_alternatives = pd.DataFrame(
        [
            [0.25, np.nan, 0.75, np.nan],
            [np.nan, 0.50, np.nan, 0.20],
            [0.10, np.nan, np.nan, 0.90],
        ],
        index=chooser_index,
        columns=np.arange(len(alternatives)),
    )
    def fake_poisson_sample_alternatives(
        chunk_sizer,
        probs,
        alternatives_arg,
        sample_size,
        alt_col_name,
        state,
        trace_label,
    ):
        assert probs.shape == sampled_alternatives.shape
        assert alternatives_arg.equals(alternatives)
        assert alt_col_name == "alt_id"
        return _expected_choices_df(sampled_alternatives, alternatives, alt_col_name)

    monkeypatch.setattr(
        interaction_sample,
        "_poisson_sample_alternatives",
        fake_poisson_sample_alternatives,
    )

    choices_df = interaction_sample.make_sample_choices_utility_based(
        state=state,
        choosers=choosers,
        utilities=utilities,
        alternatives=alternatives,
        sample_size=3,
        alternative_count=len(alternatives),
        alt_col_name="alt_id",
        allow_zero_probs=False,
        trace_label="test_make_sample_choices_utility_based_preserves_sparse_choice_order",
        chunk_sizer=_DummyChunkSizer(),
    )

    expected_choices_df = _expected_choices_df(
        sampled_alternatives, alternatives, "alt_id"
    )

    pd.testing.assert_frame_equal(choices_df, expected_choices_df)


def test_make_sample_choices_utility_based_retry_path_matches_stubbed_sampler(
    monkeypatch,
):
    chooser_index = pd.Index([11, 17, 42], name="person_id")
    choosers = pd.DataFrame(index=chooser_index)
    alternatives = pd.DataFrame(index=pd.Index([100, 300, 700, 900], name="alt_id"))
    utilities = pd.DataFrame(
        [[1.0, 0.0, -1.0, 0.5], [0.1, 0.2, 0.3, 0.4], [1.0, 2.0, 3.0, 4.0]],
        index=chooser_index,
        columns=np.arange(len(alternatives)),
    )
    probs = pd.DataFrame(
        [
            [0.20, 0.60, 0.10, 0.05],
            [0.40, 0.10, 0.30, 0.20],
            [0.30, 0.20, 0.70, 0.10],
        ],
        index=chooser_index,
        columns=np.arange(len(alternatives)),
    )
    sample_size = 2
    inclusion_probs = 1 - (1 - probs) ** sample_size
    sampled_alternatives = pd.DataFrame(
        [
            [inclusion_probs.iloc[0, 0], np.nan, np.nan, np.nan],
            [inclusion_probs.iloc[1, 0], inclusion_probs.iloc[1, 1], np.nan, np.nan],
            [np.nan, np.nan, inclusion_probs.iloc[2, 2], np.nan],
        ],
        index=chooser_index,
        columns=probs.columns,
    )

    monkeypatch.setattr(
        interaction_sample.logit,
        "utils_to_probs",
        lambda *args, **kwargs: probs,
    )

    state = _DummyState(
        _SequentialDummyRng(
            [
                np.array(
                    [
                        [0.10, 0.90, 0.50, 0.90],
                        [0.90, 0.90, 0.90, 0.90],
                        [0.80, 0.90, 0.20, 0.80],
                    ]
                ),
                np.array([[0.10, 0.05, 0.70, 0.80]]),
            ]
        )
    )

    real_choices_df = interaction_sample.make_sample_choices_utility_based(
        state=state,
        choosers=choosers,
        utilities=utilities,
        alternatives=alternatives,
        sample_size=sample_size,
        alternative_count=len(alternatives),
        alt_col_name="alt_id",
        allow_zero_probs=False,
        trace_label="test_make_sample_choices_utility_based_retry_path_matches_stubbed_sampler",
        chunk_sizer=_DummyChunkSizer(),
    )

    def fake_poisson_sample_alternatives(
        chunk_sizer,
        probs_arg,
        alternatives_arg,
        sample_size_arg,
        alt_col_name,
        state_arg,
        trace_label,
    ):
        assert probs_arg.equals(probs)
        assert alternatives_arg.equals(alternatives)
        assert sample_size_arg == sample_size
        assert alt_col_name == "alt_id"
        return _expected_choices_df(sampled_alternatives, alternatives, alt_col_name)

    monkeypatch.setattr(
        interaction_sample,
        "_poisson_sample_alternatives",
        fake_poisson_sample_alternatives,
    )

    stubbed_choices_df = interaction_sample.make_sample_choices_utility_based(
        state=_DummyState(_SequentialDummyRng([])),
        choosers=choosers,
        utilities=utilities,
        alternatives=alternatives,
        sample_size=sample_size,
        alternative_count=len(alternatives),
        alt_col_name="alt_id",
        allow_zero_probs=False,
        trace_label="test_make_sample_choices_utility_based_retry_path_matches_stubbed_sampler.stub",
        chunk_sizer=_DummyChunkSizer(),
    )

    pd.testing.assert_frame_equal(real_choices_df, stubbed_choices_df)
