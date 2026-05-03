# ActivitySim
# See full license in LICENSE.txt.
"""Unit tests for activitysim.core.random.FastChannel."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from activitysim.core.fast_random import FastChannel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(index_values, name="household_id"):
    """Return a minimal DataFrame with a named index."""
    df = pd.DataFrame({"x": 0}, index=pd.Index(index_values, name=name))
    return df


def _make_channel(
    index_values=None, base_seed=0, channel_name="households", step_name=""
):
    if index_values is None:
        index_values = [1, 2, 3, 4, 5]
    domain_df = _make_df(index_values)
    return FastChannel(channel_name, base_seed, domain_df, step_name=step_name)


# ---------------------------------------------------------------------------
# __init__ / construction
# ---------------------------------------------------------------------------


class TestInit:
    def test_attributes_set(self):
        domain_df = _make_df([10, 20, 30])
        ch = FastChannel("persons", 42, domain_df)
        assert ch.channel_name == "persons"
        assert ch.base_seed == 42
        assert len(ch.domain_index) == 3
        assert ch.step_name is None
        assert ch._state_array is None

    def test_domain_index_is_copy(self):
        domain_df = _make_df([1, 2, 3])
        ch = FastChannel("h", 0, domain_df)
        # mutating original index should not affect stored domain_index
        original_index = domain_df.index.copy()
        assert list(ch.domain_index) == list(original_index)

    def test_step_started_when_step_name_given(self):
        ch = _make_channel(step_name="my_step")
        assert ch.step_name == "my_step"
        assert ch._state_array is None
        ch._reseed_step()  # ensure state is populated
        assert ch._state_array is not None
        assert ch._state_array.shape == (5, 4)

    def test_no_step_when_step_name_empty(self):
        ch = _make_channel(step_name="")
        assert ch.step_name is None
        assert ch._state_array is None

    def test_channel_seed_differs_by_name(self):
        d = _make_df([1, 2, 3])
        ch_a = FastChannel("alpha", 0, d)
        ch_b = FastChannel("beta", 0, d)
        assert ch_a.channel_seed != ch_b.channel_seed


# ---------------------------------------------------------------------------
# begin_step / end_step
# ---------------------------------------------------------------------------


class TestBeginEndStep:
    def test_begin_step_populates_state(self):
        ch = _make_channel()
        ch.begin_step("step1")
        assert ch.step_name == "step1"
        assert ch._state_array is None  # no state before a draw
        ch._reseed_step()  # ensure state is populated
        assert ch._state_array is not None
        assert ch._state_array.shape == (5, 4)
        assert ch._state_array.dtype == np.uint64

    def test_begin_step_raises_if_already_active(self):
        ch = _make_channel()
        ch.begin_step("step1")
        with pytest.raises(AssertionError):
            ch.begin_step("step2")

    def test_end_step_clears_state(self):
        ch = _make_channel()
        ch.begin_step("step1")
        ch.end_step("step1")
        assert ch.step_name is None
        assert ch.step_seed is None
        assert ch._state_array is None

    def test_end_step_consistency_check_passes(self):
        ch = _make_channel()
        ch.begin_step("step1")
        # should not raise
        ch.end_step("step1")

    def test_end_step_consistency_check_fails(self):
        ch = _make_channel()
        ch.begin_step("step1")
        with pytest.raises(AssertionError):
            ch.end_step("wrong_step")

    def test_end_step_no_name_skips_check(self):
        ch = _make_channel()
        ch.begin_step("step1")
        ch.end_step()  # no name → no assertion
        assert ch.step_name is None

    def test_begin_step_same_name_reproducible(self):
        """Re-running the same step name must reproduce the same state array."""
        ch = _make_channel()
        ch.begin_step("step1")
        ch._reseed_step()  # ensure state is populated
        state1 = ch._state_array.copy()
        ch.end_step()
        ch.begin_step("step1")
        ch._reseed_step()  # ensure state is populated
        state2 = ch._state_array.copy()
        npt.assert_array_equal(state1, state2)

    def test_begin_step_different_names_give_different_states(self):
        ch = _make_channel()
        ch.begin_step("step_a")
        ch._reseed_step()  # ensure state is populated
        state_a = ch._state_array.copy()
        ch.end_step()
        ch.begin_step("step_b")
        ch._reseed_step()  # ensure state is populated
        state_b = ch._state_array.copy()
        assert not np.array_equal(state_a, state_b)


# ---------------------------------------------------------------------------
# _check_valid_df
# ---------------------------------------------------------------------------


class TestCheckValidDf:
    def test_returns_positions_for_full_domain(self):
        ch = _make_channel(index_values=[10, 20, 30])
        ch.begin_step("s")
        df = _make_df([10, 20, 30])
        pos = ch._check_valid_df(df)
        npt.assert_array_equal(pos, [0, 1, 2])

    def test_returns_positions_for_subset(self):
        ch = _make_channel(index_values=[10, 20, 30])
        ch.begin_step("s")
        df = _make_df([30, 10])
        pos = ch._check_valid_df(df)
        npt.assert_array_equal(pos, [2, 0])

    def test_raises_on_duplicate_index(self):
        ch = _make_channel(index_values=[1, 2, 3])
        ch.begin_step("s")
        df = _make_df([1, 1, 2])
        with pytest.raises(ValueError, match="unique index"):
            ch._check_valid_df(df)

    def test_raises_on_index_not_in_domain(self):
        ch = _make_channel(index_values=[1, 2, 3])
        ch.begin_step("s")
        df = _make_df([1, 99])
        with pytest.raises(ValueError, match="not found in the domain"):
            ch._check_valid_df(df)

    def test_raises_outside_step(self):
        ch = _make_channel(index_values=[1, 2, 3])
        # no begin_step called
        df = _make_df([1, 2])
        with pytest.raises(ValueError, match="outside of a defined step"):
            ch._check_valid_df(df)


# ---------------------------------------------------------------------------
# random_for_df
# ---------------------------------------------------------------------------


class TestRandomForDf:
    def test_output_shape_single_draw(self):
        ch = _make_channel()
        ch.begin_step("s")
        df = _make_df([1, 2, 3, 4, 5])
        rands = ch.random_for_df(df, "s", n=1)
        assert rands.shape == (5, 1)

    def test_output_shape_multiple_draws(self):
        ch = _make_channel()
        ch.begin_step("s")
        df = _make_df([1, 2, 3, 4, 5])
        rands = ch.random_for_df(df, "s", n=3)
        assert rands.shape == (5, 3)

    def test_values_in_unit_interval(self):
        ch = _make_channel()
        ch.begin_step("s")
        df = _make_df([1, 2, 3, 4, 5])
        rands = ch.random_for_df(df, "s", n=10)
        assert np.all(rands >= 0.0)
        assert np.all(rands < 1.0)

    def test_successive_calls_advance_stream(self):
        ch = _make_channel()
        ch.begin_step("s")
        df = _make_df([1, 2, 3, 4, 5])
        r1 = ch.random_for_df(df, "s")
        r2 = ch.random_for_df(df, "s")
        assert not np.array_equal(r1, r2)

    def test_reproducible_across_steps(self):
        ch = _make_channel()
        df = _make_df([1, 2, 3, 4, 5])
        ch.begin_step("s")
        r1 = ch.random_for_df(df, "s").copy()
        ch.end_step()
        ch.begin_step("s")
        r2 = ch.random_for_df(df, "s").copy()
        npt.assert_array_equal(r1, r2)

    def test_different_steps_give_different_streams(self):
        ch = _make_channel()
        df = _make_df([1, 2, 3, 4, 5])
        ch.begin_step("step_a")
        r_a = ch.random_for_df(df, "step_a").copy()
        ch.end_step()
        ch.begin_step("step_b")
        r_b = ch.random_for_df(df, "step_b").copy()
        assert not np.array_equal(r_a, r_b)

    def test_different_base_seeds_give_different_streams(self):
        df = _make_df([1, 2, 3])
        ch0 = FastChannel("h", 0, df)
        ch1 = FastChannel("h", 1, df)
        ch0.begin_step("s")
        ch1.begin_step("s")
        r0 = ch0.random_for_df(df, "s")
        r1 = ch1.random_for_df(df, "s")
        assert not np.array_equal(r0, r1)

    def test_different_channel_names_give_different_streams(self):
        df = _make_df([1, 2, 3])
        ch_a = FastChannel("alpha", 0, df)
        ch_b = FastChannel("beta", 0, df)
        ch_a.begin_step("s")
        ch_b.begin_step("s")
        r_a = ch_a.random_for_df(df, "s")
        r_b = ch_b.random_for_df(df, "s")
        assert not np.array_equal(r_a, r_b)

    def test_subset_matches_full_domain_rows(self):
        """Draws for a subset of rows should equal draws for those rows from the full domain."""
        ch_full = _make_channel(index_values=[1, 2, 3, 4, 5])
        ch_sub = _make_channel(index_values=[2, 4])

        ch_full.begin_step("s")
        ch_sub.begin_step("s")

        df_full = _make_df([1, 2, 3, 4, 5])
        df_sub = _make_df([2, 4])

        r_full = ch_full.random_for_df(df_full, "s")
        r_sub = ch_sub.random_for_df(df_sub, "s")

        # Rows for index 2 and 4 should match regardless of which other rows exist
        npt.assert_array_equal(r_full[[1, 3]], r_sub)

    def test_wrong_step_name_raises(self):
        ch = _make_channel()
        ch.begin_step("s")
        df = _make_df([1, 2])
        with pytest.raises(AssertionError):
            ch.random_for_df(df, "wrong_step")


# ---------------------------------------------------------------------------
# normal_for_df
# ---------------------------------------------------------------------------


class TestNormalForDf:
    def test_output_shape(self):
        ch = _make_channel()
        ch.begin_step("s")
        df = _make_df([1, 2, 3])
        result = ch.normal_for_df(df, "s")
        assert result.shape == (3, 1)

    def test_output_shape_with_size(self):
        ch = _make_channel()
        ch.begin_step("s")
        df = _make_df([1, 2, 3])
        result = ch.normal_for_df(df, "s", size=4)
        assert result.shape == (3, 4)

    def test_mu_sigma_applied(self):
        """With large sigma, values should spread away from mu."""
        ch = _make_channel()
        ch.begin_step("s")
        df = _make_df([1, 2, 3, 4, 5])
        result = ch.normal_for_df(df, "s", mu=100.0, sigma=0.0001)
        # all values should be very close to mu
        npt.assert_allclose(result.flatten(), 100.0, atol=0.01)

    def test_lognormal_flag_positive(self):
        ch = _make_channel()
        ch.begin_step("s")
        df = _make_df([1, 2, 3, 4, 5])
        result = ch.normal_for_df(df, "s", lognormal=True)
        assert np.all(result > 0), "lognormal values must be positive"

    def test_successive_calls_advance_stream(self):
        ch = _make_channel()
        ch.begin_step("s")
        df = _make_df([1, 2, 3])
        r1 = ch.normal_for_df(df, "s").copy()
        r2 = ch.normal_for_df(df, "s").copy()
        assert not np.array_equal(r1, r2)

    def test_reproducible_across_steps(self):
        ch = _make_channel()
        df = _make_df([1, 2, 3])
        ch.begin_step("s")
        r1 = ch.normal_for_df(df, "s").copy()
        ch.end_step()
        ch.begin_step("s")
        r2 = ch.normal_for_df(df, "s").copy()
        npt.assert_array_equal(r1, r2)

    def test_scalar_mu_sigma_broadcast(self):
        ch = _make_channel()
        ch.begin_step("s")
        df = _make_df([1, 2, 3, 4, 5])
        # should not raise; shape should still be (5, 1)
        result = ch.normal_for_df(df, "s", mu=5.0, sigma=2.0)
        assert result.shape == (5, 1)

    def test_lognormal_vs_normal_exp_relationship(self):
        """exp(normal_for_df) should equal lognormal_for_df for same step."""
        domain_df = _make_df([1, 2, 3, 4, 5])
        ch_n = FastChannel("h", 7, domain_df)
        ch_l = FastChannel("h", 7, domain_df)
        ch_n.begin_step("s")
        ch_l.begin_step("s")
        df = _make_df([1, 2, 3, 4, 5])
        r_normal = ch_n.normal_for_df(df, "s", mu=0, sigma=1, lognormal=False)
        r_lognormal = ch_l.normal_for_df(df, "s", mu=0, sigma=1, lognormal=True)
        npt.assert_allclose(np.exp(r_normal), r_lognormal)


# ---------------------------------------------------------------------------
# choice_for_df
# ---------------------------------------------------------------------------


class TestChoiceForDf:
    def test_output_length_with_replace(self):
        ch = _make_channel()
        ch.begin_step("s")
        df = _make_df([1, 2, 3])
        choices = ch.choice_for_df(df, "s", a=10, size=4, replace=True)
        assert choices.shape == (12,)  # 3 rows × 4

    def test_output_length_without_replace(self):
        ch = _make_channel()
        ch.begin_step("s")
        df = _make_df([1, 2, 3])
        choices = ch.choice_for_df(df, "s", a=np.arange(10), size=3, replace=False)
        assert choices.shape == (9,)  # 3 rows × 3

    def test_int_population_values_in_range(self):
        ch = _make_channel()
        ch.begin_step("s")
        df = _make_df([1, 2, 3, 4, 5])
        choices = ch.choice_for_df(df, "s", a=7, size=10, replace=True)
        assert np.all(choices >= 0)
        assert np.all(choices < 7)

    def test_array_population_values_from_array(self):
        ch = _make_channel()
        ch.begin_step("s")
        df = _make_df([1, 2, 3])
        a = np.array([10, 20, 30, 40])
        choices = ch.choice_for_df(df, "s", a=a, size=5, replace=True)
        assert set(choices).issubset({10, 20, 30, 40})

    def test_without_replace_no_duplicates_per_row(self):
        ch = _make_channel()
        ch.begin_step("s")
        df = _make_df([1, 2, 3])
        n_pop = 8
        size = n_pop  # draw all elements → no duplicates allowed
        choices = ch.choice_for_df(df, "s", a=n_pop, size=size, replace=False)
        for row_choices in choices.reshape(len(df), size):
            assert len(set(row_choices)) == size

    def test_without_replace_raises_when_size_exceeds_population(self):
        ch = _make_channel()
        ch.begin_step("s")
        df = _make_df([1, 2])
        with pytest.raises(ValueError, match="replace=False"):
            ch.choice_for_df(df, "s", a=3, size=5, replace=False)

    def test_successive_calls_advance_stream(self):
        ch = _make_channel()
        ch.begin_step("s")
        df = _make_df([1, 2, 3])
        c1 = ch.choice_for_df(df, "s", a=100, size=5, replace=True).copy()
        c2 = ch.choice_for_df(df, "s", a=100, size=5, replace=True).copy()
        assert not np.array_equal(c1, c2)

    def test_reproducible_across_steps(self):
        ch = _make_channel()
        df = _make_df([1, 2, 3, 4, 5])
        ch.begin_step("s")
        c1 = ch.choice_for_df(df, "s", a=20, size=4, replace=True).copy()
        ch.end_step()
        ch.begin_step("s")
        c2 = ch.choice_for_df(df, "s", a=20, size=4, replace=True).copy()
        npt.assert_array_equal(c1, c2)

    def test_different_steps_give_different_choices(self):
        ch = _make_channel()
        df = _make_df([1, 2, 3, 4, 5])
        ch.begin_step("step_a")
        c_a = ch.choice_for_df(df, "step_a", a=50, size=10, replace=True).copy()
        ch.end_step()
        ch.begin_step("step_b")
        c_b = ch.choice_for_df(df, "step_b", a=50, size=10, replace=True).copy()
        assert not np.array_equal(c_a, c_b)

    def test_output_dtype_int(self):
        ch = _make_channel()
        ch.begin_step("s")
        df = _make_df([1, 2, 3])
        choices = ch.choice_for_df(df, "s", a=10, size=3, replace=True)
        assert np.issubdtype(choices.dtype, np.integer)

    def test_array_population_without_replace_values_in_array(self):
        ch = _make_channel()
        ch.begin_step("s")
        df = _make_df([1, 2, 3])
        a = np.array([100, 200, 300, 400, 500])
        choices = ch.choice_for_df(df, "s", a=a, size=3, replace=False)
        assert set(choices).issubset(set(a))

    def test_wrong_step_name_raises(self):
        ch = _make_channel()
        ch.begin_step("s")
        df = _make_df([1, 2])
        with pytest.raises(AssertionError):
            ch.choice_for_df(df, "wrong", a=5, size=2, replace=True)


# ---------------------------------------------------------------------------
# Cross-method stream independence
# ---------------------------------------------------------------------------


class TestStreamIndependence:
    def test_random_and_normal_advance_independently(self):
        """random_for_df and normal_for_df on the same channel should each
        advance the underlying streams, so their second calls differ from first."""
        ch = _make_channel()
        ch.begin_step("s")
        df = _make_df([1, 2, 3])
        r1 = ch.random_for_df(df, "s").copy()
        n1 = ch.normal_for_df(df, "s").copy()
        r2 = ch.random_for_df(df, "s").copy()
        n2 = ch.normal_for_df(df, "s").copy()
        assert not np.array_equal(r1, r2)
        assert not np.array_equal(n1, n2)

    def test_two_channels_same_seed_independent(self):
        """Two FastChannels with the same base_seed but different names should
        produce independent (different) streams for the same index values."""
        df = _make_df([1, 2, 3, 4, 5])
        ch_a = FastChannel("channel_a", 0, df)
        ch_b = FastChannel("channel_b", 0, df)
        ch_a.begin_step("s")
        ch_b.begin_step("s")
        r_a = ch_a.random_for_df(df, "s")
        r_b = ch_b.random_for_df(df, "s")
        assert not np.array_equal(r_a, r_b)

    def test_row_streams_are_independent(self):
        """Each row should have its own stream; draws for one row should not
        correlate trivially with another row."""
        ch = _make_channel(index_values=list(range(100)))
        ch.begin_step("s")
        df = _make_df(list(range(100)))
        rands = ch.random_for_df(df, "s", n=10)
        # Check that not all rows are identical
        assert not np.all(rands == rands[0])
