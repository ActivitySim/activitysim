"""Tests for vector_random_standard_normal and vector_random_standard_uniform."""

from __future__ import annotations

import numpy as np

from activitysim.core.fast_random._fast_random import FastGenerator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_fast_generator():
    return FastGenerator()


def make_state_array(
    fg: FastGenerator, n_agents: int, base_seed: int = 0
) -> np.ndarray:
    """Return a (n_agents, 4) uint64 state array seeded from PCG64 generators.

    Each row is a valid PCG64 state obtained by seeding a fresh generator with
    ``base_seed + i`` and reading the current raw CFFI state buffer.  This
    mirrors exactly the layout consumed by the private ``_several_random_*``
    functions.
    """
    uint64_view = fg._state_bytes.view(np.uint64)
    state = np.zeros((n_agents, 4), dtype=np.uint64)
    for i in range(n_agents):
        fg._bit_generator.state = np.random.PCG64(seed=base_seed + i).state
        state[i] = uint64_view[fg._slice_start : fg._slice_end].copy()
    return state


# ---------------------------------------------------------------------------
# vector_random_standard_uniform
# ---------------------------------------------------------------------------


class TestVectorRandomStandardUniform:
    """Tests for vector_random_standard_uniform."""

    # --- output dtype and basic shape ---

    def test_returns_float64(self):
        fg = make_fast_generator()
        state = make_state_array(fg, 5)
        result = fg.vector_random_standard_uniform(state)
        assert result.dtype == np.float64

    def test_default_shape_all_agents(self):
        n = 7
        fg = make_fast_generator()
        state = make_state_array(fg, n)
        result = fg.vector_random_standard_uniform(state)
        assert result.shape == (n, 1)

    def test_int_shape(self):
        n = 6
        fg = make_fast_generator()
        state = make_state_array(fg, n)
        result = fg.vector_random_standard_uniform(state, shape=4)
        assert result.shape == (n, 4)

    def test_tuple_shape_1d(self):
        n = 5
        fg = make_fast_generator()
        state = make_state_array(fg, n)
        result = fg.vector_random_standard_uniform(state, shape=(3,))
        assert result.shape == (n, 3)

    def test_tuple_shape_2d(self):
        n = 4
        fg = make_fast_generator()
        state = make_state_array(fg, n)
        result = fg.vector_random_standard_uniform(state, shape=(2, 3))
        assert result.shape == (n, 2, 3)

    # --- values in [0, 1) ---

    def test_values_in_unit_interval(self):
        fg = make_fast_generator()
        state = make_state_array(fg, 50)
        result = fg.vector_random_standard_uniform(state, shape=100)
        assert np.all(result >= 0.0)
        assert np.all(result < 1.0)

    # --- selected_positions ---

    def test_selected_positions_shape(self):
        n = 10
        fg = make_fast_generator()
        state = make_state_array(fg, n)
        sel = np.array([0, 2, 5], dtype=np.intp)
        result = fg.vector_random_standard_uniform(
            state, selected_positions=sel, shape=4
        )
        assert result.shape == (3, 4)

    def test_selected_positions_values_in_unit_interval(self):
        n = 10
        fg = make_fast_generator()
        state = make_state_array(fg, n)
        sel = np.array([1, 4, 7], dtype=np.intp)
        result = fg.vector_random_standard_uniform(
            state, selected_positions=sel, shape=20
        )
        assert np.all(result >= 0.0)
        assert np.all(result < 1.0)

    def test_selected_positions_only_selected_rows_mutated(self):
        n = 5
        fg = make_fast_generator()
        state = make_state_array(fg, n)
        state_copy = state.copy()
        sel = np.array([1, 3], dtype=np.intp)
        fg.vector_random_standard_uniform(state, selected_positions=sel, shape=1)
        # Rows NOT in sel must be unchanged.
        for i in range(n):
            if i not in sel:
                np.testing.assert_array_equal(
                    state[i],
                    state_copy[i],
                    err_msg=f"Row {i} should not have been mutated",
                )
        # Rows IN sel must have changed.
        for i in sel:
            assert not np.array_equal(
                state[i], state_copy[i]
            ), f"Row {i} should have been mutated"

    # --- state mutation ---

    def test_state_mutated_in_place(self):
        fg = make_fast_generator()
        state = make_state_array(fg, 4)
        state_before = state.copy()
        fg.vector_random_standard_uniform(state, shape=3)
        assert not np.array_equal(state, state_before)

    # --- reproducibility ---

    def test_reproducibility(self):
        """Identical initial state must produce identical output."""
        fg = make_fast_generator()
        state_a = make_state_array(fg, 8, base_seed=99)
        state_b = state_a.copy()
        out_a = fg.vector_random_standard_uniform(state_a, shape=10)
        out_b = fg.vector_random_standard_uniform(state_b, shape=10)
        np.testing.assert_array_equal(out_a, out_b)

    def test_different_seeds_differ(self):
        fg = make_fast_generator()
        state_a = make_state_array(fg, 5, base_seed=0)
        state_b = make_state_array(fg, 5, base_seed=1000)
        out_a = fg.vector_random_standard_uniform(state_a, shape=20)
        out_b = fg.vector_random_standard_uniform(state_b, shape=20)
        assert not np.array_equal(out_a, out_b)

    # --- statistical sanity (large sample) ---

    def test_mean_close_to_half(self):
        fg = make_fast_generator()
        state = make_state_array(fg, 200)
        result = fg.vector_random_standard_uniform(state, shape=500)
        assert abs(result.mean() - 0.5) < 0.01

    def test_no_exact_ones(self):
        """U[0, 1) must never produce exactly 1.0."""
        fg = make_fast_generator()
        state = make_state_array(fg, 100)
        result = fg.vector_random_standard_uniform(state, shape=1000)
        assert not np.any(result == 1.0)


# ---------------------------------------------------------------------------
# vector_random_standard_normal
# ---------------------------------------------------------------------------


class TestVectorRandomStandardNormal:
    """Tests for vector_random_standard_normal."""

    # --- output dtype and basic shape ---

    def test_returns_float64(self):
        fg = make_fast_generator()
        state = make_state_array(fg, 5)
        result = fg.vector_random_standard_normal(state)
        assert result.dtype == np.float64

    def test_default_shape_all_agents(self):
        n = 7
        fg = make_fast_generator()
        state = make_state_array(fg, n)
        result = fg.vector_random_standard_normal(state)
        assert result.shape == (n, 1)

    def test_int_shape(self):
        n = 6
        fg = make_fast_generator()
        state = make_state_array(fg, n)
        result = fg.vector_random_standard_normal(state, shape=4)
        assert result.shape == (n, 4)

    def test_tuple_shape_1d(self):
        n = 5
        fg = make_fast_generator()
        state = make_state_array(fg, n)
        result = fg.vector_random_standard_normal(state, shape=(3,))
        assert result.shape == (n, 3)

    def test_tuple_shape_2d(self):
        n = 4
        fg = make_fast_generator()
        state = make_state_array(fg, n)
        result = fg.vector_random_standard_normal(state, shape=(2, 3))
        assert result.shape == (n, 2, 3)

    # --- selected_positions ---

    def test_selected_positions_shape(self):
        n = 10
        fg = make_fast_generator()
        state = make_state_array(fg, n)
        sel = np.array([0, 2, 5], dtype=np.intp)
        result = fg.vector_random_standard_normal(
            state, selected_positions=sel, shape=4
        )
        assert result.shape == (3, 4)

    def test_selected_positions_only_selected_rows_mutated(self):
        n = 5
        fg = make_fast_generator()
        state = make_state_array(fg, n)
        state_copy = state.copy()
        sel = np.array([0, 4], dtype=np.intp)
        fg.vector_random_standard_normal(state, selected_positions=sel, shape=1)
        for i in range(n):
            if i not in sel:
                np.testing.assert_array_equal(
                    state[i],
                    state_copy[i],
                    err_msg=f"Row {i} should not have been mutated",
                )
        for i in sel:
            assert not np.array_equal(
                state[i], state_copy[i]
            ), f"Row {i} should have been mutated"

    # --- state mutation ---

    def test_state_mutated_in_place(self):
        fg = make_fast_generator()
        state = make_state_array(fg, 4)
        state_before = state.copy()
        fg.vector_random_standard_normal(state, shape=3)
        assert not np.array_equal(state, state_before)

    # --- reproducibility ---

    def test_reproducibility(self):
        """Identical initial state must produce identical output."""
        fg = make_fast_generator()
        state_a = make_state_array(fg, 8, base_seed=77)
        state_b = state_a.copy()
        out_a = fg.vector_random_standard_normal(state_a, shape=10)
        out_b = fg.vector_random_standard_normal(state_b, shape=10)
        np.testing.assert_array_equal(out_a, out_b)

    def test_different_seeds_differ(self):
        fg = make_fast_generator()
        state_a = make_state_array(fg, 5, base_seed=0)
        state_b = make_state_array(fg, 5, base_seed=1000)
        out_a = fg.vector_random_standard_normal(state_a, shape=20)
        out_b = fg.vector_random_standard_normal(state_b, shape=20)
        assert not np.array_equal(out_a, out_b)

    # --- statistical sanity (large sample) ---

    def test_mean_close_to_zero(self):
        fg = make_fast_generator()
        state = make_state_array(fg, 200)
        result = fg.vector_random_standard_normal(state, shape=500)
        assert abs(result.mean()) < 0.05

    def test_std_close_to_one(self):
        fg = make_fast_generator()
        state = make_state_array(fg, 200)
        result = fg.vector_random_standard_normal(state, shape=500)
        assert abs(result.std() - 1.0) < 0.05

    def test_values_are_finite(self):
        fg = make_fast_generator()
        state = make_state_array(fg, 50)
        result = fg.vector_random_standard_normal(state, shape=200)
        assert np.all(np.isfinite(result))

    def test_distribution_symmetry(self):
        """Mean of absolute values should be close to sqrt(2/pi) ≈ 0.7979."""
        fg = make_fast_generator()
        expected_mean_abs = np.sqrt(2.0 / np.pi)
        state = make_state_array(fg, 200)
        result = fg.vector_random_standard_normal(state, shape=500)
        assert abs(np.abs(result).mean() - expected_mean_abs) < 0.05
