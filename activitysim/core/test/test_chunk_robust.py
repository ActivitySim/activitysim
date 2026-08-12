# ActivitySim
# See full license in LICENSE.txt.
"""Tests for robust adaptive chunking: the real-memory-ceiling budget (chunk_memory_mode=auto), the
AIMD growth cap / back-off, and the memory watchdog. See also test_mem.py for the cgroup helpers."""
from __future__ import annotations

import logging
import os

import pandas as pd
import pandas.testing as pdt
import pytest

from activitysim.core import chunk, mem, simulate, workflow

TESTDIR = os.path.dirname(__file__)
DATADIR = os.path.join(TESTDIR, "data")


@pytest.fixture
def state() -> workflow.State:
    st = workflow.State()
    st.initialize_filesystem(
        working_dir=TESTDIR, data_dir=(DATADIR,)
    ).default_settings()
    st.settings.check_for_variability = False
    return st


@pytest.fixture
def spec(state):
    return state.filesystem.read_model_spec(file_name="sample_spec.csv")


@pytest.fixture
def data():
    return pd.read_csv(os.path.join(DATADIR, "data.csv"))


EXPECTED = pd.Series([1, 1, 1])


def test_resolve_chunk_size_fixed_is_legacy(state):
    # default (fixed) mode must return the static chunk_size verbatim -> no behavior change
    state.settings.chunk_size = 123456
    assert chunk.resolve_chunk_size(state) == 123456


def test_resolve_chunk_size_auto(state):
    state.settings.chunk_size = 0
    state.settings.chunk_memory_mode = "auto"
    state.settings.chunk_memory_safety_factor = 0.75
    limit = mem.get_memory_limit()
    budget = chunk.resolve_chunk_size(state)
    # Budget = safety_factor * AVAILABLE memory (limit - current usage), kept strictly positive by the
    # floor and never above safety_factor * the real ceiling. (Available <= limit, so budget <= 0.75*limit.)
    assert budget >= chunk.AUTO_BUDGET_FLOOR
    assert 0 < budget <= max(chunk.AUTO_BUDGET_FLOOR, int(limit * 0.75))


def test_resolve_chunk_size_auto_safety_factor_scales(state):
    # a smaller safety_factor yields a smaller (or equal, if floored) budget
    state.settings.chunk_size = 0
    state.settings.chunk_memory_mode = "auto"
    state.settings.chunk_memory_safety_factor = 0.75
    hi = chunk.resolve_chunk_size(state)
    state.settings.chunk_memory_safety_factor = 0.25
    lo = chunk.resolve_chunk_size(state)
    assert lo <= hi


def test_auto_mode_simple_simulate_matches_fixed(state, data, spec):
    # auto mode must produce the same choices as the legacy path
    state.settings.chunk_size = 0
    state.settings.chunk_memory_mode = "auto"
    state.settings.chunk_growth_cap = 2.0
    choices = simulate.simple_simulate(state, choosers=data, spec=spec, nest_spec=None)
    pdt.assert_series_equal(choices.reset_index(drop=True), EXPECTED, check_dtype=False)


def test_auto_mode_splits_into_multiple_chunks(state, data, monkeypatch):
    # Force a tiny auto budget so the choosers are split into MULTIPLE chunks (not a single-chunk run),
    # exercising the auto chunk-sizing loop. Assert the chunks partition the choosers exactly.
    state.settings.chunk_size = 0
    state.settings.chunk_memory_mode = "auto"
    state.settings.chunk_training_mode = "training"
    state.settings.default_initial_rows_per_chunk = 1  # tiny first (probe) chunk
    monkeypatch.setattr(chunk, "AUTO_BUDGET_FLOOR", 1)
    monkeypatch.setattr(mem, "get_memory_limit", lambda *a, **k: 1)
    monkeypatch.setattr(mem, "get_nonreclaimable_used", lambda *a, **k: 0)

    chunks = [
        chooser_chunk.copy()
        for _i, chooser_chunk, _label, _sizer in chunk.adaptive_chunked_choosers(
            state, data, "test_auto_multichunk"
        )
    ]
    assert len(chunks) > 1  # the tiny budget forced more than one chunk
    # chunks partition the original choosers exactly (rows + order preserved, none lost/duplicated)
    pdt.assert_frame_equal(pd.concat(chunks), data)


def test_watchdog_warns_only_on_breach(state, caplog):
    state.settings.chunk_memory_circuit_breaker = True
    state.settings.chunk_memory_abort_ratio = 0.9
    limit = mem.get_memory_limit()
    chunk._watchdog_breached = False
    with caplog.at_level(logging.WARNING, logger="activitysim.core.chunk"):
        chunk.memory_watchdog_check(state, int(0.5 * limit), "below")
        assert not any("WATCHDOG" in r.message for r in caplog.records)
        chunk.memory_watchdog_check(state, int(0.95 * limit), "over")
        assert any("WATCHDOG" in r.message for r in caplog.records)


def test_watchdog_disabled_by_default(state, caplog):
    # circuit breaker off (default) -> never warns even above the ratio
    chunk._watchdog_breached = False
    with caplog.at_level(logging.WARNING, logger="activitysim.core.chunk"):
        chunk.memory_watchdog_check(state, int(0.99 * mem.get_memory_limit()), "over")
        assert not any("WATCHDOG" in r.message for r in caplog.records)
