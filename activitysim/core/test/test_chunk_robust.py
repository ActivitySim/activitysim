# ActivitySim
# See full license in LICENSE.txt.
"""Tests for the cgroup-aware ``chunk_size_mode: auto`` budget and the adaptive sizing it feeds.
See also test_mem.py for the cgroup / worker-count helpers."""
from __future__ import annotations

import os

import pandas as pd
import pandas.testing as pdt
import pytest

from activitysim.core import chunk, mem, simulate, workflow

TESTDIR = os.path.dirname(__file__)
DATADIR = os.path.join(TESTDIR, "data")

GIB = 1024**3


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
    state.settings.chunk_size_mode = "auto"
    state.settings.chunk_size_safety_factor = 0.75
    limit = mem.get_memory_limit()
    budget = chunk.resolve_chunk_size(state)
    # Budget = safety_factor * AVAILABLE memory (limit - current usage): strictly positive, and never
    # above safety_factor * the real ceiling (available <= limit).
    assert budget >= 1
    assert budget <= max(1, int(limit * 0.75))


def test_resolve_chunk_size_auto_safety_factor_scales(state):
    # a smaller safety_factor yields a smaller (or equal) budget
    state.settings.chunk_size = 0
    state.settings.chunk_size_mode = "auto"
    state.settings.chunk_size_safety_factor = 0.75
    hi = chunk.resolve_chunk_size(state)
    state.settings.chunk_size_safety_factor = 0.25
    lo = chunk.resolve_chunk_size(state)
    assert lo <= hi


def test_resolve_chunk_size_auto_zero_headroom_is_not_full_limit(state, monkeypatch):
    # available == 0 legitimately means "no headroom" and must NOT be treated as "unknown" and replaced
    # with the full limit (that would hand out a huge budget exactly when memory is exhausted).
    state.settings.chunk_size = 0
    state.settings.chunk_size_mode = "auto"
    monkeypatch.setattr(mem, "get_memory_limit", lambda *a, **k: 100 * GIB)
    monkeypatch.setattr(mem, "get_available_memory", lambda *a, **k: 0)
    budget = chunk.resolve_chunk_size(state)
    assert budget < GIB  # tiny (floored to keep chunking on), NOT ~75 GB


def test_resolve_chunk_size_auto_divides_by_workers(state, monkeypatch):
    # multiprocess: the shared budget is divided by the per-step worker count (num_processes injectable)
    state.settings.chunk_size = 0
    state.settings.chunk_size_mode = "auto"
    state.settings.chunk_size_safety_factor = 1.0
    state.settings.multiprocess = True
    monkeypatch.setattr(mem, "get_memory_limit", lambda *a, **k: 40 * GIB)
    monkeypatch.setattr(mem, "get_available_memory", lambda *a, **k: 40 * GIB)
    state.add_injectable("num_processes", 4)
    budget = chunk.resolve_chunk_size(state)
    assert budget == 10 * GIB  # 40 GB / 4 workers


def test_auto_mode_simple_simulate_matches_fixed(state, data, spec):
    # auto mode must produce the same choices as the legacy path
    state.settings.chunk_size = 0
    state.settings.chunk_size_mode = "auto"
    state.settings.chunk_growth_cap = 2.0
    choices = simulate.simple_simulate(state, choosers=data, spec=spec, nest_spec=None)
    pdt.assert_series_equal(choices.reset_index(drop=True), EXPECTED, check_dtype=False)


def test_auto_mode_splits_into_multiple_chunks(state, data, monkeypatch):
    # Force a tiny auto budget so the choosers are split into MULTIPLE chunks (not a single-chunk run),
    # exercising the auto chunk-sizing loop. Assert the chunks partition the choosers exactly.
    state.settings.chunk_size = 0
    state.settings.chunk_size_mode = "auto"
    state.settings.chunk_training_mode = "training"
    state.settings.default_initial_rows_per_chunk = 1  # tiny first (probe) chunk
    monkeypatch.setattr(mem, "get_memory_limit", lambda *a, **k: 1000)
    monkeypatch.setattr(mem, "get_available_memory", lambda *a, **k: 1)

    chunks = [
        chooser_chunk.copy()
        for _i, chooser_chunk, _label, _sizer in chunk.adaptive_chunked_choosers(
            state, data, "test_auto_multichunk"
        )
    ]
    assert len(chunks) > 1  # the tiny budget forced more than one chunk
    # chunks partition the original choosers exactly (rows + order preserved, none lost/duplicated)
    pdt.assert_frame_equal(pd.concat(chunks), data)


def test_chunk_memory_settings_validation():
    # the auto-mode knobs reject nonsensical values at configuration time
    from pydantic import ValidationError

    from activitysim.core.configuration.top import Settings

    Settings(chunk_size_safety_factor=0.5)  # in (0, 1] — ok
    Settings(chunk_growth_cap=1.5)  # off (0) or >= 1 — ok
    Settings(chunk_row_size_margin=1.3)  # >= 1 — ok
    with pytest.raises(ValidationError):
        Settings(chunk_size_safety_factor=0.0)
    with pytest.raises(ValidationError):
        Settings(chunk_size_safety_factor=1.5)
    with pytest.raises(ValidationError):
        Settings(chunk_growth_cap=0.5)  # would shrink every chunk toward collapse
    with pytest.raises(ValidationError):
        Settings(chunk_row_size_margin=0.9)
    Settings(chunk_peak_backoff_ratio=0.8)  # in (0, 1] — ok
    with pytest.raises(ValidationError):
        Settings(chunk_peak_backoff_ratio=0.0)
    with pytest.raises(ValidationError):
        Settings(chunk_peak_backoff_ratio=1.5)


def test_auto_growth_cap_default():
    # auto mode caps growth by default; fixed keeps legacy uncapped; explicit setting wins
    from activitysim.core.configuration.top import Settings

    assert chunk._effective_growth_cap(Settings(chunk_size_mode="fixed")) == 0
    assert (
        chunk._effective_growth_cap(Settings(chunk_size_mode="auto"))
        == chunk.AUTO_DEFAULT_GROWTH_CAP
    )
    assert (
        chunk._effective_growth_cap(
            Settings(chunk_size_mode="auto", chunk_growth_cap=3.0)
        )
        == 3.0
    )


def test_auto_probe_chunk_is_capped(state, monkeypatch):
    # with no cached row_size, the first (probe) chunk under auto is capped at
    # MAX_AUTO_PROBE_ROWS even when default_initial_rows_per_chunk is huge
    n = chunk.MAX_AUTO_PROBE_ROWS * 3
    data = pd.DataFrame({"x": range(n)})
    state.settings.chunk_size_mode = "auto"
    state.settings.chunk_training_mode = "training"
    state.settings.default_initial_rows_per_chunk = (
        50_000  # deliberately oversized probe
    )
    monkeypatch.setattr(mem, "get_memory_limit", lambda *a, **k: 100 * GIB)
    monkeypatch.setattr(mem, "get_available_memory", lambda *a, **k: 100 * GIB)

    sizes = [
        len(chooser_chunk)
        for _i, chooser_chunk, _label, _sizer in chunk.adaptive_chunked_choosers(
            state, data, "test_auto_probe_cap"
        )
    ]
    assert sizes[0] <= chunk.MAX_AUTO_PROBE_ROWS


def test_chunking_settings_logged_once(state, caplog):
    # the audit line contains every effective chunking parameter, once per process
    import logging

    chunk._CHUNK_SETTINGS_LOGGED = False
    state.settings.chunk_size_mode = "auto"
    with caplog.at_level(logging.INFO, logger="activitysim.core.chunk"):
        chunk.log_chunking_settings(state)
        chunk.log_chunking_settings(state)  # second call must be a no-op
    msgs = [r.message for r in caplog.records if "chunking settings:" in r.message]
    assert len(msgs) == 1
    for key in (
        "chunk_size_mode=auto",
        "chunk_size=",
        "chunk_size_safety_factor=",
        "chunk_growth_cap=",
        "(effective=",
        "chunk_row_size_margin=",
        "chunk_training_mode=",
        "chunk_method=",
        "default_initial_rows_per_chunk=",
        "auto probe cap=",
        "num_processes=",
        "chunk_peak_backoff_ratio=",
    ):
        assert key in msgs[0], key
    chunk._CHUNK_SETTINGS_LOGGED = False  # don't leak state to other tests


def test_auto_overrides_passed_static_chunk_size(state, monkeypatch):
    # callers historically pass settings.chunk_size down explicitly; under auto the
    # runtime-resolved budget must win or parts of the pipeline silently run static
    data = pd.DataFrame({"x": range(100)})
    state.settings.chunk_size_mode = "auto"
    state.settings.chunk_training_mode = "training"
    monkeypatch.setattr(mem, "get_memory_limit", lambda *a, **k: 10 * GIB)
    monkeypatch.setattr(mem, "get_available_memory", lambda *a, **k: 10 * GIB)
    budgets = [
        getattr(sizer, "base_chunk_size", None) or sizer.chunk_size
        for _i, _c, _label, sizer in chunk.adaptive_chunked_choosers(
            state, data, "test_auto_override", chunk_size=999 * GIB
        )
    ]
    assert budgets and all(
        b <= 5 * GIB for b in budgets
    )  # resolved from the 10 GiB limit, not the passed 999 GiB

    # chunk_size == 0 is the deliberate "run this component chunkless" signal — auto must
    # NOT override it (tour scheduling logsums relies on this)
    chunks = [
        c
        for _i, c, _label, _sizer in chunk.adaptive_chunked_choosers(
            state, data, "test_auto_keeps_chunkless", chunk_size=0
        )
    ]
    assert len(chunks) == 1 and len(chunks[0]) == len(data)
