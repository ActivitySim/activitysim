# ActivitySim
# See full license in LICENSE.txt.
"""Tests for the cgroup-aware memory-ceiling helpers used by adaptive chunking (chunk_size_mode=auto)."""
from __future__ import annotations

import os

import psutil

from activitysim.core import mem

GIB = 1024**3


def _write(root, rel, text):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def test_finite_limit_parsing():
    assert mem._finite_limit("62000000000") == 62000000000
    assert mem._finite_limit("max") is None
    assert mem._finite_limit(None) is None
    assert mem._finite_limit("garbage") is None
    assert mem._finite_limit("0") is None  # non-positive is not a real limit
    assert mem._finite_limit(str(mem._CGROUP_UNLIMITED)) is None  # unlimited sentinel


def test_memory_limit_cgroup_v2(tmp_path):
    root = str(tmp_path)
    _write(root, "memory.max", "60000000000\n")
    assert mem.get_memory_limit(cgroup_root=root) == 60000000000


def test_memory_limit_cgroup_v2_max_falls_back(tmp_path):
    # cgroup v2 present but unlimited ("max") -> fall through to host RAM (a positive int)
    root = str(tmp_path)
    _write(root, "memory.max", "max\n")
    limit = mem.get_memory_limit(cgroup_root=root)
    assert limit == int(psutil.virtual_memory().total)
    assert limit > 0


def test_memory_limit_cgroup_v1(tmp_path):
    root = str(tmp_path)  # no memory.max -> v1 path
    _write(root, "memory/memory.limit_in_bytes", "48000000000\n")
    assert mem.get_memory_limit(cgroup_root=root) == 48000000000


def test_memory_limit_cgroup_v1_unlimited_falls_back(tmp_path):
    root = str(tmp_path)
    _write(root, "memory/memory.limit_in_bytes", str(mem._CGROUP_UNLIMITED))
    assert mem.get_memory_limit(cgroup_root=root) == int(psutil.virtual_memory().total)


def test_memory_limit_fallback_to_host(tmp_path):
    # empty cgroup root -> psutil host total
    assert mem.get_memory_limit(cgroup_root=str(tmp_path)) == int(
        psutil.virtual_memory().total
    )


def test_available_memory_cgroup(tmp_path):
    root = str(tmp_path)
    _write(root, "memory.max", str(50 * GIB))
    _write(root, "memory.current", str(20 * GIB))
    assert mem.get_available_memory(cgroup_root=root) == 30 * GIB


def test_available_memory_fallback(tmp_path):
    # no usage file -> psutil available (a non-negative int)
    avail = mem.get_available_memory(cgroup_root=str(tmp_path))
    assert isinstance(avail, int) and avail >= 0


def test_get_peak_rss():
    # exact lifetime peak RSS (getrusage ru_maxrss) — positive, and monotonic non-decreasing
    p1 = mem.get_peak_rss()
    assert isinstance(p1, int) and p1 > 0
    _ = [0] * 1_000_000  # allocate a little
    assert mem.get_peak_rss() >= p1


def test_get_peak_rss_without_resource_module(monkeypatch):
    # Windows has no `resource` module (getrusage). The fallback must still return a
    # positive, monotonic non-decreasing peak rather than raising on import or call.
    monkeypatch.setattr(mem, "resource", None)
    monkeypatch.setattr(mem, "_PEAK_RSS_FALLBACK", 0)

    p1 = mem.get_peak_rss()
    assert isinstance(p1, int) and p1 > 0
    _ = [0] * 1_000_000  # allocate a little
    p2 = mem.get_peak_rss()
    assert p2 >= p1  # monotonic, like a real lifetime peak

    # a psutil failure degrades to the last known value instead of raising
    class _Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("no psutil here")

    monkeypatch.setattr(mem.psutil, "Process", _Boom)
    assert mem.get_peak_rss() == p2
