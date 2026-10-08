from __future__ import annotations

import ctypes
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from activitysim.core import mem


def test_release_memory_is_advisory():
    assert isinstance(mem.release_memory(), bool)


@pytest.mark.parametrize("platform", ["linux", "darwin", "win32", "unsupported"])
@pytest.mark.parametrize("gc_enabled", [True, False])
@pytest.mark.parametrize(
    "outcome", ["success", "no_release", "missing", "load_error", "call_error"]
)
def test_release_memory_platforms(monkeypatch, platform, gc_enabled, outcome):
    # Mock the platform module itself so pytest and dependencies retain the host
    # platform, and never invoke a foreign allocator on the test machine.
    monkeypatch.setattr(mem, "sys", SimpleNamespace(platform=platform))
    gc = Mock()
    gc.isenabled.return_value = gc_enabled
    monkeypatch.setattr(mem, "gc", gc)
    release = Mock(return_value=1 if outcome == "success" else 0)
    handle = Mock(return_value=1234)
    library = SimpleNamespace()
    if outcome != "missing":
        library = SimpleNamespace(
            malloc_trim=release,
            malloc_default_zone=handle,
            malloc_zone_pressure_relief=release,
            GetCurrentProcess=handle,
            SetProcessWorkingSetSize=release,
        )
    loader = Mock(return_value=library)
    if outcome == "load_error":
        loader.side_effect = OSError("allocator unavailable")
    elif outcome == "call_error":
        release.side_effect = OSError("allocator call failed")
    monkeypatch.setattr(mem.ctypes, "CDLL", loader)
    monkeypatch.setattr(mem.ctypes, "WinDLL", loader, raising=False)

    assert mem.release_memory() is (outcome == "success" and platform != "unsupported")
    gc.collect.assert_called_once_with()
    if gc_enabled:
        gc.enable.assert_not_called()
        gc.disable.assert_not_called()
    else:
        assert [call[0] for call in gc.mock_calls] == [
            "isenabled",
            "enable",
            "collect",
            "disable",
        ]
    if platform == "unsupported":
        loader.assert_not_called()
    else:
        if platform == "win32":
            loader.assert_called_once_with("kernel32", use_last_error=True)
        else:
            loader.assert_called_once_with(None)
        if outcome in ("success", "no_release", "call_error"):
            if platform == "linux":
                release.assert_called_once_with(0)
                assert release.argtypes == [ctypes.c_size_t]
                assert release.restype == ctypes.c_int
            elif platform == "darwin":
                handle.assert_called_once_with()
                release.assert_called_once_with(1234, 0)
                assert handle.restype == ctypes.c_void_p
                assert release.argtypes == [ctypes.c_void_p, ctypes.c_size_t]
                assert release.restype == ctypes.c_size_t
            else:
                handle.assert_called_once_with()
                maximum_size = ctypes.c_size_t(-1).value
                release.assert_called_once_with(1234, maximum_size, maximum_size)
                assert handle.restype == ctypes.c_void_p
                assert release.argtypes == [
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                    ctypes.c_size_t,
                ]
                assert release.restype == ctypes.c_int
