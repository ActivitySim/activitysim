from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from activitysim.core import chunk


@pytest.mark.parametrize("mode", chunk.TRAINING_MODES)
@pytest.mark.parametrize("explicit_chunk", [None, 0, 0.5, 1000])
def test_release_explicit_chunk_memory_is_gated(monkeypatch, mode, explicit_chunk):
    # Allocator release must only run when explicit chunking is both the global
    # mode and configured for the component; otherwise it is pure overhead.
    release = Mock(return_value=True)
    monkeypatch.setattr(chunk.mem, "release_memory", release)
    state = SimpleNamespace(settings=SimpleNamespace(chunk_training_mode=mode))

    expected = bool(explicit_chunk) and mode == chunk.MODE_EXPLICIT
    assert chunk.release_explicit_chunk_memory(state, explicit_chunk) is expected
    assert release.called is expected
