from __future__ import annotations

from types import SimpleNamespace

from activitysim.core import mp_tasks
from activitysim.core.run_id import RunId
from activitysim.core.workflow.runner import Runner


class ProgrammaticState:
    def __init__(self):
        self.settings = SimpleNamespace(memory_profile=False, multiprocess=True)
        self.tracing = SimpleNamespace(run_id=RunId("abc123"))

    def __contains__(self, key):
        return key == "preload_injectables"

    def get_injectable(self, key):
        assert key != "run_id"
        return f"value-for-{key}"


def test_programmatic_multiprocess_run_gets_tracing_run_id(monkeypatch):
    state = ProgrammaticState()
    runner = Runner(state)
    received = {}

    def capture_run_multiprocess(passed_state, injectables):
        assert passed_state is state
        received.update(injectables)

    monkeypatch.setattr(mp_tasks, "run_multiprocess", capture_run_multiprocess)

    runner.all(config_logger=False, filter_warnings=False)

    assert received["run_id"] == "abc123"
    assert received["settings"] is state.settings
