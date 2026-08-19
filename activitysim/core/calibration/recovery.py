# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import json
import os
from typing import Any

from activitysim.core import workflow

CALIBRATION_PROGRESS_FILE = "calibration/calibration_progress.json"


def _mark_global_iteration_in_progress(
    state: workflow.State,
    global_iteration: int,
) -> None:
    """Durably mark a global iteration in progress."""
    _write_progress(
        state,
        {
            "in_progress_iteration": global_iteration,
            "next_global_iteration": global_iteration,
            "last_completed_global_iteration": global_iteration - 1,
        },
    )


def _read_progress(state: workflow.State) -> dict[str, Any] | None:
    """Read persisted calibration progress metadata if it exists."""
    path = state.get_output_file_path(CALIBRATION_PROGRESS_FILE)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_progress(state: workflow.State, payload: dict[str, Any]) -> None:
    """Atomically write calibration progress metadata."""
    path = state.get_output_file_path(CALIBRATION_PROGRESS_FILE)
    os.makedirs(path.parent, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with open(temporary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(temporary_path, path)


def _write_completed_progress(
    state: workflow.State,
    completed_global_iterations: int,
    converged: bool,
) -> None:
    """Mark calibration complete after all final output has been written."""
    _write_progress(
        state,
        {
            "complete": True,
            "in_progress_iteration": None,
            "next_global_iteration": completed_global_iterations + 1,
            "last_completed_global_iteration": completed_global_iterations,
            "converged": converged,
        },
    )
