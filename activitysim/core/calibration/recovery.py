# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import json
import os
import shutil
from typing import Any

from activitysim.core import workflow

from .coefficients import _calibration_coefficient_paths
from .settings import CalibrationConfig

CALIBRATION_PROGRESS_FILE = "calibration/calibration_progress.json"
CALIBRATION_RECOVERY_DIR = "calibration/recovery"


def _begin_global_iteration_transaction(
    state: workflow.State,
    calibration_settings: CalibrationConfig,
    global_iteration: int,
) -> None:
    """Snapshot coefficients and durably mark a global iteration in progress."""
    recovery_dir = state.get_output_file_path(CALIBRATION_RECOVERY_DIR)
    os.makedirs(recovery_dir, exist_ok=True)

    for file_number, coefficient_path in enumerate(
        _calibration_coefficient_paths(state, calibration_settings)
    ):
        if not coefficient_path.exists():
            raise FileNotFoundError(
                f"calibration coefficient file not found: {coefficient_path}"
            )

        backup_name = f"{file_number:03d}_{coefficient_path.name}"
        shutil.copyfile(coefficient_path, recovery_dir / backup_name)

    # Write the marker only after all backups exist. If backup creation is
    # interrupted, the previous between-iteration progress remains valid.
    _write_progress(
        state,
        {
            "in_progress_iteration": global_iteration,
            "next_global_iteration": global_iteration,
            "last_completed_global_iteration": global_iteration - 1,
        },
    )


def _restore_coefficient_backups(
    state: workflow.State,
    calibration_settings: CalibrationConfig,
) -> None:
    """Restore the coefficient backups for an interrupted global iteration."""
    recovery_dir = state.get_output_file_path(CALIBRATION_RECOVERY_DIR)
    for file_number, coefficient_path in enumerate(
        _calibration_coefficient_paths(state, calibration_settings)
    ):
        backup_path = recovery_dir / f"{file_number:03d}_{coefficient_path.name}"
        if not backup_path.exists():
            raise RuntimeError(
                f"cannot recover interrupted calibration iteration: missing {backup_path}"
            )
        shutil.copyfile(backup_path, coefficient_path)


def _read_progress(state: workflow.State) -> dict[str, Any] | None:
    """Read persisted calibration progress metadata if it exists."""
    path = state.get_output_file_path(CALIBRATION_PROGRESS_FILE)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
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
