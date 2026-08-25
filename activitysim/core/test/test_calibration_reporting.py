from __future__ import annotations

from pathlib import Path

import pandas as pd

from activitysim.core.calibration.reporting import (
    _append_iteration_records,
    _coefficient_trajectory,
    _read_component_iteration_records,
)


class _State:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def get_output_file_path(self, file_name: str) -> Path:
        return self.output_dir / file_name


def _record(attempt: int, previous: float, next_value: float) -> dict:
    return {
        "global_iter": 1,
        "attempt": attempt,
        "component_iter": 1,
        "description": "test target",
        "component": "model_a",
        "coefficient": "coef_a",
        "target_value": 0.5,
        "model_value": 0.25,
        "difference": 0.25,
        "pct_difference": 50.0,
        "hold_fast": False,
        "prev_coefficient": previous,
        "coef_delta": next_value - previous,
        "next_coefficient": next_value,
        "converged": False,
        "at_min": False,
        "at_max": False,
    }


def test_recovery_attempts_preserve_complete_coefficient_trajectory(tmp_path):
    state = _State(tmp_path)
    _append_iteration_records(state, "model_a", [_record(1, 1.0, 1.5)])
    _append_iteration_records(state, "model_a", [_record(2, 1.5, 1.75)])

    stored = pd.read_csv(tmp_path / "calibration" / "calibration_iteration_records.csv")
    assert list(stored["attempt"]) == [1, 2]
    assert stored.loc[1, "prev_coefficient"] == stored.loc[0, "next_coefficient"]

    records = _read_component_iteration_records(state, "model_a")
    trajectory, labels = _coefficient_trajectory(records, ["coef_a"])

    assert labels == ["Start", "G1-A1-C1", "G1-A2-C1"]
    assert list(trajectory["coef_a"]) == [1.0, 1.5, 1.75]
