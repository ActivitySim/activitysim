from __future__ import annotations

import math
import copy

import pandas as pd
import pytest
from pathlib import Path
from pydantic import ValidationError

from activitysim.core.calibration.component import _evaluate_and_update
from activitysim.core.calibration.expressions import _compute_delta
from activitysim.core.calibration.orchestrator import (
    _components_ran_for_convergence,
)
from activitysim.core.calibration.reporting import (
    _append_iteration_records,
    _coefficient_trajectory,
    _read_component_iteration_records,
)
from activitysim.core.calibration.settings import CalibrationConfig


@pytest.mark.parametrize(
    ("method", "model_value", "target_value", "damping", "expected"),
    [
        # log_ratio: delta = log(target / model) * damping = log(2) * 0.5
        ("log_ratio", 0.25, 0.5, 0.5, math.log(2) * 0.5),
        (
            # odds_ratio: delta = log((t*(1-m)) / (m*(1-t))) * damping
            "odds_ratio",
            0.4,
            0.6,
            0.5,
            math.log((0.6 * (1 - 0.4)) / (0.4 * (1 - 0.6))) * 0.5,
        ),
    ],
)
def test_compute_delta(method, model_value, target_value, damping, expected):
    delta = _compute_delta(
        method=method,
        model_value=model_value,
        target_value=target_value,
        damping=damping,
        component_name="test_component",
        description="test target",
        default_increment=2.0,
    )

    assert delta == pytest.approx(expected)


@pytest.mark.parametrize(
    ("model_value", "target_value", "expected"),
    [
        (
            0.0,
            0.5,
            2.0,
        ),  # model is zero but target is not: fallback must nudge coefficient upward
        (
            1.0,
            0.5,
            -2.0,
        ),  # model is one but target is not: fallback must nudge coefficient downward
        (0.0, 0.0, 0.0),  # both at boundary zero: no change needed
    ],
)
def test_odds_ratio_boundary_fallback_has_correct_direction(
    model_value, target_value, expected
):
    delta = _compute_delta(
        method="odds_ratio",
        model_value=model_value,
        target_value=target_value,
        damping=1.0,
        component_name="test_component",
        description="test target",
        default_increment=2.0,
    )

    assert delta == expected


def test_evaluate_and_update_only_changes_eligible_coefficients():
    calibration_spec = pd.DataFrame(
        [
            {
                "description": "unconverged",
                "coefficient": "coef_update",
                "model_value": 0.25,
                "target_value": 0.5,
                "hold_fast": False,
                "min": -10.0,
                "max": 10.0,
                "damping": 0.5,
                "method": "log_ratio",
                "tolerance": 0.01,
            },
            {
                "description": "within tolerance",
                "coefficient": "coef_converged",
                "model_value": 0.49,
                "target_value": 0.5,
                "hold_fast": False,
                "min": -10.0,
                "max": 10.0,
                "damping": 1.0,
                "method": "log_ratio",
                "tolerance": 0.02,
            },
            {
                "description": "held fixed",
                "coefficient": "coef_held",
                "model_value": 0.25,
                "target_value": 0.5,
                "hold_fast": True,
                "min": -10.0,
                "max": 10.0,
                "damping": 1.0,
                "method": "log_ratio",
                "tolerance": 0.01,
            },
        ]
    )
    coefficients = pd.DataFrame(
        {"value": [1.0, 2.0, 3.0]},
        index=["coef_update", "coef_converged", "coef_held"],
    )

    records, _, updated, component_converged = _evaluate_and_update(
        component_name="test_component",
        calibration_spec_df=calibration_spec,
        coefficients_df=coefficients,
        eval_context={},
        global_iter=1,
        component_iter=1,
    )

    assert updated.loc["coef_update", "value"] == pytest.approx(
        1.0 + math.log(2) * 0.5
    )  # unconverged: full log-ratio delta applied
    assert (
        updated.loc["coef_converged", "value"] == 2.0
    )  # within tolerance: value must not change
    assert (
        updated.loc["coef_held", "value"] == 3.0
    )  # hold_fast=True: value must not change
    assert records[1]["coef_delta"] == 0.0  # converged row records zero change
    assert records[2]["coef_delta"] == 0.0  # held row records zero change
    assert (
        component_converged is False
    )  # one unconverged row means the whole component is not done


@pytest.mark.parametrize(
    (
        "first_model_idx",
        "last_calib_model_idx",
        "global_iter",
        "start_global_iter",
        "expected",
    ),
    [
        (
            None,
            10,
            1,
            1,
            True,
        ),  # no pre-calibration model ran: always counts toward convergence
        (
            5,
            10,
            1,
            1,
            True,
        ),  # model ran before last calibration model in pipeline: counts
        (
            11,
            10,
            1,
            1,
            False,
        ),  # model ran after last calibration model: skipped on first global iter
        (
            11,
            10,
            2,
            1,
            True,
        ),  # same position but later global iter: skip only applies to first pass
    ],
)
def test_components_ran_for_convergence(
    first_model_idx,
    last_calib_model_idx,
    global_iter,
    start_global_iter,
    expected,
):
    assert (
        _components_ran_for_convergence(
            first_model_idx=first_model_idx,
            last_calib_model_idx=last_calib_model_idx,
            global_iter=global_iter,
            start_global_iter=start_global_iter,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("location", "unknown_setting"),
    [
        ((), "cleanup_pipeline_after_run"),  # unknown at top-level CalibrationConfig
        (("run",), "restart_after"),  # unknown inside CalibrationRunSettings
        (
            ("model_settings", "test_component"),
            "survey_file",
        ),  # unknown inside CalibrationComponentSettings
        (
            ("model_settings", "test_component", "reports"),
            "unexpected_report",
        ),  # unknown inside CalibrationReportSettings
    ],
)
def test_calibration_settings_reject_unknown_fields(location, unknown_setting):
    settings = {
        "enable": True,
        "run": {
            "calibrate_models": ["test_component"],
            "global_iterations": 1,
        },
        "model_settings": {
            "test_component": {
                "calibration_spec": "test_calibration.csv",
                "reports": {"generic": True},
            }
        },
    }
    invalid_settings = copy.deepcopy(settings)
    container = invalid_settings
    for key in location:
        container = container[key]
    container[unknown_setting] = True

    with pytest.raises(ValidationError) as error:
        CalibrationConfig.model_validate(invalid_settings)

    assert error.value.errors()[0]["loc"] == (*location, unknown_setting)


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
    # attempt 1: coefficient moves 1.0 → 1.5
    _append_iteration_records(state, "model_a", [_record(1, 1.0, 1.5)])
    # attempt 2: picks up exactly where attempt 1 ended, 1.5 → 1.75
    _append_iteration_records(state, "model_a", [_record(2, 1.5, 1.75)])

    stored = pd.read_csv(tmp_path / "calibration" / "calibration_iteration_records.csv")
    assert list(stored["attempt"]) == [1, 2]
    # the chain must be unbroken: attempt 2 prev_coefficient must equal attempt 1 next_coefficient
    assert stored.loc[1, "prev_coefficient"] == stored.loc[0, "next_coefficient"]

    records = _read_component_iteration_records(state, "model_a")
    trajectory, labels = _coefficient_trajectory(records, ["coef_a"])

    # labels: one "Start" entry (initial value) plus one label per recorded iteration
    assert labels == ["Start", "G1-A1-C1", "G1-A2-C1"]
    assert list(trajectory["coef_a"]) == [1.0, 1.5, 1.75]
