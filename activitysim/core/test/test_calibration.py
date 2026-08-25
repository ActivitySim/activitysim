from __future__ import annotations

import math

import pandas as pd
import pytest

from activitysim.core.calibration.component import _evaluate_and_update
from activitysim.core.calibration.expressions import _compute_delta
from activitysim.core.calibration.orchestrator import (
    _components_ran_for_convergence,
)


@pytest.mark.parametrize(
    ("method", "model_value", "target_value", "damping", "expected"),
    [
        ("log_ratio", 0.25, 0.5, 0.5, math.log(2) * 0.5),
        (
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
    [(0.0, 0.5, 2.0), (1.0, 0.5, -2.0), (0.0, 0.0, 0.0)],
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

    assert updated.loc["coef_update", "value"] == pytest.approx(1.0 + math.log(2) * 0.5)
    assert updated.loc["coef_converged", "value"] == 2.0
    assert updated.loc["coef_held", "value"] == 3.0
    assert records[1]["coef_delta"] == 0.0
    assert records[2]["coef_delta"] == 0.0
    assert component_converged is False


@pytest.mark.parametrize(
    (
        "first_model_idx",
        "last_calib_model_idx",
        "global_iter",
        "start_global_iter",
        "expected",
    ),
    [
        (None, 10, 1, 1, True),
        (5, 10, 1, 1, True),
        (11, 10, 1, 1, False),
        (11, 10, 2, 1, True),
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
