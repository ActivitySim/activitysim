# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from activitysim.core import workflow

from .settings import CalibrationConfig


def _persist_coefficients_to_config(
    state: workflow.State,
    model_settings: dict[str, Any] | Any,
    coefficients_df: pd.DataFrame,
) -> None:
    """Write updated coefficients back to the component coefficient file in configs."""
    coeff_file = _setting_value(model_settings, "COEFFICIENTS")
    if not coeff_file:
        raise RuntimeError("component model settings missing COEFFICIENTS")

    output = coefficients_df.copy()
    output.index.name = "coefficient_name"

    coeff_path = state.filesystem.get_config_file_path(coeff_file)
    output.to_csv(coeff_path)


def _infer_model_settings_file(component_name: str) -> str:
    """Infer model settings yaml filename from component step name."""
    # This follows the dominant naming convention in the existing codebase.
    if component_name.endswith("_simulate"):
        base = component_name[: -len("_simulate")]
    else:
        base = component_name
    return f"{base}.yaml"


def _settings_to_dict(model_settings: dict[str, Any] | Any) -> dict[str, Any]:
    """Convert pydantic or dict model settings to a plain dictionary."""
    if isinstance(model_settings, dict):
        return model_settings
    if hasattr(model_settings, "model_dump"):
        return model_settings.model_dump()
    return dict(model_settings)


def _setting_value(model_settings: dict[str, Any] | Any, key: str, default=None):
    """Read a setting value from dict-like or attribute-based settings."""
    if isinstance(model_settings, dict):
        return model_settings.get(key, default)
    return getattr(model_settings, key, default)


def _calibration_coefficient_paths(
    state: workflow.State,
    calibration_settings: CalibrationConfig,
) -> list[Path]:
    """Return the unique coefficient files modified by this calibration run."""
    paths: list[Path] = []
    seen: set[str] = set()

    for component_name in calibration_settings.run.calibrate_models:
        model_settings_file = _infer_model_settings_file(component_name)
        model_settings = state.filesystem.read_model_settings(
            model_settings_file, mandatory=True
        )
        coefficient_file = _setting_value(model_settings, "COEFFICIENTS")
        if not coefficient_file:
            raise RuntimeError(
                f"component {component_name} model settings missing COEFFICIENTS"
            )

        path = Path(state.filesystem.get_config_file_path(coefficient_file)).resolve()
        key = os.path.normcase(str(path))
        if key not in seen:
            paths.append(path)
            seen.add(key)

    return paths
