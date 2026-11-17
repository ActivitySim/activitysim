# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from activitysim.core import workflow

from .settings import CalibrationComponentSettings


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

    coeff_path = Path(state.filesystem.get_config_file_path(coeff_file))
    temporary_path = coeff_path.with_name(f".{coeff_path.name}.tmp")
    output.to_csv(temporary_path)
    os.replace(temporary_path, coeff_path)


def _infer_model_settings_file(component_name: str) -> str:
    """Infer model settings yaml filename from component step name."""
    # This follows the dominant naming convention in the existing codebase.
    if component_name.endswith("_simulate"):
        base = component_name[: -len("_simulate")]
    else:
        base = component_name
    return f"{base}.yaml"


def _resolve_model_settings_file(
    component_name: str,
    component_settings: CalibrationComponentSettings,
) -> str:
    """Return an explicit component settings file or infer the conventional name."""
    return component_settings.model_settings_file or _infer_model_settings_file(
        component_name
    )


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
