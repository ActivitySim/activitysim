# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import importlib
import importlib.util
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from activitysim.core import workflow

from .reporting import _component_output_dir
from .settings import CalibrationComponentSettings

logger = logging.getLogger("calibration")


def _build_expression_context(
    state: workflow.State,
    helper_symbols: dict[str, Any],
    component_name: str,
    component_settings: CalibrationComponentSettings,
) -> dict[str, Any]:
    """Create the evaluation context for model_value and target_value expressions."""
    context: dict[str, Any] = {
        "state": state,
        "np": np,
        "pd": pd,
        "component_output_dir": _component_output_dir(state, component_name),
        "component_settings": component_settings,
    }

    # Load active tables into context for direct expression access.
    for table_name in list(state.existing_table_names):
        try:
            context[table_name] = state.get_dataframe(table_name, as_copy=False)
        except Exception:
            # Some entries may not be available as dataframes in all contexts.
            continue
    try:
        network_los = state.get_injectable("network_los")
        context["network_los"] = network_los
        context["skim_dict"] = network_los.get_default_skim_dict()
    except Exception:
        # Network LOS may not be available in all contexts.
        pass

    context.update(helper_symbols)
    # Explicit function-call context used by calibration expressions.
    context["context"] = context
    return context


def _eval_numeric_value(
    raw_value: Any,
    context: dict[str, Any],
    component_name: str,
    description: str,
    field_name: str,
) -> float:
    """Evaluate numeric or expression value and enforce finite numeric result."""
    if isinstance(raw_value, (int, float, np.number)) and not pd.isna(raw_value):
        value = float(raw_value)
    else:
        try:
            value = eval(str(raw_value), {}, context)
        except Exception as err:
            raise RuntimeError(
                f"error evaluating {field_name} for {component_name} / {description}: {raw_value}"
            ) from err

    try:
        value = float(value)
    except Exception as err:
        raise RuntimeError(
            f"{field_name} did not evaluate to a numeric value for {component_name} / {description}: {value}"
        ) from err

    if not np.isfinite(value):
        raise RuntimeError(
            f"{field_name} evaluated to non-finite value for {component_name} / {description}: {value}"
        )

    return value


def _compute_delta(
    method: str,
    model_value: float,
    target_value: float,
    damping: float,
    component_name: str,
    description: str,
    default_increment: float,
) -> float:
    """Compute damped coefficient delta using selected method."""
    if damping < 0:
        raise RuntimeError(
            f"negative damping not allowed for {component_name} / {description}: {damping}"
        )

    if method == "log_ratio":
        if model_value <= 0 or target_value <= 0:
            logger.warning(
                f"log_ratio requires positive model and target values for {component_name} / {description}. Falling back to default increment {default_increment}"
            )
            if model_value <= 0 and target_value > 0:
                return default_increment
            elif model_value > 0 and target_value <= 0:
                return -default_increment
            else:
                return 0
        delta = math.log(target_value / model_value) * damping

    elif method == "odds_ratio":
        # Formula requested by the calibration outline.
        numerator = (target_value * model_value) - target_value
        denominator = (target_value * model_value) - model_value

        if numerator <= 0 or denominator <= 0:
            logger.warning(
                f"odds_ratio produced invalid numerator/denominator for {component_name} / {description}. Falling back to default increment {default_increment}"
            )
            if model_value <= 0 and target_value > 0:
                return default_increment
            elif model_value > 0 and target_value <= 0:
                return -default_increment
            else:
                return 0

        ratio = numerator / denominator
        if ratio <= 0 or not np.isfinite(ratio):
            raise RuntimeError(
                f"odds_ratio produced invalid ratio for {component_name} / {description}"
            )

        delta = math.log(ratio) * damping

    else:
        raise RuntimeError(f"unsupported calibration method: {method}")

    if not np.isfinite(delta):
        raise RuntimeError(
            f"coefficient delta is non-finite for {component_name} / {description}"
        )

    return delta


def _load_helper_symbols(
    state: workflow.State,
    component_settings: CalibrationComponentSettings,
) -> tuple[dict[str, Any], Any | None, Any | None]:
    """Load helper module and return evaluation symbols and bespoke function."""
    if not component_settings.helper_module:
        return {}, None, None

    module = _load_helper_module(state, component_settings.helper_module)
    symbols = {
        name: obj for name, obj in vars(module).items() if not name.startswith("__")
    }

    bespoke = None
    if component_settings.reports and component_settings.reports.bespoke:
        fn_name = component_settings.reports.bespoke
        if not hasattr(module, fn_name):
            raise RuntimeError(
                f"helper module does not define bespoke function {fn_name}"
            )
        bespoke = getattr(module, fn_name)

    return symbols, bespoke, module


def _load_helper_module(state: workflow.State, helper_module: str):
    """Load helper module by file path or import path."""
    if helper_module.endswith(".py"):
        helper_path = state.filesystem.get_config_file_path(helper_module)
        module_name = Path(helper_module).stem

        spec = importlib.util.spec_from_file_location(module_name, helper_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"unable to load helper module from {helper_path}")

        module = importlib.util.module_from_spec(spec)
        # Expose state as a global for compatibility with helper examples.
        module.state = state
        spec.loader.exec_module(module)
        return module

    module = importlib.import_module(helper_module)
    setattr(module, "state", state)
    return module
