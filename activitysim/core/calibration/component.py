# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging
import math
import re
from typing import Any

import numpy as np
import pandas as pd

from activitysim.core import simulate, workflow

from .coefficients import (
    _persist_coefficients_to_config,
    _resolve_model_settings_file,
    _setting_value,
    _settings_to_dict,
)
from .execution import _prep_model_data
from .expressions import (
    _build_expression_context,
    _compute_delta,
    _eval_numeric_value,
    _load_helper_symbols,
)
from .multiprocess import _run_mp_single_component
from .reporting import (
    _append_iteration_records,
    _append_summary_records,
    _write_generic_report,
)
from .settings import CalibrationComponentResult, CalibrationComponentSettings

logger = logging.getLogger("calibration")

DEFAULT_INCREMENT = 2.0
CALIBRATION_REQUIRED_COLUMNS = [
    "description",
    "coefficient",
    "model_value",
    "target_value",
    "hold_fast",
    "min",
    "max",
    "damping",
    "method",
    "tolerance",
]


def _run_component_model(
    state: workflow.State,
    component_name: str,
    run_model_name: str,
    prior_step: str | None,
    mp_restore_checkpoint: str | None,
    shared_data_buffers: dict | None,
) -> None:
    """Run one component simulation from its fixed pre-component state."""
    if state.settings.multiprocess and shared_data_buffers is not None:
        _run_mp_single_component(
            state,
            component_name=component_name,
            run_label=run_model_name.replace(";", "_").replace(".", "_"),
            restore_checkpoint=mp_restore_checkpoint,
            shared_data_buffers=shared_data_buffers,
        )
        return

    extra_models = _prep_model_data(state, resume_after=prior_step)
    if extra_models:
        for model_name in extra_models:
            state.run.by_name(model_name)
    state.checkpoint.add(prior_step)
    state.run.by_name(run_model_name)


def _calibrate_component(
    state: workflow.State,
    component_name: str,
    component_settings: CalibrationComponentSettings,
    prior_step: str,
    global_iter: int,
    attempt: int,
    shared_data_buffers: dict | None = None,
) -> CalibrationComponentResult:
    """Run iterative coefficient calibration for one component."""
    model_settings_file = _resolve_model_settings_file(
        component_name, component_settings
    )
    model_settings = state.filesystem.read_model_settings(
        model_settings_file, mandatory=True
    )

    coefficients_df = state.filesystem.read_model_coefficients(
        model_settings=model_settings
    )
    helper_symbols, bespoke_callable, helper_module = _load_helper_symbols(
        state,
        component_settings,
    )

    calibration_spec_df = _read_calibration_spec(
        state, component_settings.calibration_spec
    )

    utility_coeff_names = _extract_utility_coefficient_names(state, model_settings)
    _validate_calibration_coefficients_against_utility_spec(
        component_name,
        calibration_spec_df,
        utility_coeff_names,
    )

    coefficients_df = _ensure_coefficients_exist(
        component_name,
        calibration_spec_df,
        coefficients_df,
    )

    _warn_if_initial_values_outside_bounds(
        component_name, calibration_spec_df, coefficients_df
    )

    component_converged = False
    component_iterations = 0

    # Determine the checkpoint name to restore from for component re-runs.
    # In MP mode, the checkpoint that represents prior_step's completed state
    # is the last checkpoint in the pipeline before we run the component.
    # We capture it once and reuse across component iterations.
    mp_restore_checkpoint = None
    if state.settings.multiprocess and shared_data_buffers is not None:
        # The pipeline should already be open from _restore_parent_state_from_pipeline
        # called after precursor/intermediate models ran. The last checkpoint
        # in the pipeline represents the state at prior_step.
        if state.checkpoint.checkpoints:
            mp_restore_checkpoint = state.checkpoint.last_checkpoint.get(
                "checkpoint_name", "_"
            )
        else:
            mp_restore_checkpoint = "_"

    for component_iter in range(1, component_settings.submodel_max_iterations + 1):
        component_iterations = component_iter
        run_model_name = (
            f"{component_name}.c_i{component_iter};" f"g_i{global_iter};a_i{attempt}"
        )
        _run_component_model(
            state=state,
            component_name=component_name,
            run_model_name=run_model_name,
            prior_step=prior_step,
            mp_restore_checkpoint=mp_restore_checkpoint,
            shared_data_buffers=shared_data_buffers,
        )

        eval_context = _build_expression_context(
            state, helper_symbols, component_name, component_settings
        )
        eval_context["calibration_global_iteration"] = global_iter
        eval_context["calibration_attempt"] = attempt
        eval_context["calibration_component_iteration"] = component_iter

        (
            row_records,
            summary_record,
            new_coefficients_df,
            component_converged,
        ) = _evaluate_and_update(
            component_name=component_name,
            calibration_spec_df=calibration_spec_df,
            coefficients_df=coefficients_df,
            eval_context=eval_context,
            global_iter=global_iter,
            component_iter=component_iter,
            attempt=attempt,
        )

        coefficients_df = new_coefficients_df

        _persist_coefficients_to_config(state, model_settings, coefficients_df)
        _append_iteration_records(state, component_name, row_records)
        _append_summary_records(state, [summary_record])

        if component_settings.reports.generic:
            try:
                _write_generic_report(state, component_name, row_records)
            except Exception as e:
                logger.exception(
                    "calibration component %s iteration %s completed, but its "
                    "optional generic report could not be written: %s",
                    component_name,
                    component_iter,
                    e,
                )

        if bespoke_callable is not None:
            try:
                bespoke_callable(eval_context)
            except Exception as e:
                logger.exception(
                    "calibration component %s iteration %s completed, but its "
                    "optional bespoke report could not be written: %s",
                    component_name,
                    component_iter,
                    e,
                )

        if component_converged:
            break

        if component_iter == component_settings.submodel_max_iterations:
            # The update just persisted above has not yet been simulated. Run
            # the component once more so final pipeline tables and downstream
            # models use the coefficient values left in the config file.
            _run_component_model(
                state=state,
                component_name=component_name,
                run_model_name=(
                    f"{component_name}.c_final;g_i{global_iter};a_i{attempt}"
                ),
                prior_step=prior_step,
                mp_restore_checkpoint=mp_restore_checkpoint,
                shared_data_buffers=shared_data_buffers,
            )

    state.checkpoint.add(component_name)

    return CalibrationComponentResult(
        component=component_name,
        converged=component_converged,
        component_iterations=component_iterations,
    )


def _read_calibration_spec(state: workflow.State, file_name: str) -> pd.DataFrame:
    """Read calibration spec CSV and validate required columns."""
    path = state.filesystem.get_config_file_path(file_name)
    df = pd.read_csv(path, comment="#")

    missing = [c for c in CALIBRATION_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"calibration_spec {file_name} is missing required columns: {missing}"
        )

    df = df[CALIBRATION_REQUIRED_COLUMNS].copy()
    df["description"] = df["description"].astype(str)
    df["coefficient"] = df["coefficient"].astype(str)

    # Normalize booleans and defaults to keep row math deterministic.
    df["hold_fast"] = (
        df["hold_fast"]
        .fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["1", "true", "t", "yes", "y"])
    )

    for c in ["min", "max", "damping", "tolerance"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["method"] = df["method"].astype(str).str.strip().str.lower()

    bad_methods = df.loc[~df["method"].isin(["log_ratio", "odds_ratio"]), "method"]
    if not bad_methods.empty:
        raise ValueError(
            f"unsupported calibration method(s): {sorted(set(bad_methods))}"
        )

    if df["damping"].isna().any():
        raise ValueError("calibration_spec damping must be numeric")
    if df["tolerance"].isna().any():
        raise ValueError("calibration_spec tolerance must be numeric")

    return df


def _extract_utility_coefficient_names(
    state: workflow.State,
    model_settings: dict[str, Any] | Any,
) -> set[str]:
    """
    Extract coefficient names used by the configured utility specifications.

    Templated logit models map utility-spec row labels to actual coefficient
    names by segment, so their template cell values are the source of truth.
    Other models use coefficient tokens in utility-spec columns.
    """
    if _setting_value(model_settings, "COEFFICIENT_TEMPLATE"):
        template = simulate.read_model_coefficient_template(
            state.filesystem, model_settings
        )
        return {str(name) for name in template.to_numpy().ravel()}

    names: set[str] = set()

    model_settings_dict = _settings_to_dict(model_settings)
    spec_keys = [
        k for k in model_settings_dict.keys() if str(k).upper().endswith("SPEC")
    ]
    for key in spec_keys:
        spec_file = model_settings_dict.get(key)
        if not spec_file:
            continue

        spec_path = state.filesystem.get_config_file_path(spec_file)
        try:
            raw = pd.read_csv(spec_path, comment="#")
        except Exception:
            # Do not fail hard on optional or model-specific supplemental specs.
            continue

        utility_columns = [
            c
            for c in raw.columns
            if c
            not in [
                "Description",
                "Expression",
                "Label",
                "description",
                "expression",
                "label",
            ]
        ]

        for col in utility_columns:
            for value in raw[col].dropna().tolist():
                if isinstance(value, (int, float, np.number)):
                    continue
                text = str(value)
                for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
                    names.add(token)

    return names


def _validate_calibration_coefficients_against_utility_spec(
    component_name: str,
    calibration_spec_df: pd.DataFrame,
    utility_coeff_names: set[str],
) -> None:
    """Ensure calibration coefficients are referenced in utility specifications."""
    missing = sorted(
        set(calibration_spec_df["coefficient"].tolist()) - set(utility_coeff_names)
    )
    if missing:
        raise ValueError(
            f"calibration coefficients not present in model utility specification for {component_name}: {missing}"
        )


def _ensure_coefficients_exist(
    component_name: str,
    calibration_spec_df: pd.DataFrame,
    coefficients_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add calibration coefficients missing from coefficient file with value 0.0."""
    coeffs = coefficients_df.copy()

    for coefficient_name in calibration_spec_df["coefficient"].tolist():
        if coefficient_name not in coeffs.index:
            logger.warning(
                "component %s coefficient %s missing from coefficient file, adding with value 0.0",
                component_name,
                coefficient_name,
            )
            coeffs.loc[coefficient_name, "value"] = 0.0

    coeffs["value"] = pd.to_numeric(coeffs["value"], errors="coerce")
    if coeffs["value"].isna().any():
        bad = coeffs[coeffs["value"].isna()].index.tolist()
        raise ValueError(f"non-numeric coefficient values found: {bad}")

    return coeffs


def _warn_if_initial_values_outside_bounds(
    component_name: str,
    calibration_spec_df: pd.DataFrame,
    coefficients_df: pd.DataFrame,
) -> None:
    """Warn when initial coefficient values violate provided bounds."""
    for _, row in calibration_spec_df.iterrows():
        coefficient_name = row["coefficient"]
        current_value = float(coefficients_df.loc[coefficient_name, "value"])
        lower = row["min"]
        upper = row["max"]

        if not pd.isna(lower) and current_value < float(lower):
            logger.warning(
                "component %s coefficient %s starts below min bound (%s < %s)",
                component_name,
                coefficient_name,
                current_value,
                lower,
            )
        if not pd.isna(upper) and current_value > float(upper):
            logger.warning(
                "component %s coefficient %s starts above max bound (%s > %s)",
                component_name,
                coefficient_name,
                current_value,
                upper,
            )


def _evaluate_and_update(
    component_name: str,
    calibration_spec_df: pd.DataFrame,
    coefficients_df: pd.DataFrame,
    eval_context: dict[str, Any],
    global_iter: int,
    component_iter: int,
    attempt: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any], pd.DataFrame, bool]:
    """Evaluate spec rows, update coefficients, and return detailed records."""
    updated = coefficients_df.copy()
    records: list[dict[str, Any]] = []

    max_difference = -math.inf
    max_difference_coefficient = ""
    max_change = -math.inf
    max_change_coefficient = ""

    num_converged = 0

    for _, row in calibration_spec_df.iterrows():
        coefficient_name = row["coefficient"]
        description = row["description"]
        method = row["method"]
        hold_fast = bool(row["hold_fast"])

        default_increment = (
            row["default_increment"]
            if "default_increment" in row.index
            else DEFAULT_INCREMENT
        )

        prev_value = float(updated.loc[coefficient_name, "value"])

        model_value = _eval_numeric_value(
            row["model_value"],
            eval_context,
            component_name,
            description,
            "model_value",
        )
        target_value = _eval_numeric_value(
            row["target_value"],
            eval_context,
            component_name,
            description,
            "target_value",
        )

        difference = target_value - model_value
        pct_difference = _safe_percent_difference(difference, target_value)

        tolerance = float(row["tolerance"])
        converged = abs(difference) <= tolerance

        damping = float(row["damping"])
        raw_delta = _compute_delta(
            method=method,
            model_value=model_value,
            target_value=target_value,
            damping=damping,
            component_name=component_name,
            description=description,
            default_increment=default_increment,
        )

        candidate_value = (
            prev_value if hold_fast or converged else prev_value + raw_delta
        )

        at_min = False
        at_max = False

        lower = row["min"]
        upper = row["max"]

        if not pd.isna(lower) and candidate_value <= float(lower):
            candidate_value = float(lower)
            at_min = True
        if not pd.isna(upper) and candidate_value >= float(upper):
            candidate_value = float(upper)
            at_max = True

        if not np.isfinite(candidate_value):
            raise RuntimeError(
                f"non-finite next coefficient for {component_name} / {description} / {coefficient_name}"
            )

        updated.loc[coefficient_name, "value"] = candidate_value

        abs_diff = abs(difference)
        abs_change = abs(candidate_value - prev_value)

        if abs_diff > max_difference:
            max_difference = abs_diff
            max_difference_coefficient = coefficient_name

        if abs_change > max_change:
            max_change = abs_change
            max_change_coefficient = coefficient_name

        if converged:
            num_converged += 1

        records.append(
            {
                "global_iter": global_iter,
                "attempt": attempt,
                "component_iter": component_iter,
                "description": description,
                "component": component_name,
                "coefficient": coefficient_name,
                "target_value": target_value,
                "model_value": model_value,
                "difference": difference,
                "pct_difference": pct_difference,
                "hold_fast": hold_fast,
                "prev_coefficient": prev_value,
                "coef_delta": abs_change,
                "next_coefficient": candidate_value,
                "converged": converged,
                "at_min": at_min,
                "at_max": at_max,
            }
        )

    total_rows = len(calibration_spec_df)
    num_unconverged = total_rows - num_converged
    component_converged = num_unconverged == 0

    summary_record = {
        "global_iter": global_iter,
        "attempt": attempt,
        "component_iter": component_iter,
        "component": component_name,
        "max_difference": max_difference if max_difference != -math.inf else 0.0,
        "max_difference_coefficient": max_difference_coefficient,
        "max_change": max_change if max_change != -math.inf else 0.0,
        "max_change_coefficient": max_change_coefficient,
        "num_converged_iter": num_converged,
        "tot_converged": num_converged,
        "num_unconverged": num_unconverged,
    }

    return records, summary_record, updated, component_converged


def _safe_percent_difference(difference: float, target_value: float) -> float:
    """Return a stable percentage difference with zero-target handling."""
    if target_value == 0:
        return math.inf if difference != 0 else 0.0
    return (difference / target_value) * 100.0
