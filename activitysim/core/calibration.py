# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pydantic import model_validator

from activitysim.core import workflow
from activitysim.core.configuration import PydanticReadable
from activitysim.core.configuration.base import PydanticBase

logger = logging.getLogger("calibration")

plt.style.use("seaborn-v0_8-darkgrid")

CALIBRATION_SETTINGS_FILE_NAME = "calibration.yaml"
CALIBRATION_OUTPUT_DIR = "calibration"
CALIBRATION_PROGRESS_FILE = "calibration/calibration_progress.json"
CALIBRATION_ITERATION_FILE = "calibration/calibration_iteration_records.csv"
CALIBRATION_SUMMARY_FILE = "calibration/calibration_iteration_summary.csv"
CALIBRATION_FINAL_COEFFICIENTS_FILE = "calibration/final_calibrated_coefficients.csv"

DEFAULT_INCREMENT = 2.0
MAX_COEFFS_IN_GRAPH = 10

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

MP_INJECTABLES = [
    "data_dir",
    "configs_dir",
    "data_model_dir",
    "output_dir",
    "cache_dir",
    "settings_file_name",
    "imported_extensions",
    "run_timestamp",
    "run_id",
]


class CalibrationRunSettings(PydanticBase):
    """Run-control settings for calibration."""

    resume_after: Optional[str] = None
    calibrate_models: list[str]
    restart_after: list[str] = []
    global_iterations: int = 1
    complete_steps: bool = False


class CalibrationReportsSettings(PydanticBase):
    """Reporting settings for a calibrated component."""

    generic: bool = True
    bespoke: str | None = None


class CalibrationComponentSettings(PydanticBase):
    """Settings for one calibratable model component."""

    calibration_spec: str
    helper_module: str | None = None
    submodel_max_iterations: int = 1
    reports: CalibrationReportsSettings = CalibrationReportsSettings()
    survey_file: str


class CalibrationConfig(PydanticReadable):
    """Top-level calibration configuration."""

    enable: bool = False
    run: CalibrationRunSettings
    model_settings: dict[str, CalibrationComponentSettings] = {}

    @model_validator(mode="after")
    def validate_model_settings(self):
        """Validate that configured components are aligned with run settings."""
        for component in self.run.calibrate_models:
            if component not in self.model_settings:
                raise ValueError(
                    f"calibration model '{component}' is not in model_settings"
                )

        for component in self.run.restart_after:
            if component not in self.run.calibrate_models:
                raise ValueError(
                    f"restart_after component '{component}' is not in calibrate_models"
                )

        if self.run.global_iterations < 1:
            raise ValueError("max_iterations must be >= 1")

        return self


@dataclass
class CalibrationComponentResult:
    """Result details from calibrating one component."""

    component: str
    converged: bool
    component_iterations: int


@dataclass
class CalibrationRunResult:
    """Result details from a complete global calibration loop."""

    converged: bool
    completed_global_iterations: int


def read_calibration_settings(state: workflow.State) -> CalibrationConfig | None:
    """Read and validate calibration settings if calibration.yaml exists."""
    return CalibrationConfig.read_settings_file(
        state.filesystem,
        CALIBRATION_SETTINGS_FILE_NAME,
        mandatory=False,
    )


def calibration_enabled(state: workflow.State) -> bool:
    """Return True when calibration.yaml exists and is enabled."""
    settings = read_calibration_settings(state)
    return bool(settings and settings.enable)


def run_calibration_loop(
    state: workflow.State,
    models: list[str],
    memory_sidecar_process=None,
) -> CalibrationRunResult:
    """
    Run the global calibration workflow.

    This function intentionally minimizes changes to the existing run mechanics:
    it always reuses ActivitySim's normal model execution paths and only adds
    calibration orchestration around them.
    """
    calibration_settings = read_calibration_settings(state)
    if not calibration_settings or not calibration_settings.enable:
        raise RuntimeError("calibration loop called while calibration is disabled")

    assert all(
        [c in models for c in calibration_settings.run.calibrate_models]
    ), f"settings.yaml steps list does not include calibration model{'s' if len([c for c in calibration_settings.run.calibrate_models if c not in models]) != 1 else ''} {[c for c in calibration_settings.run.calibrate_models if c not in models]}"

    # sort calibration models into main model order
    calibration_settings.run.calibrate_models = sorted(
        calibration_settings.run.calibrate_models, key=lambda x: models.index(x)
    )
    first_calib_model_idx = models.index(calibration_settings.run.calibrate_models[0])

    _ensure_calibration_output_dir(state)

    # If there is recoverable calibration progress from a prior interrupted run,
    # continue from that iteration. Coefficient updates are persisted in config
    # coefficient files, so restarting from a later global iteration is compatible
    # with current checkpoint semantics.
    progress = _read_progress(state)
    start_global_iter = int(progress.get("next_global_iteration", 1)) if progress else 1

    original_pipeline_name = state.filesystem.pipeline_file_name

    try:
        for global_iter in range(
            start_global_iter,
            start_global_iter + calibration_settings.run.global_iterations,
        ):
            logger.info(
                "calibration global iteration %s/%s",
                global_iter - start_global_iter,
                calibration_settings.run.global_iterations,
            )

            # Run ActivitySim normally from resume_after through production model steps.
            _run_precursor_components(
                state,
                models=models[:first_calib_model_idx],
                resume_after=calibration_settings.run.resume_after
                if global_iter == 1
                else _prior_step_name(
                    models, calibration_settings.run.calibrate_models[0]
                ),
                global_iter=global_iter,
                memory_sidecar_process=memory_sidecar_process,
            )

            all_converged = True

            last_calibrated_component = None
            for component in calibration_settings.run.calibrate_models:
                component_settings = calibration_settings.model_settings[component]

                prior_step = _prior_step_name(models, component)

                if last_calibrated_component is not None:

                    # run all models b/w the last calibrated model and the current one
                    _run_intermediate_components(
                        state,
                        models=models[
                            models.index(last_calibrated_component)
                            + 1 : models.index(component)
                        ],
                        resume_after=last_calibrated_component,
                        memory_sidecar_process=memory_sidecar_process,
                    )

                component_result = _calibrate_component(
                    state=state,
                    component_name=component,
                    component_settings=component_settings,
                    prior_step=prior_step,
                    global_iter=global_iter,
                )

                all_converged = all_converged and component_result.converged

                last_calibrated_component = component

            if calibration_settings.run.complete_steps or (
                start_global_iter + calibration_settings.run.global_iterations
                == global_iter + 1
            ):
                # finish the full model chain
                _run_subsequent_components(
                    state,
                    models=models[models.index(last_calibrated_component) + 1 :],
                    resume_after=last_calibrated_component,
                    memory_sidecar_process=memory_sidecar_process,
                )

            _write_progress(
                state,
                {
                    "next_global_iteration": global_iter + 1,
                    "last_completed_global_iteration": global_iter,
                },
            )

        _write_final_coefficients_snapshot(state, calibration_settings)

        iteration_records = (
            pd.read_csv(state.get_output_file_path(CALIBRATION_ITERATION_FILE))
            .set_index(["global_iter", "component_iter", "coefficient"])
            .sort_index()
        )

        for component in iteration_records.component.unique():

            recs = iteration_records.loc[iteration_records.component == component]
            coefs = sorted(recs.index.get_level_values("coefficient").unique())
            n_sets = math.ceil(len(coefs) / MAX_COEFFS_IN_GRAPH)
            for coef_set in range(n_sets):
                set_coefs = coefs[
                    coef_set
                    * MAX_COEFFS_IN_GRAPH : min(
                        len(coefs), (coef_set + 1) * MAX_COEFFS_IN_GRAPH
                    )
                ]
                ax = (
                    recs[recs.index.get_level_values("coefficient").isin(set_coefs)]
                    .next_coefficient.unstack("coefficient")
                    .plot()
                )
                ax.xaxis.set_label_text("Component iteration")
                ax.yaxis.set_label_text("Coefficient value")

                ax.legend(title="Coefficient label")
                ax.figure.savefig(
                    os.path.join(
                        state.filesystem.output_dir,
                        "calibration",
                        f"{component}_coefficient_progress_set_{coef_set}.png",
                    )
                )

                last_global = recs[
                    recs.index.get_level_values("coefficient").isin(set_coefs)
                ].index.get_level_values("global_iter")[-1]
                last_comp = (
                    recs[recs.index.get_level_values("coefficient").isin(set_coefs)]
                    .loc[last_global]
                    .index.get_level_values("component_iter")[-1]
                )

                last_records = recs[
                    recs.index.get_level_values("coefficient").isin(set_coefs)
                ].xs((last_global, last_comp), level=("global_iter", "component_iter"))[
                    ["target_value", "model_value"]
                ]
                ax = last_records.plot.barh()
                ax.xaxis.set_tick_params(rotation=45)
                ax.xaxis.set_label_text("Component value")
                plt.tight_layout()
                ax.figure.savefig(
                    os.path.join(
                        state.filesystem.output_dir,
                        "calibration",
                        f"{component}_final_components_set_{coef_set}.png",
                    )
                )

                _ = plt.subplots()

                pct_diff = (
                    last_records.diff(axis=1).model_value / last_records.target_value
                )
                ax = pct_diff.plot.barh()
                ax.xaxis.set_tick_params(rotation=45)
                ax.xaxis.set_label_text("Coefficient")
                ax.yaxis.set_label_text("% Change")
                plt.tight_layout()
                ax.figure.savefig(
                    os.path.join(
                        state.filesystem.output_dir,
                        "calibration",
                        f"{component}_final_pct_change_set_{coef_set}.png",
                    )
                )

        return CalibrationRunResult(
            converged=False,
            completed_global_iterations=calibration_settings.run.global_iterations,
        )
    finally:
        state.filesystem.pipeline_file_name = original_pipeline_name


def _run_precursor_components(
    state: workflow.State,
    models: list[str],
    resume_after: str,
    global_iter: int,
    memory_sidecar_process=None,
) -> None:
    """Run the normal ActivitySim model flow for one global calibration iteration."""

    assert (resume_after is None) or (
        resume_after in models
    ), f"resume_after step {resume_after} not in models preceding calibration models"
    if global_iter > 1:
        # Seed a fresh pipeline from the configured resume checkpoint to avoid
        # duplicate checkpoint-name collisions across global calibration loops.
        prior_pipeline = state.checkpoint.store.filename
        state.checkpoint.close_store()
        state.filesystem.pipeline_file_name = f"pipeline_calibration_iter_{global_iter}"
        state.checkpoint.restore_from(prior_pipeline, checkpoint_name=resume_after)
    else:

        _run_in_configured_mode(
            state,
            models=models,
            resume_after=resume_after,
            memory_sidecar_process=memory_sidecar_process,
        )


def _run_intermediate_components(
    state: workflow.State,
    models: list[str],
    resume_after: str,
    memory_sidecar_process=None,
) -> None:
    # don't modify the pipeline, just run the models needed
    _run_in_configured_mode(
        state,
        models=models,
        resume_after=resume_after,
        memory_sidecar_process=memory_sidecar_process,
    )


def _run_subsequent_components(
    state: workflow.State,
    models: list[str],
    resume_after: str,
    memory_sidecar_process=None,
) -> None:
    # don't modify the pipeline, just run the models needed
    _run_in_configured_mode(
        state,
        models=models,
        resume_after=resume_after,
        memory_sidecar_process=memory_sidecar_process,
    )


def _calibrate_component(
    state: workflow.State,
    component_name: str,
    component_settings: CalibrationComponentSettings,
    prior_step: str,
    global_iter: int,
) -> CalibrationComponentResult:
    """Run iterative coefficient calibration for one component."""
    model_settings_file = _infer_model_settings_file(component_name)
    model_settings = state.filesystem.read_model_settings(
        model_settings_file, mandatory=True
    )

    coefficients_df = state.filesystem.read_model_coefficients(
        model_settings=model_settings
    )
    helper_symbols, bespoke_callable = _load_helper_symbols(
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

    for component_iter in range(1, component_settings.submodel_max_iterations + 1):
        component_iterations = component_iter

        # Re-run only this component from its prior checkpoint so model values
        # reflect the current candidate coefficients for this component.
        if state.settings.multiprocess:
            # In multiprocess mode, preserve the standard multiprocess orchestration
            # so table coalescing semantics match the initial global run path.
            _run_in_configured_mode(
                state,
                models=state.settings.models,
                resume_after=prior_step,
            )
        else:
            run_model_name = (
                f"{component_name}.calibration_component_iter={component_iter};"
                f"calibration_global_iter={global_iter}"
            )
            state.run(models=[run_model_name], resume_after=prior_step)

        eval_context = _build_expression_context(state, helper_symbols)

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
        )

        coefficients_df = new_coefficients_df

        _persist_coefficients_to_config(state, model_settings, coefficients_df)
        _append_iteration_records(state, row_records)
        _append_summary_records(state, [summary_record])

        if component_settings.reports.generic:
            _write_generic_report(state, component_name, row_records)

        if bespoke_callable is not None:
            # Preserve compatibility with helper modules that expect a global
            # `state` symbol and/or no explicit arguments.
            kwargs = {"state": state, "component_settings": component_settings}
            bespoke_callable(**kwargs)

        if component_converged:
            break
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
    Extract coefficient tokens from configured utility spec files.

    The extraction scans all settings keys ending with "SPEC" and parses
    tokens from utility columns (all non-description/expression columns).
    """
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


def _build_expression_context(
    state: workflow.State,
    helper_symbols: dict[str, Any],
) -> dict[str, Any]:
    """Create the evaluation context for model_value and target_value expressions."""
    context: dict[str, Any] = {
        "state": state,
        "np": np,
        "pd": pd,
    }

    # Load active tables into context for direct expression access.
    for table_name in list(state.existing_table_names):
        try:
            context[table_name] = state.get_dataframe(table_name, as_copy=False)
        except Exception:
            # Some entries may not be available as dataframes in all contexts.
            continue

    context.update(helper_symbols)
    return context


def _evaluate_and_update(
    component_name: str,
    calibration_spec_df: pd.DataFrame,
    coefficients_df: pd.DataFrame,
    eval_context: dict[str, Any],
    global_iter: int,
    component_iter: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], pd.DataFrame, bool]:
    """Evaluate spec rows, update coefficients, and return detailed records."""
    updated = coefficients_df.copy()
    records: list[dict[str, Any]] = []

    max_difference = -math.inf
    max_difference_component = ""
    max_difference_coefficient = ""
    max_change = -math.inf
    max_change_component = ""
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

        candidate_value = prev_value if hold_fast else prev_value + raw_delta

        under_min = False
        over_max = False

        lower = row["min"]
        upper = row["max"]

        if not pd.isna(lower) and candidate_value < float(lower):
            candidate_value = float(lower)
            under_min = True
        if not pd.isna(upper) and candidate_value > float(upper):
            candidate_value = float(upper)
            over_max = True

        if not np.isfinite(candidate_value):
            raise RuntimeError(
                f"non-finite next coefficient for {component_name} / {description} / {coefficient_name}"
            )

        updated.loc[coefficient_name, "value"] = candidate_value

        abs_diff = abs(difference)
        abs_change = abs(candidate_value - prev_value)

        if abs_diff > max_difference:
            max_difference = abs_diff
            max_difference_component = component_name
            max_difference_coefficient = coefficient_name

        if abs_change > max_change:
            max_change = abs_change
            max_change_component = component_name
            max_change_coefficient = coefficient_name

        if converged:
            num_converged += 1

        records.append(
            {
                "global_iter": global_iter,
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
                "under_min": under_min,
                "over_max": over_max,
            }
        )

    total_rows = len(calibration_spec_df)
    num_unconverged = total_rows - num_converged
    component_converged = num_unconverged == 0

    summary_record = {
        "global_iter": global_iter,
        "component_iter": component_iter,
        "component": component_name,
        "max_difference": max_difference if max_difference != -math.inf else 0.0,
        "max_difference_component": max_difference_component,
        "max_difference_coefficient": max_difference_coefficient,
        "max_change": max_change if max_change != -math.inf else 0.0,
        "max_change_component": max_change_component,
        "max_change_coefficient": max_change_coefficient,
        "num_converged_iter": num_converged,
        "tot_converged": num_converged,
        "num_unconverged": num_unconverged,
    }

    return records, summary_record, updated, component_converged


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


def _safe_percent_difference(difference: float, target_value: float) -> float:
    """Return a stable percentage difference with zero-target handling."""
    if target_value == 0:
        return math.inf if difference != 0 else 0.0
    return (difference / target_value) * 100.0


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


def _append_iteration_records(
    state: workflow.State, records: list[dict[str, Any]]
) -> None:
    """Append per-coefficient calibration iteration records."""
    if not records:
        return
    path = state.get_output_file_path(CALIBRATION_ITERATION_FILE)
    df = pd.DataFrame(records)
    _append_csv(df, path)


def _append_summary_records(
    state: workflow.State, records: list[dict[str, Any]]
) -> None:
    """Append per-iteration summary records."""
    if not records:
        return
    path = state.get_output_file_path(CALIBRATION_SUMMARY_FILE)
    df = pd.DataFrame(records)
    _append_csv(df, path)


def _append_csv(df: pd.DataFrame, path: Path) -> None:
    """Append a dataframe to a CSV file with header-once behavior."""
    os.makedirs(path.parent, exist_ok=True)
    write_header = not path.exists()
    df.to_csv(path, mode="a", index=False, header=write_header)


def _write_generic_report(
    state: workflow.State,
    component_name: str,
    row_records: list[dict[str, Any]],
) -> None:
    """Write a simple dashboard-friendly generic report for the current component iteration."""
    if not row_records:
        return

    df = pd.DataFrame(row_records)
    report = (
        df[
            [
                "global_iter",
                "component_iter",
                "component",
                "description",
                "difference",
                "pct_difference",
                "converged",
            ]
        ]
        .copy()
        .sort_values(["global_iter", "component_iter", "description"])
    )

    path = state.get_output_file_path(
        f"calibration/{component_name}_generic_report.csv"
    )
    _append_csv(report, path)


def _load_helper_symbols(
    state: workflow.State,
    component_settings: CalibrationComponentSettings,
) -> tuple[dict[str, Any], Any | None]:
    """Load helper module and return evaluation symbols and bespoke function."""
    if not component_settings.helper_module:
        return {}, None

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

    return symbols, bespoke


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


def _prior_step_name(models: list[str], component_name: str) -> str | None:
    """Return the step name immediately preceding component_name in models."""
    if component_name not in models:
        return None
    idx = models.index(component_name)
    if idx == 0:
        return None
    return models[idx - 1]


def _write_final_coefficients_snapshot(
    state: workflow.State,
    calibration_settings: CalibrationConfig,
) -> None:
    """Write a combined final coefficients file snapshot for calibrated components."""
    frames = []
    for component_name in calibration_settings.run.calibrate_models:
        model_settings_file = _infer_model_settings_file(component_name)
        model_settings = state.filesystem.read_model_settings(
            model_settings_file, mandatory=True
        )
        coeff_df = state.filesystem.read_model_coefficients(
            model_settings=model_settings
        ).copy()
        coeff_df = coeff_df.reset_index().rename(columns={"index": "coefficient_name"})
        coeff_df.insert(0, "component", component_name)
        frames.append(coeff_df)

    if not frames:
        return

    final_df = pd.concat(frames, ignore_index=True)
    path = state.get_output_file_path(CALIBRATION_FINAL_COEFFICIENTS_FILE)
    os.makedirs(path.parent, exist_ok=True)
    final_df.to_csv(path, index=False)


def _ensure_calibration_output_dir(state: workflow.State) -> None:
    """Ensure output/calibration exists."""
    path = state.get_output_file_path(CALIBRATION_OUTPUT_DIR)
    os.makedirs(path, exist_ok=True)


def _read_progress(state: workflow.State) -> dict[str, Any] | None:
    """Read persisted calibration progress metadata if it exists."""
    path = state.get_output_file_path(CALIBRATION_PROGRESS_FILE)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_progress(state: workflow.State, payload: dict[str, Any]) -> None:
    """Write calibration progress metadata."""
    path = state.get_output_file_path(CALIBRATION_PROGRESS_FILE)
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _run_in_configured_mode(
    state: workflow.State,
    models: list[str],
    resume_after: str | None,
    memory_sidecar_process=None,
) -> None:
    """Run models using the same single/multiprocess mode as the parent run."""
    if state.settings.multiprocess:
        _run_multiprocess_with_overrides(
            state,
            models=models,
            resume_after=resume_after,
        )
        return

    state.run(
        models=models,
        resume_after=resume_after,
        memory_sidecar_process=memory_sidecar_process,
    )


def _run_multiprocess_with_overrides(
    state: workflow.State,
    models: list[str],
    resume_after: str | None,
) -> None:
    """Run multiprocess with temporary settings overrides for calibration passes."""
    from activitysim.core import mp_tasks

    original_models = state.settings.models
    original_resume_after = state.settings.resume_after

    state.settings.models = models
    state.settings.resume_after = resume_after

    try:
        injectables = {}
        for key in MP_INJECTABLES:
            try:
                injectables[key] = state.get_injectable(key)
            except KeyError:
                pass
        injectables["settings"] = state.settings
        mp_tasks.run_multiprocess(state, injectables)
    finally:
        state.settings.models = original_models
        state.settings.resume_after = original_resume_after
