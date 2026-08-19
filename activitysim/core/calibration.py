# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import math
import multiprocessing
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pydantic import model_validator

from activitysim.core import simulate, workflow
from activitysim.core.configuration import PydanticReadable
from activitysim.core.configuration.base import PydanticBase
from activitysim.core.configuration.top import MultiprocessStep

logger = logging.getLogger("calibration")

plt.style.use("seaborn-v0_8-darkgrid")
matplotlib.use("Agg")  # Forces non-interactive background rendering

CALIBRATION_SETTINGS_FILE_NAME = "calibration.yaml"
CALIBRATION_OUTPUT_DIR = "calibration"
CALIBRATION_PROGRESS_FILE = "calibration/calibration_progress.json"
CALIBRATION_ITERATION_FILE = "calibration/calibration_iteration_records.csv"
CALIBRATION_SUMMARY_FILE = "calibration/calibration_iteration_summary.csv"
CALIBRATION_FINAL_COEFFICIENTS_FILE = "calibration/final_calibrated_coefficients.csv"
CALIBRATION_RECOVERY_DIR = "calibration/recovery"

DEFAULT_INCREMENT = 2.0
MAX_COEFFS_IN_GRAPH = 15

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
    "pipeline_file_name",
]


class CalibrationRunSettings(PydanticBase):
    """Run-control settings for calibration."""

    resume_after: Optional[str] = None
    calibrate_models: list[str]
    global_iterations: int = 1
    complete_steps: bool = False
    invalidate_tables: list[str] | None = None

    @model_validator(mode="after")
    def validate_run_settings(self):
        if not self.calibrate_models:
            raise ValueError(
                "calibration.run.calibrate_models must contain at least one model name"
            )
        return self

    """Tables to drop from state after each calibration restore so their
    ``@workflow.table`` factories regenerate from current data.

    Default (None): invalidates ``["vehicles"]``.  Set to ``[]`` to disable.

    A table should be listed here when ALL of the following are true:

    1. It is created by a ``@workflow.table`` factory from another table's
       values (not just from input data files).
    2. That source table is modified by a calibrated model or by a model
       whose outputs change when calibrated coefficients change.
    3. The factory uses source-table values to determine **row identity**
       (index values) or **row count**, not just column values.

    The canonical example is ``vehicles``: its factory repeats household
    rows by ``households["auto_ownership"]`` and derives ``vehicle_id``
    from ``household_id``.  When ``auto_ownership_simulate`` is calibrated,
    different coefficients produce different ownership counts, so the
    stale vehicles table loaded from a prior checkpoint would have the
    wrong number of rows and wrong vehicle IDs.  Dropping it forces the
    factory to regenerate vehicles consistent with the current households.

    Tables that only read *column values* from upstream tables (without
    affecting row identity) generally do NOT need invalidation — their
    content will be correct as long as the upstream table is correct at
    the restored checkpoint.
    """


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
    survey_file: Optional[str] = None


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

    if state.settings.duplicate_step_execution != "allow":
        state.settings.duplicate_step_execution = "allow"
        logger.warning(
            "Overriding duplicate_step_execution setting: must be enabled for calibration"
        )

    if not calibration_settings.run.calibrate_models:
        raise ValueError(
            "calibration.run.calibrate_models must contain at least one model name"
        )

    missing_calibration_models = [
        component
        for component in calibration_settings.run.calibrate_models
        if component not in models
    ]
    if missing_calibration_models:
        raise ValueError(
            "settings.yaml models list does not include configured calibration "
            f"model(s): {missing_calibration_models}"
        )

    resume_after = state.settings.resume_after
    if resume_after is not None and resume_after not in models:
        raise ValueError(
            f"settings.yaml resume_after={resume_after!r} is not present in the "
            "settings.yaml models list. Calibration requires resume_after to be "
            "a model-level checkpoint name."
        )

    # sort calibration models into main model order
    calibration_settings.run.calibrate_models = sorted(
        calibration_settings.run.calibrate_models, key=lambda x: models.index(x)
    )
    first_calib_model_idx = models.index(calibration_settings.run.calibrate_models[0])
    last_calib_model_idx = models.index(calibration_settings.run.calibrate_models[-1])
    first_model_idx = models.index(resume_after) + 1 if resume_after else None
    first_calibration_restart_step = _prior_step_name(
        models, calibration_settings.run.calibrate_models[0]
    )

    if resume_after is not None:
        skipped_calibration_models = [
            component
            for component in calibration_settings.run.calibrate_models
            if models.index(component) <= models.index(resume_after)
        ]
        if skipped_calibration_models:
            logger.warning(
                "Calibration is honoring settings.yaml resume_after=%r using strict "
                "ActivitySim semantics. The following calibrated model(s) occur at "
                "or before resume_after and will be skipped during the first global "
                "iteration: %s",
                resume_after,
                skipped_calibration_models,
            )

    _ensure_calibration_output_dir(state)

    progress = _read_progress(state)
    if progress and progress.get("complete"):
        logger.info(
            "calibration progress is already complete; remove %s to start a "
            "fresh calibration run",
            CALIBRATION_PROGRESS_FILE,
        )
        return CalibrationRunResult(
            converged=bool(progress.get("converged", False)),
            completed_global_iterations=int(
                progress.get(
                    "last_completed_global_iteration",
                    calibration_settings.run.global_iterations,
                )
            ),
        )

    interrupted_iteration = (
        progress.get("in_progress_iteration") if progress else None
    )
    if interrupted_iteration is not None:
        interrupted_iteration = int(interrupted_iteration)
        logger.warning(
            "recovering interrupted calibration global iteration %s",
            interrupted_iteration,
        )
        _restore_coefficient_backups(state, calibration_settings)
        progress = {
            "in_progress_iteration": None,
            "next_global_iteration": interrupted_iteration,
            "last_completed_global_iteration": interrupted_iteration - 1,
        }
        _write_progress(state, progress)

    # Progress files from earlier versions contain next_global_iteration, so
    # they remain compatible with the corrected total-count semantics.
    start_global_iter = int(progress.get("next_global_iteration", 1)) if progress else 1
    completed_global_iterations = start_global_iter - 1

    if start_global_iter > calibration_settings.run.global_iterations:
        logger.info(
            "calibration progress already reached configured global_iterations=%s",
            calibration_settings.run.global_iterations,
        )
        converged = bool(progress.get("converged", False)) if progress else False
        _write_final_coefficients_snapshot(state, calibration_settings)
        _write_completed_progress(
            state,
            completed_global_iterations,
            converged,
        )
        return CalibrationRunResult(
            converged=converged,
            completed_global_iterations=completed_global_iterations,
        )

    if state.settings.resume_after is None:
        # compute_accessibility requires its accessibility table to be empty;
        # unlike most model steps, it will not overwrite a prior result.
        # Remove a cached result before restore clears table-status metadata,
        # so the table factory recreates its empty placeholder for the replay.
        state.drop_table("accessibility")
        state.checkpoint.restore()

    original_pipeline_name = state.filesystem.pipeline_file_name

    # Initialize shared resources for multiprocess mode (skims, shadow pricing).
    # These are allocated once and reused across all calibration iterations.
    shared_data_buffers = None
    if state.settings.multiprocess:
        shared_data_buffers = _initialize_mp_shared_resources(state)

    try:
        # skip precursors if, on first iter, resume_after exists and is >= first_calib_model_idx
        if (
            state.settings.resume_after is None
            or first_model_idx < first_calib_model_idx
        ):
            # Run ActivitySim normally from resume_after through production model steps.
            _run_precursor_components(
                state,
                models=models[:first_calib_model_idx]
                if first_model_idx is None
                else models[first_model_idx:first_calib_model_idx],
                resume_after=state.settings.resume_after,
                global_iter=start_global_iter,
                shared_data_buffers=shared_data_buffers,
            )
        else:
            # Precursors skipped — but the pipeline must still be initialized
            # at the resume_after point so that _calibrate_component (and its
            # apportion subprocess) starts from the correct state without
            # downstream model data.
            extra_models = _prep_model_data(
                state, resume_after=state.settings.resume_after
            )
            if extra_models:
                # No model-level checkpoint exists for resume_after; we must
                # run models from the prior step through resume_after to
                # recreate the correct intermediate state.
                _run_in_configured_mode(
                    state,
                    models=extra_models,
                    resume_after=None,
                    shared_data_buffers=shared_data_buffers,
                )
            elif not any(
                cp.get("checkpoint_name") == state.settings.resume_after
                for cp in state.checkpoint.checkpoints
            ):
                # _prep_model_data took its fallback path — the pipeline either
                # doesn't exist or doesn't contain resume_after's checkpoint.
                # The restored state is incomplete (precursor models never ran).
                logger.warning(
                    "calibration: resume_after=%r not found in restored pipeline; "
                    "running precursor models",
                    state.settings.resume_after,
                )
                _run_precursor_components(
                    state,
                    models=models[:first_calib_model_idx],
                    resume_after=None,
                    global_iter=start_global_iter,
                    shared_data_buffers=shared_data_buffers,
                )
            else:
                state.checkpoint.add(state.settings.resume_after)
                state.checkpoint.close_store()

        for global_iter in range(
            start_global_iter,
            calibration_settings.run.global_iterations + 1,
        ):
            _begin_global_iteration_transaction(
                state,
                calibration_settings,
                global_iter,
            )

            # Every global iteration after the first begins from the immutable
            # checkpoint directly before the first calibrated model. This makes
            # global reruns independent of the state left by the final calibrated
            # model in the preceding iteration.
            if global_iter > start_global_iter:
                logger.info(
                    "Restarting global calibration iteration %s from checkpoint %r",
                    global_iter,
                    first_calibration_restart_step,
                )
                extra_models = _prep_model_data(
                    state, resume_after=first_calibration_restart_step
                )
                if extra_models:
                    _run_in_configured_mode(
                        state,
                        models=extra_models,
                        resume_after=None,
                        shared_data_buffers=shared_data_buffers,
                    )
                _invalidate_derived_tables(state)
                if first_calibration_restart_step is not None:
                    state.checkpoint.add(first_calibration_restart_step)
                    state.checkpoint.close_store()

            logger.info(
                "calibration global iteration %s/%s",
                global_iter,
                calibration_settings.run.global_iterations,
            )

            # suppress early termination on first iteration if resume_after is after all calibrated models
            all_converged = (
                first_model_idx is not None and first_model_idx <= last_calib_model_idx
            ) or global_iter > start_global_iter

            last_calibrated_component = None
            for component in calibration_settings.run.calibrate_models:
                # on the first global iter, skip model if it's before or == resume_after
                if (
                    global_iter == start_global_iter
                    and first_model_idx is not None
                    and first_model_idx > models.index(component)
                ):
                    continue
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
                        shared_data_buffers=shared_data_buffers,
                    )

                component_result = _calibrate_component(
                    state=state,
                    component_name=component,
                    component_settings=component_settings,
                    prior_step=prior_step,
                    global_iter=global_iter,
                    shared_data_buffers=shared_data_buffers,
                )
                _write_component_plots(state, component)

                all_converged = all_converged and component_result.converged

                last_calibrated_component = component

            if (
                calibration_settings.run.complete_steps
                or global_iter == calibration_settings.run.global_iterations
                or (
                    global_iter == start_global_iter
                    and state.settings.resume_after is not None
                    and first_model_idx > last_calib_model_idx
                )
            ):
                subsequent_components = (
                    models[first_model_idx:]
                    if global_iter == start_global_iter
                    and first_model_idx > last_calib_model_idx
                    else models[models.index(last_calibrated_component) + 1 :]
                )
                # finish the full model chain
                _run_subsequent_components(
                    state,
                    models=subsequent_components,
                    resume_after=state.settings.resume_after
                    if global_iter == start_global_iter
                    and first_model_idx > last_calib_model_idx
                    else last_calibrated_component,
                    shared_data_buffers=shared_data_buffers,
                )

            completed_global_iterations = global_iter
            iteration_is_complete = (
                all_converged
                or global_iter == calibration_settings.run.global_iterations
            )
            if not iteration_is_complete:
                _write_progress(
                    state,
                    {
                        "in_progress_iteration": None,
                        "next_global_iteration": global_iter + 1,
                        "last_completed_global_iteration": global_iter,
                        "converged": all_converged,
                    },
                )

            if all_converged:
                logger.info(
                    "calibration converged after global iteration %s/%s",
                    global_iter,
                    calibration_settings.run.global_iterations,
                )
                break

        _write_final_coefficients_snapshot(state, calibration_settings)
        _write_completed_progress(
            state,
            completed_global_iterations,
            all_converged,
        )

        return CalibrationRunResult(
            converged=all_converged,
            completed_global_iterations=completed_global_iterations,
        )
    finally:
        state.filesystem.pipeline_file_name = original_pipeline_name


def _run_precursor_components(
    state: workflow.State,
    models: list[str],
    resume_after: str,
    global_iter: int,
    shared_data_buffers: dict | None = None,
) -> None:
    """Run the normal ActivitySim model flow for one global calibration iteration."""

    # if global_iter > 1 and resume_after is not None:
    #     # Seed a fresh pipeline from the configured resume checkpoint to avoid
    #     # duplicate checkpoint-name collisions across global calibration loops.
    #     prior_pipeline = state.checkpoint.store.filename
    #     state.checkpoint.close_store()
    #     state.filesystem.pipeline_file_name = f"pipeline_calibration_iter_{global_iter}"
    #     state.checkpoint.restore_from(prior_pipeline, checkpoint_name=resume_after)
    # else:

    _run_in_configured_mode(
        state,
        models=models,
        resume_after=resume_after,
        shared_data_buffers=shared_data_buffers,
    )


def _run_intermediate_components(
    state: workflow.State,
    models: list[str],
    resume_after: str,
    shared_data_buffers: dict | None = None,
) -> None:
    if len(models) == 0:
        return
    _run_in_configured_mode(
        state,
        models=models,
        resume_after=resume_after,
        shared_data_buffers=shared_data_buffers,
    )


def _run_subsequent_components(
    state: workflow.State,
    models: list[str],
    resume_after: str,
    shared_data_buffers: dict | None = None,
) -> None:
    _run_in_configured_mode(
        state,
        models=models,
        resume_after=resume_after,
        shared_data_buffers=shared_data_buffers,
    )


def _calibrate_component(
    state: workflow.State,
    component_name: str,
    component_settings: CalibrationComponentSettings,
    prior_step: str,
    global_iter: int,
    shared_data_buffers: dict | None = None,
) -> CalibrationComponentResult:
    """Run iterative coefficient calibration for one component."""
    model_settings_file = _infer_model_settings_file(component_name)
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
        run_model_name = f"{component_name}.c_i{component_iter};" f"g_i{global_iter}"
        # Re-run only this component from its prior checkpoint so model values
        # reflect the current candidate coefficients for this component.
        if state.settings.multiprocess and shared_data_buffers is not None:
            # Use direct MP orchestration with explicit checkpoint control.
            # This ensures we always apportion from prior_step's state,
            # even after multiple component iterations.
            _run_mp_single_component(
                state,
                component_name=component_name,
                # Always restore from the same immutable pre-component
                # checkpoint. LAST_CHECKPOINT may point to the prior iteration's
                # coalesced component output and is therefore not a safe baseline.
                restore_checkpoint=mp_restore_checkpoint,
                shared_data_buffers=shared_data_buffers,
            )
        else:
            # Restore to prior_step ourselves then run the model directly.
            # state.run(resume_after=prior_step) would trigger
            # checkpoint.restore → init_state which creates a fresh RNG.
            # If prior_step is before the calibrated model created its table
            # (e.g. vehicles), the table won't be in that checkpoint and the
            # RNG channel won't be registered — causing a crash when the
            # model tries to use it.  By restoring here and calling by_name,
            # we keep the RNG channels from _prep_model_data intact.
            extra_models = _prep_model_data(state, resume_after=prior_step)
            if extra_models:
                # prior_step checkpoint not found directly; run intermediate
                # models (e.g. annotators) to recreate the correct state.
                for m in extra_models:
                    state.run.by_name(m)
            _invalidate_derived_tables(state)
            state.checkpoint.add(prior_step)
            state.run.by_name(run_model_name)

        eval_context = _build_expression_context(
            state, helper_symbols, component_name, component_settings
        )

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
        _append_iteration_records(state, component_name, row_records)
        _append_summary_records(state, [summary_record])

        if component_settings.reports.generic:
            _write_generic_report(state, component_name, row_records)

        if bespoke_callable is not None:
            bespoke_callable(eval_context)

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

        candidate_value = prev_value if hold_fast else prev_value + raw_delta

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
    state: workflow.State, component_name: str, records: list[dict[str, Any]]
) -> None:
    """Append per-coefficient calibration iteration records."""
    if not records:
        return
    df = pd.DataFrame(records)

    # Save a global iteration history file
    global_path = state.get_output_file_path(CALIBRATION_ITERATION_FILE)
    _append_csv(
        df,
        global_path,
        unique_on=["global_iter", "component_iter", "component", "coefficient"],
    )

    # Also write component-local iteration history
    component_path = (
        _component_output_dir(state, component_name)
        / Path(CALIBRATION_ITERATION_FILE).name
    )
    _append_csv(
        df,
        component_path,
        unique_on=["global_iter", "component_iter", "component", "coefficient"],
    )


def _append_summary_records(
    state: workflow.State, records: list[dict[str, Any]]
) -> None:
    """Append per-iteration summary records."""
    if not records:
        return
    path = state.get_output_file_path(CALIBRATION_SUMMARY_FILE)
    df = pd.DataFrame(records)
    _append_csv(
        df,
        path,
        unique_on=["global_iter", "component_iter", "component"],
    )


def _append_csv(
    df: pd.DataFrame, path: Path, unique_on: list[str] | None = None
) -> None:
    """Append a dataframe to a CSV file, replacing rows with matching keys."""
    os.makedirs(path.parent, exist_ok=True)
    if unique_on and path.exists():
        existing = pd.read_csv(path)
        df = pd.concat([existing, df], ignore_index=True).drop_duplicates(
            subset=unique_on, keep="last"
        )
        df.to_csv(path, index=False)
        return

    write_header = not path.exists()
    df.to_csv(path, mode="a", index=False, header=write_header)


def _component_output_dir(state: workflow.State, component_name: str) -> Path:
    """Return output/calibration/<component_name> and ensure it exists."""
    component_dir = state.get_output_file_path(f"calibration/{component_name}")
    os.makedirs(component_dir, exist_ok=True)
    return component_dir


def _write_component_plots(state: workflow.State, component_name: str) -> None:
    """Write/update all standard plots for one calibrated component."""
    recs = _read_component_iteration_records(state, component_name)
    if recs is None or recs.empty:
        return

    # Segment coefficients into manageable sets for plotting
    coefs = sorted(recs.index.get_level_values("coefficient").unique())
    n_sets = math.ceil(len(coefs) / MAX_COEFFS_IN_GRAPH)
    for coef_set in range(n_sets):
        set_coefs = coefs[
            coef_set
            * MAX_COEFFS_IN_GRAPH : min(
                len(coefs), (coef_set + 1) * MAX_COEFFS_IN_GRAPH
            )
        ]
        _plot_coefficient_progress(state, component_name, recs, set_coefs, coef_set)
        last_records = _component_last_records(recs, set_coefs)
        _plot_component_values(state, component_name, last_records, coef_set)
        _plot_component_pct_change(state, component_name, last_records, coef_set)


def _read_component_iteration_records(
    state: workflow.State, component_name: str
) -> pd.DataFrame | None:
    """Read all iteration records for a single component."""
    path = state.get_output_file_path(CALIBRATION_ITERATION_FILE)
    if not path.exists():
        return None

    iteration_records = (
        pd.read_csv(path)
        .set_index(["global_iter", "component_iter", "coefficient"])
        .sort_index()
    )
    return iteration_records.loc[iteration_records.component == component_name]


def _plot_coefficient_progress(
    state: workflow.State,
    component_name: str,
    recs: pd.DataFrame,
    set_coefs: list[str],
    coef_set: int,
) -> None:
    """Plot coefficient value progression for one coefficient subset."""
    component_dir = _component_output_dir(state, component_name)
    ax = (
        recs[recs.index.get_level_values("coefficient").isin(set_coefs)]
        .next_coefficient.unstack("coefficient")
        .plot(figsize=(10, 5))
    )
    ax.xaxis.set_label_text("Component iteration")
    ax.yaxis.set_label_text("Coefficient value")
    ax.legend(title="Coefficient label", loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    ax.figure.savefig(
        component_dir / f"coefficient_progress_set_{coef_set}.png",
        bbox_inches="tight",
    )
    plt.close(ax.figure)


def _component_last_records(recs: pd.DataFrame, set_coefs: list[str]) -> pd.DataFrame:
    """Select target/model values for the latest iteration and coefficient subset."""
    filtered = recs[recs.index.get_level_values("coefficient").isin(set_coefs)]
    last_global = filtered.index.get_level_values("global_iter")[-1]
    last_comp = filtered.loc[last_global].index.get_level_values("component_iter")[-1]
    return filtered.xs(
        (last_global, last_comp), level=("global_iter", "component_iter")
    )[["target_value", "model_value"]]


def _plot_component_values(
    state: workflow.State,
    component_name: str,
    last_records: pd.DataFrame,
    coef_set: int,
) -> None:
    """Plot final target/model component values for one coefficient subset."""
    component_dir = _component_output_dir(state, component_name)
    ax = last_records.plot.bar(figsize=(10, 5))
    ax.xaxis.set_tick_params(rotation=45)
    ax.xaxis.set_label_text("Component value")
    plt.tight_layout()
    ax.figure.savefig(component_dir / f"final_components_set_{coef_set}.png")
    plt.close(ax.figure)


def _plot_component_pct_change(
    state: workflow.State,
    component_name: str,
    last_records: pd.DataFrame,
    coef_set: int,
) -> None:
    """Plot final percent difference for one coefficient subset."""
    component_dir = _component_output_dir(state, component_name)
    fig, ax = plt.subplots(figsize=(10, 5))
    pct_diff = last_records.diff(axis=1).model_value / last_records.target_value
    ax = pct_diff.plot.bar(ax=ax)
    ax.xaxis.set_tick_params(rotation=45)
    ax.xaxis.set_label_text("Coefficient")
    ax.yaxis.set_label_text("% Difference between Model and Target")
    plt.tight_layout()
    ax.figure.savefig(component_dir / f"final_pct_change_set_{coef_set}.png")
    plt.close(ax.figure)


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

    path = _component_output_dir(state, component_name) / "generic_report.csv"
    _append_csv(
        report,
        path,
        unique_on=["global_iter", "component_iter", "component", "description"],
    )


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


def _run_mp_single_component(
    state: workflow.State,
    component_name: str,
    restore_checkpoint: str,
    shared_data_buffers: dict,
) -> None:
    """Run a single component in multiprocess mode with explicit checkpoint control.

    This directly orchestrates the apportion → simulate → coalesce flow
    without going through run_multiprocess/get_run_list, giving us precise
    control over which checkpoint to apportion from. This is essential for
    calibration component re-runs where we must always restart from prior_step's
    state regardless of what other checkpoints exist in the pipeline.

    Parameters
    ----------
    state : workflow.State
    component_name : str
        The model component to run.
    restore_checkpoint : str
        The checkpoint name to restore from before apportioning.
        This should be the checkpoint representing prior_step's state.
    shared_data_buffers : dict
        Pre-allocated shared memory buffers for skims/shadow pricing.
    """
    from activitysim.core import mp_tasks

    # Determine slice info from original settings
    original_steps = state.settings.multiprocess_steps
    all_models = state.settings.models

    slice_info = None
    num_processes = state.settings.num_processes or 2
    chunk_size = state.settings.chunk_size or 0

    # Find which original step this component belongs to
    step_boundaries = []
    for i, step in enumerate(original_steps):
        step_boundaries.append(all_models.index(step.begin))
    step_boundaries.append(len(all_models))

    component_idx = all_models.index(component_name)
    for i, step in enumerate(original_steps):
        if step_boundaries[i] <= component_idx < step_boundaries[i + 1]:
            if step.slice:
                slice_info = step.slice.model_dump()
            if step.num_processes:
                num_processes = step.num_processes
            if step.chunk_size:
                chunk_size = step.chunk_size
            break

    # Build step_info dict matching what mp_tasks functions expect
    step_info = {
        "name": component_name,
        "models": [component_name],
        "num_processes": num_processes,
        "chunk_size": chunk_size,
        "step_num": 0,
        "slice": slice_info,
        "last_checkpoint_in_previous_multiprocess_step": restore_checkpoint,
    }

    injectables = _build_calibration_injectables(state)

    if num_processes == 1:
        sub_proc_names = [component_name]
    else:
        sub_proc_names = [f"{component_name}_{i}" for i in range(num_processes)]

    fail_fast = state.settings.fail_fast

    # Apportion pipeline (split tables across sub-processes)
    if num_processes > 1 and slice_info is not None:
        mp_tasks.run_sub_task(
            state,
            multiprocessing.Process(
                target=mp_tasks.mp_apportion_pipeline,
                name=f"{component_name}_apportion",
                args=(injectables, sub_proc_names, step_info),
            ),
        )

    # For multi-process runs, subprocesses must restore from the apportioned
    # pipeline (which has one checkpoint). Use LAST_CHECKPOINT so they don't
    # overwrite the apportioned data with a fresh pipeline.
    # For single-process runs (no apportion), use restore_checkpoint to resume
    # from the correct point in the main pipeline.
    if num_processes > 1:
        sim_resume_after = "_"  # LAST_CHECKPOINT in apportioned sub-pipeline
    else:
        sim_resume_after = restore_checkpoint

    # Run simulations in sub-processes
    completed = mp_tasks.run_sub_simulations(
        state,
        injectables,
        shared_data_buffers,
        step_info,
        sub_proc_names,
        sim_resume_after,
        [],  # previously_completed
        fail_fast,
    )

    if len(completed) != num_processes:
        from activitysim.core.exceptions import SubprocessError

        raise SubprocessError(
            f"{num_processes - len(completed)} processes failed in "
            f"calibration step {component_name}"
        )

    # Coalesce sub-process pipelines back into main pipeline
    if num_processes > 1 and slice_info is not None:
        mp_tasks.run_sub_task(
            state,
            multiprocessing.Process(
                target=mp_tasks.mp_coalesce_pipelines,
                name=f"{component_name}_coalesce",
                args=(injectables, sub_proc_names, slice_info),
            ),
        )

    # Restore coalesced results into parent state
    _restore_parent_state_from_pipeline(state)


def _run_in_configured_mode(
    state: workflow.State,
    models: list[str],
    resume_after: str | None,
    shared_data_buffers: dict | None = None,
) -> None:
    """Run models using the same single/multiprocess mode as the parent run."""
    if not models:
        return

    extra_models = _prep_model_data(state, resume_after=resume_after)
    if extra_models:
        # Models from the step beginning through resume_after must run first
        # to recreate the correct intermediate state (since no model-level
        # checkpoint existed for resume_after).
        models = extra_models + models

    if state.settings.multiprocess:
        # Write the restored state as a checkpoint so LAST_CHECKPOINT on disk
        # reflects the correct (clean) state for the apportion subprocess.
        # When extra_models were prepended, the state is from a PRIOR step
        # (not the actual resume_after point) — use a non-conflicting name
        # so it won't be mistakenly loaded as the resume_after state on a
        # subsequent restart.
        if extra_models:
            state.checkpoint.add(f"_calibration_staging")
        else:
            state.checkpoint.add(resume_after or models[0])
        state.checkpoint.close_store()

        # When subprocess pipelines from a prior run already have the
        # resume_after checkpoint (Path 2 in _prep_model_data), subprocesses
        # can skip models before resume_after by reusing those pipelines
        # instead of freshly apportioning.  Signal this by passing
        # can_reuse_subprocs=True.
        can_reuse = not extra_models and resume_after is not None

        _run_multiprocess_with_overrides(
            state,
            models=models,
            resume_after=resume_after,
            shared_data_buffers=shared_data_buffers,
            can_reuse_subprocs=can_reuse,
        )
        # After multiprocess completes, the coalesced pipeline exists on disk.
        # Restore it into the parent process state so tables are accessible
        # for calibration expression evaluation.
        _restore_parent_state_from_pipeline(state)
        # Add a checkpoint named after the last model so that model-name
        # references (e.g. _prior_step_name, resume_after on global_iter > 1)
        # resolve correctly. Without this, only the step-level coalesce name
        # exists in the pipeline.
        state.checkpoint.add(models[-1])
        return

    # State is already at the correct point from _prep_model_data above.
    # Do NOT call _prep_model_data again — the second call would build its
    # table_checkpoint_map from the now-truncated in-memory checkpoint history,
    # losing references to tables created after resume_after (e.g. vehicles).
    state.checkpoint.add(resume_after or models[0])
    for model in models:
        state.run.by_name(model)
    # Ensure final model's state is persisted even if should_save_checkpoint
    # returned False for it — _calibrate_component needs to restore to it.
    if models:
        state.checkpoint.add(models[-1])


def _prep_model_data(state, resume_after=None):
    """Restore the pipeline to the correct state before running models.

    Resolution priority:
    1. Direct model-level checkpoint in the main pipeline (fastest, exact).
    2. Subprocess pipelines from a prior multiprocess run — performs a
       "coalesce at specific checkpoint" to recover the exact intermediate
       state without re-running anything.
    3. Previous step checkpoint + re-run models from step begin through
       resume_after (slowest but always works).

    Returns
    -------
    list[str]
        Models that must be prepended to the caller's models list to reach
        the correct state at ``resume_after``.  Empty when the exact
        checkpoint was found and restored directly (paths 1 or 2).
    """
    if resume_after:
        try:
            if state.checkpoint.store_is_open():
                checkpoint_names = [
                    cp.get("checkpoint_name", "") for cp in state.checkpoint.checkpoints
                ]
            else:
                from activitysim.core.workflow.checkpoint import HdfStore, ParquetStore

                pipeline_path = Path(state.checkpoint.default_pipeline_file_path())
                if state.settings.checkpoint_format == "hdf":
                    store = HdfStore(pipeline_path, mode="r")
                else:
                    store = ParquetStore(pipeline_path, mode="r")
                try:
                    checkpoint_names = store.list_checkpoint_names()
                finally:
                    store.close()

            # Path 1: direct model-level checkpoint in main pipeline
            if resume_after in checkpoint_names:
                _restore_parent_state_from_pipeline(state, checkpoint_name=resume_after)
                return []

            # Path 2: subprocess pipelines (model-level checkpoints preserved)
            if _restore_from_subprocess_pipelines(state, resume_after):
                return []

            # Path 3: restore from previous step and re-run
            all_models = state.settings.models
            mp_steps = state.settings.multiprocess_steps
            if mp_steps and resume_after in all_models:
                resume_idx = all_models.index(resume_after)
                step_boundaries = [all_models.index(s.begin) for s in mp_steps]
                step_boundaries.append(len(all_models))
                for i, step in enumerate(mp_steps):
                    if step_boundaries[i] <= resume_idx < step_boundaries[i + 1]:
                        if i > 0 and mp_steps[i - 1].name in checkpoint_names:
                            _restore_parent_state_from_pipeline(
                                state, checkpoint_name=mp_steps[i - 1].name
                            )
                            step_begin_idx = step_boundaries[i]
                            extra_models = all_models[step_begin_idx : resume_idx + 1]
                            return extra_models
                        elif i == 0:
                            _restore_parent_state_from_pipeline(
                                state, checkpoint_name="_"
                            )
                            extra_models = all_models[: resume_idx + 1]
                            return extra_models
                        break
        except Exception:
            logger.warning(
                "calibration: could not restore from checkpoint %r, "
                "falling back to LAST_CHECKPOINT",
                resume_after,
            )

    # Fallback: load LAST_CHECKPOINT (appropriate after a coalesce that
    # only ran the desired models)
    _restore_parent_state_from_pipeline(state)
    return []


def _restore_from_subprocess_pipelines(
    state: workflow.State, resume_after: str
) -> bool:
    """Restore state from subprocess pipelines at a specific model checkpoint.

    Subprocess pipelines retain model-level checkpoints that don't exist in
    the main pipeline.  This performs a "coalesce at checkpoint" — reading
    mirrored tables from one subprocess and concatenating sliced tables from
    all subprocesses at the specified checkpoint name.

    Parameters
    ----------
    state : workflow.State
    resume_after : str
        Model-level checkpoint name to restore from.

    Returns
    -------
    bool
        True if the restore succeeded; False if subprocess pipelines don't
        exist or don't contain the requested checkpoint.
    """
    from activitysim.core.workflow.checkpoint import (
        CHECKPOINT_NAME,
        CHECKPOINT_TABLE_NAME,
        HdfStore,
        NON_TABLE_COLUMNS,
        ParquetStore,
    )

    all_models = state.settings.models
    mp_steps = state.settings.multiprocess_steps
    if not mp_steps or resume_after not in all_models:
        return False

    # Find the multiprocess step containing resume_after
    resume_idx = all_models.index(resume_after)
    step_boundaries = [all_models.index(s.begin) for s in mp_steps]
    step_boundaries.append(len(all_models))

    enclosing_step = None
    num_processes = state.settings.num_processes or 2
    slice_info = None
    for i, step in enumerate(mp_steps):
        if step_boundaries[i] <= resume_idx < step_boundaries[i + 1]:
            enclosing_step = step
            if step.num_processes:
                num_processes = step.num_processes
            if step.slice:
                slice_info = (
                    step.slice.model_dump()
                    if hasattr(step.slice, "model_dump")
                    else step.slice
                )
            break

    if enclosing_step is None or num_processes <= 1:
        return False

    # Build subprocess pipeline file paths
    step_name = enclosing_step.name
    pipeline_file_name = state.filesystem.pipeline_file_name
    sub_proc_names = [f"{step_name}_{i}" for i in range(num_processes)]

    def _subprocess_path(proc_name):
        base = state.get_output_file_path(pipeline_file_name, prefix=proc_name)
        if state.settings.checkpoint_format == "hdf":
            return base
        pq = Path(str(base)).with_suffix(ParquetStore.extension)
        return pq if pq.exists() else base

    first_path = _subprocess_path(sub_proc_names[0])
    if not first_path.exists():
        return False

    # Open first subprocess pipeline and verify checkpoint exists
    if state.settings.checkpoint_format == "hdf":
        first_store = HdfStore(first_path, mode="r")
    else:
        first_store = ParquetStore(first_path, mode="r")

    try:
        cp_names = first_store.list_checkpoint_names()
        if resume_after not in cp_names:
            return False

        # Read checkpoint row to get table→checkpoint mapping
        cp_df = first_store.get_dataframe(CHECKPOINT_TABLE_NAME)
        cp_row = cp_df[cp_df[CHECKPOINT_NAME] == resume_after].iloc[-1]

        table_map = {}
        for col in cp_row.index:
            if col not in NON_TABLE_COLUMNS and cp_row[col]:
                table_map[col] = cp_row[col]

        # Read all tables from first subprocess at this checkpoint
        tables = {}
        for table_name, cp_for_table in table_map.items():
            try:
                tables[table_name] = first_store.get_dataframe(table_name, cp_for_table)
            except (FileNotFoundError, KeyError):
                logger.warning(
                    f"calibration: subprocess pipeline missing table "
                    f"{table_name} at {cp_for_table}"
                )
    finally:
        first_store.close()

    if not tables:
        return False

    # Determine sliced tables that need concatenation across processes
    sliced_table_names = set(slice_info.get("tables", [])) if slice_info else set()

    # Read sliced tables from remaining subprocesses and concatenate
    if num_processes > 1 and sliced_table_names:
        omnibus = {t: [tables[t]] for t in sliced_table_names if t in tables}

        for proc_name in sub_proc_names[1:]:
            proc_path = _subprocess_path(proc_name)
            if not proc_path.exists():
                logger.warning(
                    f"calibration: subprocess pipeline not found: {proc_path}"
                )
                return False

            if state.settings.checkpoint_format == "hdf":
                proc_store = HdfStore(proc_path, mode="r")
            else:
                proc_store = ParquetStore(proc_path, mode="r")

            try:
                proc_cp_df = proc_store.get_dataframe(CHECKPOINT_TABLE_NAME)
                proc_row = proc_cp_df[proc_cp_df[CHECKPOINT_NAME] == resume_after].iloc[
                    -1
                ]

                for table_name in list(omnibus.keys()):
                    cp_for_table = proc_row.get(table_name, "")
                    if cp_for_table:
                        omnibus[table_name].append(
                            proc_store.get_dataframe(table_name, cp_for_table)
                        )
            finally:
                proc_store.close()

        # Replace sliced tables with concatenated versions
        for table_name, dfs in omnibus.items():
            tables[table_name] = pd.concat(dfs, sort=False)

    # Load into parent state
    prior_rng_channels = list(state.get_injectable("rng_channels", []))
    prior_index_to_channel = (
        dict(state.rng().index_to_channel)
        if hasattr(state.rng(), "index_to_channel")
        else {}
    )

    state.init_state()
    if state.checkpoint.store_is_open():
        state.checkpoint.close_store()
    state.checkpoint.open_store(overwrite=False)

    for table_name, df in tables.items():
        state.add_table(table_name, df)

    _reregister_rng_channels(state, prior_rng_channels, prior_index_to_channel)

    # Mark all tables dirty for subsequent checkpoint.add
    for table_name in list(state.existing_table_names):
        state.existing_table_status[table_name] = True

    logger.info(
        "calibration: restored %d tables from subprocess pipelines at "
        "checkpoint '%s'",
        len(tables),
        resume_after,
    )
    return True


def _run_multiprocess_with_overrides(
    state: workflow.State,
    models: list[str],
    resume_after: str | None,
    shared_data_buffers: dict | None = None,
    can_reuse_subprocs: bool = False,
) -> None:
    """Run multiprocess with temporary settings overrides for calibration passes.

    Parameters
    ----------
    can_reuse_subprocs : bool, default False
        When True, subprocess pipelines from a prior run are assumed to exist
        and contain the ``resume_after`` checkpoint.  Breadcrumbs are written
        so that ``get_run_list`` populates ``step_info["resume_after"]``,
        apportion is skipped (reusing existing subprocess pipelines), and
        subprocesses resume from their model-level checkpoint — skipping
        already-completed models.
    """
    from collections import OrderedDict

    from activitysim.core import mp_tasks

    original_models = state.settings.models
    original_mp_steps = state.settings.multiprocess_steps
    original_resume_after = state.settings.resume_after

    # Build valid multiprocess_steps for the requested model subset.
    calibration_mp_steps = _build_calibration_mp_steps(
        models=models,
        original_steps=original_mp_steps,
        all_models=original_models,
    )

    state.settings.models = models
    state.settings.multiprocess_steps = calibration_mp_steps

    if can_reuse_subprocs and resume_after:
        # Include resume_after in the models list so get_breadcrumbs can
        # locate the step containing it.  Subprocesses will skip this model
        # (it's already checkpointed in their pipeline) and run the rest.
        models = [resume_after] + models

        # Rebuild steps with resume_after included.
        calibration_mp_steps = _build_calibration_mp_steps(
            models=models,
            original_steps=original_mp_steps,
            all_models=original_models,
        )
        state.settings.models = models
        state.settings.multiprocess_steps = calibration_mp_steps
        state.settings.resume_after = resume_after

        # Write minimal breadcrumbs indicating the step containing
        # resume_after has completed apportion (so it's skipped) but
        # simulate/coalesce need re-running.
        breadcrumbs = OrderedDict()
        for step in calibration_mp_steps:
            step_dict = {"name": step.name, "apportion": True}
            breadcrumbs[step.name] = step_dict
            # Find the step containing resume_after
            all_models = state.settings.models
            if resume_after in all_models:
                step_begin = all_models.index(step.begin)
                step_models_in_step = [
                    m for m in all_models[step_begin:] if m in models
                ]
                if resume_after in step_models_in_step:
                    # This step contains resume_after — stop here.
                    # get_breadcrumbs will mark simulate/coalesce for re-run.
                    break

        mp_tasks.write_breadcrumbs(state, breadcrumbs)
    else:
        # No reuse: calibration manages pipeline state externally via
        # _restore_parent_state_from_pipeline and checkpoint.add, so the MP
        # system's breadcrumb-based resume logic must not be triggered.
        state.settings.resume_after = None

    try:
        injectables = _build_calibration_injectables(state)
        mp_tasks.run_multiprocess(
            state,
            injectables,
            shared_data_buffers=shared_data_buffers,
            skip_final_checkpoint=True,
            force_resume=resume_after is not None and not can_reuse_subprocs,
        )
    finally:
        state.settings.models = original_models
        state.settings.resume_after = original_resume_after
        state.settings.multiprocess_steps = original_mp_steps


def _reregister_rng_channels(
    state: workflow.State,
    prior_channels: list[str],
    prior_index_to_channel: dict[str, str] = None,
) -> None:
    """Re-register RNG channels that were lost during init_state()."""
    current_channels = set(state.get_injectable("rng_channels", []))
    for channel_name in prior_channels:
        if channel_name not in current_channels and state.is_table(channel_name):
            try:
                state.rng().add_channel(channel_name, state.get_dataframe(channel_name))
                current_channels.add(channel_name)
            except Exception:
                pass
    # For channels whose tables don't exist at the restored checkpoint,
    # register an empty channel.  Do NOT pre-load from a later checkpoint
    # in the store — that data may include modifications from downstream
    # models and would pollute the pre-model state.  The empty channel
    # allows the model's normal table factory to create the table fresh
    # and extend the channel without hitting the disjoint-index assertion.
    if prior_index_to_channel:
        for index_name, channel_name in prior_index_to_channel.items():
            if index_name not in state.rng().index_to_channel:
                if channel_name not in state.rng().channels:
                    empty_df = pd.DataFrame(
                        index=pd.Index([], dtype="int64", name=index_name)
                    )
                    state.rng().add_channel(channel_name, empty_df)
                else:
                    state.rng().index_to_channel[index_name] = channel_name
                current_channels.add(channel_name)
    state.add_injectable("rng_channels", list(current_channels))


def _invalidate_derived_tables(state: workflow.State) -> None:
    """Drop factory-produced tables that may be stale after a calibration restore.

    When a calibrated model (e.g. auto_ownership) changes a table that a
    @workflow.table factory depends on (e.g. vehicles depends on households),
    the checkpoint may contain a stale version of that factory table.  Dropping
    it forces the factory to regenerate from current data on next access.

    Auto-detection rule: invalidate any table that is (a) registered as a
    @workflow.table factory, (b) has DataFrame parameters (= table dependencies),
    and (c) is in RANDOM_CHANNELS.  This currently matches only 'vehicles' but
    will automatically cover future factory tables with the same pattern.
    """
    settings = read_calibration_settings(state)
    if not settings:
        return

    tables_to_invalidate = settings.run.invalidate_tables
    if tables_to_invalidate is None:
        # Vehicles needs regeneration only when calibration changes
        # households.auto_ownership. Downstream calibration components must
        # retain vehicle_type_choice's vehicle attributes.
        tables_to_invalidate = (
            ["vehicles"]
            if "auto_ownership_simulate" in settings.run.calibrate_models
            else []
        )

    logger.debug(
        "calibration: tables detected for invalidation: %s", tables_to_invalidate
    )
    tables_before = set(state.existing_table_names)

    for table_name in tables_to_invalidate:
        if state.is_table(table_name):
            state.drop_table(table_name)
            state.rng().drop_channel(table_name)
            state.get_dataframe(table_name, as_copy=False)
            logger.debug("calibration: invalidated derived table '%s'", table_name)

    tables_after = set(state.existing_table_names)
    lost = tables_before - tables_after - set(tables_to_invalidate)
    if lost:
        logger.error(
            "calibration: tables unexpectedly removed during invalidation: %s", lost
        )


def _restore_parent_state_from_pipeline(
    state: workflow.State, checkpoint_name: str = "_"
) -> None:
    """Restore pipeline tables into the parent process state.

    After a multiprocess run, the parent's in-memory state is stale.
    This loads a specific checkpoint from the pipeline store so that
    calibration expressions can evaluate against model outputs.

    Parameters
    ----------
    checkpoint_name : str, default "_"
        The checkpoint to restore from.  Use a model-level checkpoint name
        (e.g. the prior step name) to get the exact state at that point,
        avoiding pollution from downstream models that may have added rows
        to shared tables like ``tours``.  The default ``"_"`` loads the
        last checkpoint, which is appropriate immediately after a coalesce
        that only ran the desired models.

    All tables are explicitly re-checkpointed so that subsequent apportion
    subprocesses can load them from a direct file path without relying on
    checkpoint backtracking through potentially ambiguous checkpoint history.
    """
    # Capture RNG state before restore — models may have dynamically
    # added channels (e.g. "vehicles") that aren't in the default
    # rng_channels injectable and would be lost by init_state().
    prior_rng_channels = list(state.get_injectable("rng_channels", []))
    prior_index_to_channel = (
        dict(state.rng().index_to_channel)
        if hasattr(state.rng(), "index_to_channel")
        else {}
    )

    if state.checkpoint.store_is_open():
        state.checkpoint.close_store()
    state.checkpoint.restore(resume_after=checkpoint_name)

    _reregister_rng_channels(state, prior_rng_channels, prior_index_to_channel)

    # After restore, all tables are clean (status=False). Mark them dirty so
    # the next checkpoint.add() writes them to disk at a known checkpoint name.
    # This ensures apportion subprocesses find table files at a single,
    # unambiguous checkpoint rather than needing to backtrack through history.
    for table_name in list(state.existing_table_names):
        state.existing_table_status[table_name] = True


def _initialize_mp_shared_resources(state: workflow.State) -> dict:
    """Allocate shared data buffers (skims, shadow pricing) once for reuse.

    This mirrors the allocation logic in mp_tasks.run_multiprocess but
    is called once at calibration start rather than on every sub-run.
    """
    from activitysim.core import mp_tasks, tracing

    shared_data_buffers = {}
    sharrow_enabled = state.settings.sharrow

    t0 = tracing.print_elapsed_time()
    if not sharrow_enabled:
        shared_data_buffers.update(mp_tasks.allocate_shared_skim_buffers(state))
        t0 = tracing.print_elapsed_time("calibration: allocate shared skim buffer", t0)

    shared_data_buffers.update(mp_tasks.allocate_shared_shadow_pricing_buffers(state))
    t0 = tracing.print_elapsed_time(
        "calibration: allocate shared shadow_pricing buffer", t0
    )

    shared_data_buffers.update(
        mp_tasks.allocate_shared_shadow_pricing_buffers_choice(state)
    )
    t0 = tracing.print_elapsed_time(
        "calibration: allocate shared shadow_pricing choice buffer", t0
    )

    # Load skim data into the shared buffers.
    if sharrow_enabled:
        shared_data_buffers["skim_dataset"] = "sh.Dataset:skim_dataset"
        from activitysim.core import flow, skim_dataset  # noqa: F401

        state.get_injectable("skim_dataset")
    else:
        if len(shared_data_buffers) > 0:
            injectables = _build_calibration_injectables(state)
            mp_tasks.run_sub_task(
                state,
                multiprocessing.Process(
                    target=mp_tasks.mp_setup_skims,
                    name="mp_setup_skims_calibration",
                    args=(injectables,),
                    kwargs=shared_data_buffers,
                ),
            )

    # Make skims available in the parent process for expression evaluation.
    state.add_injectable("data_buffers", shared_data_buffers)
    try:
        state.get_injectable("network_los")
    except Exception:
        logger.warning(
            "calibration: could not resolve network_los in parent process; "
            "skim-dependent expressions may fail"
        )

    return shared_data_buffers


def _build_calibration_mp_steps(
    models: list[str],
    original_steps: list[MultiprocessStep],
    all_models: list[str],
) -> list[MultiprocessStep]:
    """Build valid MultiprocessStep objects for a calibration model subset.

    The key challenge is that get_run_list() in mp_tasks requires:
    - The first step's begin == models[0]
    - Steps are ordered and non-overlapping
    - Each step's begin is in the models list

    We intersect the original multiprocess_steps with the requested model
    subset and construct new steps that satisfy these constraints.
    """
    if not models:
        return []

    # Determine which original step each model in the full list belongs to.
    # Build a mapping: model_name -> original step index
    model_to_step: dict[str, int] = {}
    step_boundaries = []
    for i, step in enumerate(original_steps):
        begin_idx = all_models.index(step.begin)
        step_boundaries.append(begin_idx)
    step_boundaries.append(len(all_models))

    for i, step in enumerate(original_steps):
        for model_idx in range(step_boundaries[i], step_boundaries[i + 1]):
            model_to_step[all_models[model_idx]] = i

    # Group the requested models by their original step
    from collections import OrderedDict

    step_model_groups: OrderedDict[int, list[str]] = OrderedDict()
    for model in models:
        step_idx = model_to_step.get(model)
        if step_idx is None:
            continue
        step_model_groups.setdefault(step_idx, []).append(model)

    # Build new MultiprocessStep for each group.
    # Some original steps (e.g. mp_initialize) omit num_processes, slice,
    # and chunk_size — these default to None on MultiprocessStep and
    # get_run_list() applies global defaults when they are absent.
    # Step names include the first model to ensure uniqueness across multiple
    # intermediate runs that draw from the same original step.
    new_steps = []
    for step_idx, step_models in step_model_groups.items():
        orig_step = original_steps[step_idx]
        kwargs: dict[str, Any] = {
            "name": orig_step.name,
            "begin": step_models[0],
        }
        if orig_step.num_processes is not None:
            kwargs["num_processes"] = orig_step.num_processes
        if orig_step.slice is not None:
            kwargs["slice"] = orig_step.slice
        if orig_step.chunk_size is not None:
            kwargs["chunk_size"] = orig_step.chunk_size
        new_steps.append(MultiprocessStep(**kwargs))

    return new_steps


def _build_calibration_injectables(state: workflow.State) -> dict:
    """Build the injectables dict for multiprocess sub-processes."""
    injectables = {}
    for key in MP_INJECTABLES:
        try:
            injectables[key] = state.get_injectable(key)
        except KeyError:
            pass
    injectables["settings"] = state.settings
    return injectables
