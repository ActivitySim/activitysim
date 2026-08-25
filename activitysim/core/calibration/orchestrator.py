# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging

from activitysim.core import workflow

from .component import _calibrate_component
from .execution import (
    _prep_model_data,
    _run_in_configured_mode,
)
from .multiprocess import _initialize_mp_shared_resources
from .recovery import (
    CALIBRATION_PROGRESS_FILE,
    _mark_global_iteration_in_progress,
    _read_progress,
    _write_completed_progress,
    _write_progress,
)
from .reporting import (
    _ensure_calibration_output_dir,
    _write_component_plots,
    _write_final_coefficients_snapshot,
)
from .settings import (
    CalibrationRunResult,
    read_calibration_settings,
)

logger = logging.getLogger("calibration")


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
        completed_global_iterations = int(
            progress.get("last_completed_global_iteration", 0)
        )
        completed_for_global_iterations = int(
            progress.get(
                "configured_global_iterations", completed_global_iterations
            )
        )
        if calibration_settings.run.global_iterations > completed_for_global_iterations:
            logger.info(
                "calibration global_iterations increased from %s to %s; "
                "continuing with global iteration %s",
                completed_for_global_iterations,
                calibration_settings.run.global_iterations,
                completed_global_iterations + 1,
            )
            progress = {
                "complete": False,
                "in_progress_iteration": None,
                "next_global_iteration": completed_global_iterations + 1,
                "last_completed_global_iteration": completed_global_iterations,
                "converged": bool(progress.get("converged", False)),
                "configured_global_iterations": calibration_settings.run.global_iterations,
                "attempt": 0,
                "completed_components": {},
            }
            _write_progress(state, progress)
        else:
            if resume_after is not None:
                raise RuntimeError(
                    f"settings.yaml resume_after={resume_after!r} cannot be honored "
                    "because calibration progress is already complete. Remove "
                    f"{CALIBRATION_PROGRESS_FILE} or use a new output directory to "
                    "start a new calibration run; current coefficient values will "
                    "be preserved."
                )
            logger.info(
                "calibration progress is already complete; remove %s to start a "
                "fresh calibration run",
                CALIBRATION_PROGRESS_FILE,
            )
            return CalibrationRunResult(
                converged=bool(progress.get("converged", False)),
                completed_global_iterations=completed_global_iterations,
            )

    interrupted_iteration = progress.get("in_progress_iteration") if progress else None
    if interrupted_iteration is not None:
        start_global_iter = int(interrupted_iteration)
        start_attempt = int(progress.get("attempt", 1)) + 1
        start_completed_components = dict(progress.get("completed_components", {}))
        logger.warning(
            "continuing interrupted calibration global iteration %s as attempt %s "
            "using the current coefficient files",
            start_global_iter,
            start_attempt,
        )
    else:
        # Progress files from earlier versions contain next_global_iteration, so
        # they remain compatible with the corrected total-count semantics.
        start_global_iter = (
            int(progress.get("next_global_iteration", 1)) if progress else 1
        )
        start_attempt = 1
        start_completed_components = {}
    completed_global_iterations = start_global_iter - 1

    if interrupted_iteration is not None and resume_after is not None:
        rerun_completed_components = [
            component
            for component in start_completed_components
            if component in calibration_settings.run.calibrate_models
            and models.index(component) > models.index(resume_after)
        ]
        if rerun_completed_components:
            logger.warning(
                "resume_after=%r rewinds across completed calibrated component(s) "
                "%s. Their coefficient values will not be rolled back; rerun "
                "results will be appended as attempt %s of global iteration %s.",
                resume_after,
                rerun_completed_components,
                start_attempt,
                start_global_iter,
            )

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
            calibration_settings.run.global_iterations,
            attempt=int(progress.get("attempt", 1)) if progress else 1,
            completed_components=(
                dict(progress.get("completed_components", {})) if progress else {}
            ),
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
            attempt = start_attempt if global_iter == start_global_iter else 1
            completed_components = (
                dict(start_completed_components)
                if global_iter == start_global_iter
                else {}
            )
            _mark_global_iteration_in_progress(
                state,
                global_iter,
                attempt,
                completed_components,
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
                if first_calibration_restart_step is not None:
                    state.checkpoint.add(first_calibration_restart_step)
                    state.checkpoint.close_store()

            logger.info(
                "calibration global iteration %s/%s attempt %s",
                global_iter,
                calibration_settings.run.global_iterations,
                attempt,
            )

            skipped_components = _skipped_calibration_components(
                calibration_models=calibration_settings.run.calibrate_models,
                models=models,
                first_model_idx=first_model_idx,
                global_iter=global_iter,
                start_global_iter=start_global_iter,
            )
            if skipped_components:
                all_converged = all(
                    bool(completed_components.get(component, {}).get("converged"))
                    for component in skipped_components
                )
            else:
                all_converged = _components_ran_for_convergence(
                    first_model_idx=first_model_idx,
                    last_calib_model_idx=last_calib_model_idx,
                    global_iter=global_iter,
                    start_global_iter=start_global_iter,
                )

            last_calibrated_component = None
            for component in calibration_settings.run.calibrate_models:
                # on the first global iter, skip model if it's before or == resume_after
                if component in skipped_components:
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
                    attempt=attempt,
                    shared_data_buffers=shared_data_buffers,
                )
                _write_component_plots(state, component)

                all_converged = all_converged and component_result.converged

                completed_components[component] = {
                    "attempt": attempt,
                    "converged": component_result.converged,
                    "component_iterations": component_result.component_iterations,
                }
                _mark_global_iteration_in_progress(
                    state,
                    global_iter,
                    attempt,
                    completed_components,
                )

                last_calibrated_component = component

            iteration_is_complete = (
                all_converged
                or global_iter == calibration_settings.run.global_iterations
            )
            resumed_after_all_calibrated_models = (
                global_iter == start_global_iter
                and state.settings.resume_after is not None
                and first_model_idx > last_calib_model_idx
            )

            if (
                calibration_settings.run.complete_steps
                or iteration_is_complete
                or resumed_after_all_calibrated_models
            ):
                subsequent_components = (
                    models[first_model_idx:]
                    if resumed_after_all_calibrated_models
                    else models[models.index(last_calibrated_component) + 1 :]
                )
                # finish the full model chain
                _run_subsequent_components(
                    state,
                    models=subsequent_components,
                    resume_after=state.settings.resume_after
                    if resumed_after_all_calibrated_models
                    else last_calibrated_component,
                    shared_data_buffers=shared_data_buffers,
                )

            completed_global_iterations = global_iter
            if not iteration_is_complete:
                _write_progress(
                    state,
                    {
                        "in_progress_iteration": None,
                        "next_global_iteration": global_iter + 1,
                        "last_completed_global_iteration": global_iter,
                        "converged": all_converged,
                        "attempt": 0,
                        "completed_components": {},
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
            calibration_settings.run.global_iterations,
            attempt=attempt,
            completed_components=completed_components,
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


def _prior_step_name(models: list[str], component_name: str) -> str | None:
    """Return the step name immediately preceding component_name in models."""
    if component_name not in models:
        return None
    idx = models.index(component_name)
    if idx == 0:
        return None
    return models[idx - 1]


def _components_ran_for_convergence(
    first_model_idx: int | None,
    last_calib_model_idx: int,
    global_iter: int,
    start_global_iter: int,
) -> bool:
    """Return whether component results can establish convergence this iteration."""
    return (
        first_model_idx is None
        or first_model_idx <= last_calib_model_idx
        or global_iter > start_global_iter
    )


def _skipped_calibration_components(
    calibration_models: list[str],
    models: list[str],
    first_model_idx: int | None,
    global_iter: int,
    start_global_iter: int,
) -> list[str]:
    """Return calibrated components skipped by resume_after this iteration."""
    if global_iter != start_global_iter or first_model_idx is None:
        return []
    return [
        component
        for component in calibration_models
        if first_model_idx > models.index(component)
    ]
