# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

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
    CALIBRATION_SETTINGS_FILE_NAME,
    CalibrationRunResult,
    read_calibration_settings,
)

logger = logging.getLogger("calibration")


@dataclass(frozen=True)
class CalibrationRestartPlan:
    """Side-effect-free decision about how persisted calibration should continue."""

    action: Literal["run", "finalize", "noop", "error"]
    start_global_iteration: int | None
    attempt: int | None
    completed_global_iterations: int
    completed_components: dict
    message: str | None = None


def _plan_calibration_restart(
    progress: dict | None,
    global_iterations: int,
) -> CalibrationRestartPlan:
    """Apply the user-facing global-iteration restart contract.

    The contract is intentionally based on completed logical iterations:

    1. An unchanged target on a completed run is a no-op.
    2. A changed target above the completed count extends to that new total,
       even when it is below the previous maximum after early convergence.
    3. A changed target already satisfied by the completed count is a no-op.
    4. ``resume_after`` is applied separately by the orchestrator, only to the
       first global iteration entered by the current invocation.
    5. The orchestrator counts an iteration only when it has a durable
       calibrated-component result from some attempt of that iteration.
    6. An active iteration cannot be discarded by lowering the target.
    7. A lower target at a clean boundary is finalized without another
       coefficient update so downstream outputs reflect current coefficients.
    """
    if not progress:
        return CalibrationRestartPlan("run", 1, 1, 0, {})

    completed = int(progress.get("last_completed_global_iteration", 0))
    completed_components = dict(progress.get("completed_components", {}))

    if progress.get("complete"):
        previous_target = int(progress.get("configured_global_iterations", completed))
        if global_iterations == previous_target or global_iterations <= completed:
            return CalibrationRestartPlan(
                "noop", None, None, completed, completed_components
            )
        return CalibrationRestartPlan("run", completed + 1, 1, completed, {})

    interrupted = progress.get("in_progress_iteration")
    if interrupted is not None:
        interrupted = int(interrupted)
        if global_iterations < interrupted:
            return CalibrationRestartPlan(
                "error",
                None,
                None,
                completed,
                completed_components,
                "Cannot set calibration run.global_iterations to "
                f"{global_iterations} because global iteration {interrupted} is "
                "currently in progress and coefficient files may already contain "
                f"iteration {interrupted} updates. Set global_iterations to at "
                f"least {interrupted} to resume, or deliberately reset calibration "
                "progress and restore the desired coefficient files before starting "
                "a new run.",
            )
        return CalibrationRestartPlan(
            "run",
            interrupted,
            int(progress.get("attempt", 1)) + 1,
            completed,
            completed_components,
        )

    next_iteration = int(progress.get("next_global_iteration", completed + 1))
    if next_iteration > global_iterations:
        return CalibrationRestartPlan("finalize", next_iteration, 1, completed, {})
    return CalibrationRestartPlan("run", next_iteration, 1, completed, {})


def calibration_run_should_preserve_outputs(state: workflow.State) -> bool:
    """Return whether preflight must preserve outputs before orchestration."""
    # Keep missing and disabled calibration files on the ordinary ActivitySim
    # cleanup path. Only enabled calibration runs get calibration-specific
    # protection during preflight.
    try:
        raw_settings = state.filesystem.read_settings_file(
            CALIBRATION_SETTINGS_FILE_NAME,
            mandatory=False,
        )
    except Exception:
        logger.debug(
            "preserving outputs because calibration settings could not be read",
            exc_info=True,
        )
        return True
    if not raw_settings or not raw_settings.get("enable", False):
        return False

    # Calibration validation normally runs after output cleanup. Preserve the
    # outputs if enabled settings are invalid, then let the normal settings
    # checker aggregate and report the validation error later.
    try:
        calibration_settings = read_calibration_settings(state)
    except Exception:
        logger.debug(
            "preserving outputs because calibration settings validation failed",
            exc_info=True,
        )
        return True

    if not calibration_settings or not calibration_settings.enable:
        return False

    progress = _read_progress(state)
    if not progress or not progress.get("complete"):
        if not progress:
            return False
    configured_global_iterations = calibration_settings.run.global_iterations
    plan = _plan_calibration_restart(progress, configured_global_iterations)
    return plan.action in {"noop", "error"}


def _validate_counted_iteration_has_calibration(
    calibration_models: list[str],
    skipped_components: list[str],
    completed_components: dict,
) -> None:
    """Do not count a new logical iteration that has no calibration result."""
    if set(skipped_components) != set(calibration_models):
        return
    if any(component in completed_components for component in calibration_models):
        return
    raise RuntimeError(
        "settings.yaml resume_after skips every calibrated model in the first "
        "pending global iteration, and that iteration has no durable calibrated "
        "component result from an earlier attempt. Choose a resume_after before "
        "the last calibrated model, or remove resume_after, so the requested "
        "global calibration iteration performs a coefficient update."
    )


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

    skipped_calibration_models = []
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
    restart_plan = _plan_calibration_restart(
        progress,
        calibration_settings.run.global_iterations,
    )
    logger.info(
        "calibration restart plan: action=%s completed=%s requested=%s "
        "start_iteration=%s attempt=%s resume_after=%r",
        restart_plan.action,
        restart_plan.completed_global_iterations,
        calibration_settings.run.global_iterations,
        restart_plan.start_global_iteration,
        restart_plan.attempt,
        resume_after,
    )
    if restart_plan.action == "error":
        raise RuntimeError(restart_plan.message)
    if restart_plan.action == "noop":
        logger.info(
            "calibration progress is already complete for the requested target; "
            "remove %s to start a fresh calibration run",
            CALIBRATION_PROGRESS_FILE,
        )
        return CalibrationRunResult(
            converged=bool(progress.get("converged", False)),
            completed_global_iterations=restart_plan.completed_global_iterations,
        )

    # Validate the requested restart before changing a formerly complete
    # progress record. A rejected resume_after must leave durable progress in
    # exactly the state in which it was found.
    if restart_plan.action == "run":
        _validate_counted_iteration_has_calibration(
            calibration_settings.run.calibrate_models,
            skipped_calibration_models,
            restart_plan.completed_components,
        )

    progress_was_complete = bool(progress and progress.get("complete"))
    if progress_was_complete:
        previous_target = int(
            progress.get(
                "configured_global_iterations",
                restart_plan.completed_global_iterations,
            )
        )
        logger.info(
            "calibration global_iterations changed from %s to %s; continuing "
            "with global iteration %s",
            previous_target,
            calibration_settings.run.global_iterations,
            restart_plan.start_global_iteration,
        )
        progress = {
            "complete": False,
            "in_progress_iteration": None,
            "next_global_iteration": restart_plan.start_global_iteration,
            "last_completed_global_iteration": (
                restart_plan.completed_global_iterations
            ),
            "converged": bool(progress.get("converged", False)),
            "configured_global_iterations": calibration_settings.run.global_iterations,
            "attempt": 0,
            "completed_components": {},
        }
        _write_progress(state, progress)

    interrupted_iteration = progress.get("in_progress_iteration") if progress else None
    start_global_iter = restart_plan.start_global_iteration
    start_attempt = restart_plan.attempt
    start_completed_components = restart_plan.completed_components
    completed_global_iterations = restart_plan.completed_global_iterations

    if interrupted_iteration is not None:
        logger.warning(
            "continuing interrupted calibration global iteration %s as attempt %s "
            "using the current coefficient files",
            start_global_iter,
            start_attempt,
        )

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
        if restart_plan.action == "finalize":
            # The target was lowered at a clean boundary between iterations.
            # No active coefficient update is being discarded, but the normal
            # model sequence still runs so final outputs use current coefficients.
            logger.info(
                "calibration global_iterations=%s is below next global iteration "
                "%s; running the final model sequence without another "
                "calibration update",
                calibration_settings.run.global_iterations,
                start_global_iter,
            )
            final_models = (
                models if first_model_idx is None else models[first_model_idx:]
            )
            _run_in_configured_mode(
                state,
                models=final_models,
                resume_after=state.settings.resume_after,
                shared_data_buffers=shared_data_buffers,
            )
            completed_global_iterations = restart_plan.completed_global_iterations
            converged = bool(progress.get("converged", False)) if progress else False
            _write_final_coefficients_snapshot(state, calibration_settings)
            _write_completed_progress(
                state,
                completed_global_iterations,
                converged,
                calibration_settings.run.global_iterations,
                attempt=int(progress.get("attempt", 1)) if progress else 1,
                completed_components={},
            )
            return CalibrationRunResult(
                converged=converged,
                completed_global_iterations=completed_global_iterations,
            )

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

                try:
                    _write_component_plots(state, component)
                except Exception:
                    logger.exception(
                        "calibration component %s completed, but its optional "
                        "standard plots could not be written",
                        component,
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
