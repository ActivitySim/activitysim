from __future__ import annotations

import pytest

from activitysim.core.calibration.orchestrator import (
    _plan_calibration_restart,
    _skipped_calibration_components,
    _validate_counted_iteration_has_calibration,
)


@pytest.mark.parametrize(
    (
        "progress",
        "global_iterations",
        "expected_action",
        "expected_start",
        "expected_attempt",
    ),
    [
        # no prior progress file: start fresh from iteration 1, attempt 1
        (None, 2, "run", 1, 1),
        (
            # interrupted mid-run at iteration 3; resume with an incremented attempt number
            {
                "in_progress_iteration": 3,
                "last_completed_global_iteration": 2,
                "attempt": 1,
                "completed_components": {},
            },
            3,
            "run",
            3,
            2,
        ),
        (
            # calibration iterations done but downstream finalization models still pending
            {
                "last_completed_global_iteration": 2,
                "next_global_iteration": 3,
            },
            2,
            "finalize",
            3,
            1,
        ),
        (
            # fully complete and global_iterations unchanged: nothing left to do
            {
                "complete": True,
                "last_completed_global_iteration": 2,
                "configured_global_iterations": 2,
            },
            2,
            "noop",
            None,
            None,
        ),
        (
            # fully complete but global_iterations increased: extend the calibration run
            {
                "complete": True,
                "last_completed_global_iteration": 2,
                "configured_global_iterations": 2,
            },
            4,
            "run",
            3,
            1,
        ),
        # Convergence completed only two iterations against a target of five.
        # Leaving the target unchanged is a no-op, while changing it to a value
        # above the completed count explicitly requests more iterations.
        (
            # converged early at iter 2; configured target of 5 not changed → no-op
            {
                "complete": True,
                "last_completed_global_iteration": 2,
                "configured_global_iterations": 5,
            },
            5,
            "noop",
            None,
            None,
        ),
        (
            # converged early at iter 2; target lowered to 3 → run one more iteration
            {
                "complete": True,
                "last_completed_global_iteration": 2,
                "configured_global_iterations": 5,
            },
            3,
            "run",
            3,
            1,
        ),
        (
            # converged early at iter 2; target lowered to 4 → run two more iterations starting at 3
            {
                "complete": True,
                "last_completed_global_iteration": 2,
                "configured_global_iterations": 5,
            },
            4,
            "run",
            3,
            1,
        ),
        (
            # converged early at iter 2; target lowered to 2 (≤ completed count) → no-op
            {
                "complete": True,
                "last_completed_global_iteration": 2,
                "configured_global_iterations": 5,
            },
            2,
            "noop",
            None,
            None,
        ),
    ],
)
def test_plan_calibration_restart(
    progress,
    global_iterations,
    expected_action,
    expected_start,
    expected_attempt,
):
    plan = _plan_calibration_restart(progress, global_iterations)

    assert plan.action == expected_action
    assert plan.start_global_iteration == expected_start
    assert plan.attempt == expected_attempt


def test_cannot_lower_limit_below_interrupted_iteration():
    # iteration 3 is in progress; lowering global_iterations to 2 would abandon it mid-run
    plan = _plan_calibration_restart(
        {
            "in_progress_iteration": 3,
            "last_completed_global_iteration": 2,
            "attempt": 1,
            "completed_components": {"model_a": {"attempt": 1, "converged": True}},
        },
        global_iterations=2,
    )

    assert plan.action == "error"
    # error message must identify the blocked iteration and the minimum required setting
    assert "global iteration 3 is currently in progress" in plan.message
    assert "global_iterations to at least 3" in plan.message


def test_resume_after_only_skips_components_in_first_entered_iteration():
    # model_a ran and completed in a prior attempt during iteration 3; model_b had not yet started
    # first_model_idx is the pipeline position immediately after model_a
    models = ["initialize", "model_a", "intermediate", "model_b", "output"]
    calibration_models = ["model_a", "model_b"]
    first_model_idx = models.index("model_a") + 1

    # re-entering the interrupted iteration: model_a already ran, so it must be skipped
    assert _skipped_calibration_components(
        calibration_models,
        models,
        first_model_idx,
        global_iter=3,
        start_global_iter=3,
    ) == ["model_a"]
    # on a later full iteration nothing was partially skipped before, run everything normally
    assert (
        _skipped_calibration_components(
            calibration_models,
            models,
            first_model_idx,
            global_iter=4,
            start_global_iter=3,
        )
        == []
    )


def test_new_iteration_cannot_skip_every_calibrated_component():
    # with no prior-attempt results in completed_components, skipping all calibration
    # models would produce an iteration with zero calibration updates — that is a bug
    with pytest.raises(RuntimeError, match="skips every calibrated model"):
        _validate_counted_iteration_has_calibration(
            calibration_models=["model_a", "model_b"],
            skipped_components=["model_a", "model_b"],
            completed_components={},
        )


def test_prior_attempt_result_allows_downstream_only_recovery():
    # model_a has a completed result from a prior attempt, so skipping both calibration
    # models is valid: downstream models will still pick up model_a's updated coefficients
    _validate_counted_iteration_has_calibration(
        calibration_models=["model_a", "model_b"],
        skipped_components=["model_a", "model_b"],
        completed_components={"model_a": {"attempt": 1, "converged": True}},
    )
