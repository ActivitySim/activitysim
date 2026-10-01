# ActivitySim calibration package

This package implements ActivitySim's automated coefficient calibration workflow.
The modules are divided by responsibility so orchestration, model execution, and
calibration math can evolve independently.

- `__init__.py` defines the public API and compatibility access to former private
  attributes from `activitysim.core.calibration`.
- `settings.py` defines calibration configuration models, result types, and
  configuration loading.
- `orchestrator.py` coordinates global iterations and model-component sequencing.
- `component.py` runs component iterations, validates calibration specifications,
  and calculates coefficient updates.
- `expressions.py` builds expression contexts, evaluates target/model expressions,
  and loads optional helper modules.
- `coefficients.py` locates, reads, and writes model coefficient files.
- `reporting.py` writes iteration histories, summaries, plots, and final snapshots.
- `recovery.py` durably tracks calibration progress. Coefficient files are the
  authoritative current state and are never rolled back when a run resumes.
- `execution.py` restores pipeline state and dispatches model execution.
- `multiprocess.py` contains subprocess orchestration, shared-resource setup, and
  multiprocess pipeline restoration.

Component settings may specify `model_settings_file` when a workflow step does
not follow the conventional `<component_name>.yaml` naming pattern. Components
that share model settings may point to the same coefficient file.

## Global iterations, recovery attempts, and `resume_after`

`calibration.yaml` `run.global_iterations` is the desired total number of
completed logical calibration iterations, not the number to execute on each
ActivitySim invocation. The run-control contract is:

1. An unchanged setting on a completed run is a no-op, detected before normal
   output cleanup so the pipeline and final outputs are preserved.
2. If a completed run's setting is changed to a value greater than the number
   actually completed, calibration continues until the new total. This includes
   lowering a previous maximum after early convergence, such as changing 5 to 3
   after convergence completed iteration 2.
3. A changed setting less than or equal to the number already completed is a
   no-op; completed coefficient updates are never undone implicitly.
4. Top-level `settings.yaml` `resume_after` has normal ActivitySim semantics for
   the first global iteration entered by the current invocation. Later global
   iterations ignore it and execute every calibrated component.
5. A global iteration counts only if it has at least one durable calibrated
   component result, either from the current attempt or an earlier attempt of
   that same logical iteration.
6. `global_iterations` cannot be lowered below an interrupted iteration because
   its coefficient files may already contain updates from that iteration. The
   run stops with instructions to resume the iteration or deliberately reset
   progress and coefficients.
7. Startup logs report the detected completed count, requested target, selected
   action, starting iteration and attempt, and `resume_after` value.

Calibration records the state needed to apply this contract in
`output/calibration/calibration_progress.json`.

Recovery attempts are distinct from global iterations. The first execution of a
global iteration is attempt 1. Restarting an interrupted global iteration creates
attempt 2, then attempt 3 if another restart is needed. A recovery attempt does
not consume an additional global iteration.

Calibration uses only the top-level `settings.yaml` `resume_after` setting. On
the first global iteration executed by an ActivitySim invocation, including a
new attempt of an interrupted iteration, it behaves like `resume_after` in a
non-calibration run:

- when set to a model name, that model is treated as complete, its checkpoint is
  restored, and execution begins with the following model; and
- when unset or `null`, execution starts at the beginning of the top-level
  `models` list. Calibration does not automatically continue after the last
  model completed by the preceding attempt.

A named value must occur in the top-level `models` list and must identify a
model-level checkpoint. Calibration does not define a special `initialize`
value. A value such as `initialize_landuse` has ordinary model-name semantics:
that model is skipped and execution begins with the next model. The `_` shorthand
for the last checkpoint is not accepted in calibration mode.

`resume_after` affects only the first global iteration entered by the current
invocation. Calibrated models at or before a named resume point are skipped in
that iteration. Any later global iterations in the same invocation ignore
`resume_after`, restart immediately before the first calibrated model, and run
the normal complete calibration sequence.

For example, suppose global iteration 3 attempt 1 was interrupted after
calibrated `model_a` completed:

- `resume_after: model_a` starts attempt 2 after model A, preserving its
  attempt-1 result; and
- `resume_after: null` starts attempt 2 at the beginning of the complete model
  list, so model A runs again.

Coefficient files are the authoritative current state and are never rolled back
during recovery. Updates written before the interruption, along with subsequent
manual edits, become the starting coefficients for the new attempt. Rewinding
across a completed calibrated model therefore applies another calibration update
to its current coefficients. Calibration logs a warning when `resume_after`
causes this kind of rewind.

Iteration histories are append-only across attempts. Standard record, summary,
and generic-report rows include `global_iter`, `attempt`, and `component_iter`,
so a rerun does not replace the coefficient transition written by an earlier
attempt. The coefficient trajectory plots show the complete sequence, including
the initial value and every subsequent update. X-axis labels use the compact
form `G<global>-A<attempt>-C<component>`, for example `G3-A2-C1`.

After each calibrated component completes, the progress file records its attempt,
component-iteration count, and convergence result. A restart that skips that
component retains this state. This allows a run that crashed in downstream,
non-calibrated models to finish the same terminal global iteration without losing
the calibrated components' convergence decision.

Calibration is marked complete only after the terminal global iteration has run
all remaining models in the top-level `models` list and the final coefficient
snapshot has been written. A crash after the last calibrated model but before the
last production model therefore leaves the global iteration in progress.

Dependencies should flow from `orchestrator.py` into the focused modules. Lower
level modules should not import the orchestrator. Multiprocess code imports back
into `execution.py` only inside the single-component runner to avoid an import-time
cycle.
