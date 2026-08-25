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

`calibration.yaml` `run.global_iterations` is the maximum total number of
logical global calibration iterations, not the number to execute on each
ActivitySim invocation. Calibration records durable progress in
`output/calibration/calibration_progress.json`:

- a new output directory starts at global iteration 1;
- a crash re-enters the interrupted global iteration;
- a cleanly completed iteration advances to the next global iteration;
- convergence can finish the run before the configured maximum; and
- after a completed run, increasing `global_iterations` continues at the next
  unfinished global iteration. An unchanged or lower value remains a no-op.

Recovery attempts are distinct from global iterations. The first execution of a
global iteration is attempt 1. Restarting an interrupted global iteration creates
attempt 2, then attempt 3 if another restart is needed. A recovery attempt does
not consume an additional global iteration.

Calibration uses the top-level `settings.yaml` `resume_after` setting. It has
strict ActivitySim semantics: the named model is treated as complete, its
checkpoint is restored, and execution begins with the following model. The name
must occur in the top-level `models` list and must be a model-level checkpoint;
the `_` shorthand for the last checkpoint is not accepted in calibration mode.
The similarly named `calibration.yaml` `run.resume_after` compatibility field is
not used by the orchestrator.

On the first global iteration entered by an invocation, calibrated models at or
before `resume_after` are skipped. Later global iterations in the same invocation
restart immediately before the first calibrated model and run the normal complete
calibration sequence. For example, if global iteration 3 was interrupted after
calibrated `model_a` completed:

- `resume_after: model_a` preserves model A's attempt-1 result and resumes with
  the following model under attempt 2;
- `resume_after: initialize` rewinds pipeline state and reruns model A under
  attempt 2; and
- no `resume_after` replays the interrupted iteration from the beginning under
  attempt 2.

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
