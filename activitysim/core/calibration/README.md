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

When a run is interrupted, calibration continues the active global iteration.
Any coefficient updates written before the interruption, including subsequent
manual edits, are preserved. On the first resumed iteration, top-level
`settings.yaml` `resume_after` uses standard ActivitySim semantics: the named
model is treated as complete and execution begins with the following model.

Dependencies should flow from `orchestrator.py` into the focused modules. Lower
level modules should not import the orchestrator. Multiprocess code imports back
into `execution.py` only inside the single-component runner to avoid an import-time
cycle.
