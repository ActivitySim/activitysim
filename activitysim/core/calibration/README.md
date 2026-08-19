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
- `recovery.py` maintains calibration progress and start-of-iteration coefficient
  backups.
- `execution.py` restores pipeline state and dispatches model execution.
- `multiprocess.py` contains subprocess orchestration, shared-resource setup, and
  multiprocess pipeline restoration.

Dependencies should flow from `orchestrator.py` into the focused modules. Lower
level modules should not import the orchestrator. Multiprocess code imports back
into `execution.py` only inside the single-component runner to avoid an import-time
cycle.
