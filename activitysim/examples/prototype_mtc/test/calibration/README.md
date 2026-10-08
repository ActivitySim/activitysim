# Prototype MTC calibration run-mode test

This fixture exercises calibration with three real prototype MTC models:

- `workplace_location` evaluates its target with the `DIST` network skim;
- `auto_ownership_simulate` provides a second upstream coefficient update; and
- `tour_mode_choice_simulate` is the downstream restart target.

The test runs uninterrupted single- and multiprocess references. It then runs
each mode with a deliberately invalid tour-mode calibration expression, verifies
that global iteration 1 remains in progress, replaces the expression, and
restarts with `resume_after: non_mandatory_tour_scheduling`. The resumed runs
must preserve the workplace and auto-ownership coefficient files and match the
uninterrupted final tables and calibrated coefficients across both modes.
The fixture also rewinds a failed run to `initialize_landuse`, verifies that the
completed calibrated models rerun as attempt 2 of the same global iteration,
and confirms that attempt-1 and attempt-2 coefficient transitions are both
retained with a continuous `next_coefficient` to `prev_coefficient` chain.

Each run copies mutable coefficient files into its temporary config directory;
the shared prototype MTC configs and data remain unchanged. The 25-household
sample minimizes simulation work, although process startup and pipeline
apportion/coalesce make the full matrix an integration test rather than a fast
unit test.

Run from the repository root with:

```powershell
.venv\Scripts\python.exe -m pytest activitysim/examples/prototype_mtc/test/calibration/test_run_modes.py -q
```
