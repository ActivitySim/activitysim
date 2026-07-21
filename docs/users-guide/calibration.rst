.. _calibration:

===========================
ActivitySim Auto-Calibration
===========================

ActivitySim includes an automated calibration framework that iteratively adjusts
model coefficients to match observed survey targets. Calibration wraps around
ActivitySim's normal model execution — it reuses the standard model runner and
only adds orchestration logic around it.

Overview
========

The calibration loop works as follows:

1. Run all model steps preceding the first calibrated component (the "precursor" models).
2. For each calibrated component, iteratively:

   - Run the component from its prior checkpoint.
   - Evaluate model output shares and compare against survey targets.
   - Compute coefficient adjustments using a damped log-ratio or odds-ratio method.
   - Write updated coefficients back to the config file on disk.
   - Check convergence; stop early if all coefficients are within tolerance.

3. Run any intermediate (non-calibrated) model steps between calibrated components.
4. Optionally run the remaining model steps after the last calibrated component.
5. Repeat the entire process for a configurable number of **global iterations**.
6. Write a final snapshot of all calibrated coefficients.

Both single-process and multiprocess execution modes are fully supported. Shared
resources (skim buffers, shadow pricing) are allocated once and reused across all
calibration iterations.

Quick Start
===========

1. Prepare Your Configs Directory
----------------------------------

Add the following files to your ActivitySim configs directory (or a
calibration-specific overlay directory):

- ``calibration.yaml`` — top-level calibration configuration
- One **calibration spec CSV** per calibrated component
- One **coefficients CSV** per calibrated component (may already exist from estimation)
- Optionally, one **helper module** (``.py``) per component for custom expressions or reports

2. Create ``calibration.yaml``
-------------------------------

.. code-block:: yaml

   enable: True

   run:
     calibrate_models:
       - workplace_location
       - auto_ownership_simulate
       - tour_mode_choice_simulate
     resume_after: null        # checkpoint to resume from on global iteration 1
     restart_after: []          # components after which to restart (advanced)
     global_iterations: 3       # number of full calibration passes
     complete_steps: false      # run model steps after the last calibrated component

   model_settings:
     workplace_location:
       calibration_spec: workplace_location_calibration.csv
       helper_module: workplace_location_calib_helper.py
       submodel_max_iterations: 3
       survey_file: survey_persons.csv
       reports:
         generic: true
         bespoke: report_workplace_location

     auto_ownership_simulate:
       calibration_spec: auto_ownership_calibration.csv
       helper_module: auto_ownership_calib_helper.py
       submodel_max_iterations: 3
       survey_file: survey_households.csv
       reports:
         generic: true
         bespoke: report_auto_ownership

     tour_mode_choice_simulate:
       calibration_spec: tour_mode_choice_calibration.csv
       helper_module: tour_mode_choice_calib_helper.py
       submodel_max_iterations: 3
       survey_file: survey_tours.csv
       reports:
         generic: true
         bespoke: report_tour_mode_choice

3. Run Calibration
-------------------

Calibration is triggered through the standard ``activitysim run`` command. No
special subcommand is needed — when ``calibration.yaml`` exists in the config
directory with ``enable: True``, the calibration loop runs automatically instead
of the normal model flow.

.. code-block:: bash

   activitysim run -c configs_calibration -c configs -d data -o output

Use multiple ``-c`` flags to layer a calibration-specific config directory on top
of your base configs. The calibration configs override base configs via
ActivitySim's standard config resolution order (earlier directories take priority).


Configuration Reference
=======================

``calibration.yaml`` — Top-Level Settings
-------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Field
     - Type
     - Default
     - Description
   * - ``enable``
     - ``bool``
     - ``False``
     - Master switch. Set to ``True`` to activate calibration.
   * - ``run``
     - object
     - *required*
     - Run-control settings (see below).
   * - ``model_settings``
     - dict
     - ``{}``
     - Per-component calibration config, keyed by component name.

**Validation rules:**

- Every component listed in ``run.calibrate_models`` must have a corresponding
  entry in ``model_settings``.
- Every component in ``run.restart_after`` must also appear in ``run.calibrate_models``.
- ``run.global_iterations`` must be ≥ 1.

``run`` — Run Control Settings
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 10 55

   * - Field
     - Type
     - Default
     - Description
   * - ``calibrate_models``
     - ``list[str]``
     - *required*
     - Model component names to calibrate. Must match names in ``settings.yaml``
       ``models`` list.
   * - ``resume_after``
     - ``str`` or ``null``
     - ``null``
     - Checkpoint to resume from on the first global iteration. Equivalent to
       ``resume_after`` in ``settings.yaml``. Use this to skip expensive
       initialization steps that do not change across calibration iterations.
   * - ``restart_after``
     - ``list[str]``
     - ``[]``
     - Components after which to restart.
   * - ``global_iterations``
     - ``int``
     - ``1``
     - Number of full outer-loop calibration passes over all components. Each
       global iteration re-runs all precursor models and re-calibrates all
       components from scratch using the latest coefficients.
   * - ``complete_steps``
     - ``bool``
     - ``False``
     - Whether to run all model steps after the last calibrated component.
       When ``False``, only runs them on the final global iteration. Set to
       ``True`` if downstream model outputs are needed every iteration (e.g.,
       for dashboarding).

``model_settings.<component_name>`` — Per-Component Settings
-------------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 10 50

   * - Field
     - Type
     - Default
     - Description
   * - ``calibration_spec``
     - ``str``
     - *required*
     - Filename of the calibration spec CSV (must be in a configs directory).
   * - ``helper_module``
     - ``str`` or ``null``
     - ``null``
     - Python file path (e.g., ``my_helper.py``) or importable module path.
       Functions defined in this module are available as evaluation context for
       ``model_value`` and ``target_value`` expressions.
   * - ``submodel_max_iterations``
     - ``int``
     - ``1``
     - Maximum number of inner-loop iterations per component per global
       iteration. The component re-runs from its prior checkpoint each iteration.
   * - ``survey_file``
     - ``str``
     - *required*
     - Survey data CSV filename. Made available via
       ``component_settings.survey_file`` in the expression context.
   * - ``reports.generic``
     - ``bool``
     - ``True``
     - Write a generic CSV report each iteration.
   * - ``reports.bespoke``
     - ``str`` or ``null``
     - ``null``
     - Name of a function in the helper module to call for custom reporting. The
       function receives the full evaluation context dict.


Calibration Spec CSV
====================

The calibration spec is a CSV file that defines — for each coefficient — how to
compute model and target values, the adjustment method, convergence tolerance,
and bounds. Each row represents one coefficient to calibrate.

Required Columns
-----------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Column
     - Type
     - Description
   * - ``description``
     - ``str``
     - Human-readable label for the coefficient (used in reports and plots).
   * - ``coefficient``
     - ``str``
     - Name of the coefficient in the component's coefficients CSV file. Must
       also appear in the component's utility specification.
   * - ``model_value``
     - numeric or expression
     - The model's current output for the target metric. Can be a literal number
       or a Python expression evaluated against the expression context.
   * - ``target_value``
     - numeric or expression
     - The observed/survey target. Can be a literal number or a Python expression.
   * - ``hold_fast``
     - ``bool``
     - If ``True``, the coefficient is evaluated but never updated. Useful for
       monitoring convergence of a held coefficient.
   * - ``min``
     - numeric
     - Lower bound for the coefficient value. Leave blank for no bound.
   * - ``max``
     - numeric
     - Upper bound for the coefficient value. Leave blank for no bound.
   * - ``damping``
     - numeric
     - Damping factor (≥ 0) applied to the computed delta. ``1.0`` means no
       damping; values < 1 slow convergence for stability.
   * - ``method``
     - ``str``
     - Adjustment method: ``log_ratio`` or ``odds_ratio``.
   * - ``tolerance``
     - numeric
     - Absolute difference threshold. A coefficient is "converged" when
       ``|target_value - model_value| <= tolerance``.

Optional Columns
-----------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 10 55

   * - Column
     - Type
     - Default
     - Description
   * - ``default_increment``
     - numeric
     - ``2.0``
     - Fallback delta when the chosen method encounters invalid inputs (e.g.,
       zero or negative values for ``log_ratio``).

Comment Rows
-------------

Lines beginning with ``#`` are treated as comments and ignored. This is useful
for temporarily disabling individual coefficients without removing them.

Example: Auto Ownership
-------------------------

.. code-block:: text

   description,coefficient,model_value,target_value,hold_fast,min,max,damping,method,tolerance
   0 auto ownership share,coef_calib_auto_0,len(households[households.auto_ownership==0]) / len(households),0.06812,FALSE,-5,5,1,log_ratio,0.02
   2 auto ownership share,coef_calib_auto_2,len(households[households.auto_ownership==2]) / len(households),0.348413,FALSE,-5,5,1,log_ratio,0.01
   3 auto ownership share,coef_calib_auto_3,len(households[households.auto_ownership==3]) / len(households),0.13718,FALSE,-5,5,1,log_ratio,0.01
   4 auto ownership share,coef_calib_auto_4,len(households[households.auto_ownership==4]) / len(households),0.057501,FALSE,-5,5,1,log_ratio,0.01

In this example:

- ``model_value`` is a Python expression that computes the share of households
  with a given auto ownership level from the pipeline ``households`` table.
- ``target_value`` is a fixed numeric target derived from observed survey data.
- Each coefficient is bounded to ``[-5, 5]`` with no damping (``1.0``) and uses
  the ``log_ratio`` method.

Example: Workplace Location (Using Helper Functions)
-----------------------------------------------------

.. code-block:: text

   description,coefficient,model_value,target_value,hold_fast,min,max,damping,method,tolerance
   Distance 0 to 2 mi share,coef_calib_dist_0_2,"summarize_model(context, min_dist=.0, max_dist=.5)",0.05,FALSE,-5,5,1,log_ratio,0.02
   Distance 5 to 15 mi share,coef_calib_dist_5_15,"summarize_model(context, min_dist=.5, max_dist=1)",0.25,FALSE,-5,5,1,log_ratio,0.01
   Distance 15+ mi share,coef_calib_dist_15_up,"summarize_model(context, min_dist=2, max_dist=999)",0.1,FALSE,-5,5,1,log_ratio,0.01

Here ``summarize_model()`` is a function defined in the helper module. Both
``model_value`` and ``target_value`` can call helper functions via ``context``.


Adjustment Methods
==================

``log_ratio``
--------------

Computes the coefficient delta as:

.. math::

   \Delta = \ln\!\left(\frac{\text{target_value}}{\text{model_value}}\right) \times \text{damping}

Requires both ``model_value`` and ``target_value`` to be positive. If either is
zero or negative, falls back to ``default_increment`` (positive or negative
depending on direction).

Best suited for **share-based targets** where both model and target represent
proportions.

``odds_ratio``
---------------

Computes the coefficient delta as:

.. math::

   \Delta = \ln\!\left(\frac{T \cdot M - T}{T \cdot M - M}\right) \times \text{damping}

where :math:`T` = ``target_value`` and :math:`M` = ``model_value``.

Falls back to ``default_increment`` when the numerator or denominator is
non-positive. Appropriate for **logit-based models** where coefficients operate
in utility space.

Damping
--------

The ``damping`` factor multiplies the computed delta. Use values less than 1.0
(e.g., ``0.5``) to slow convergence and improve stability when coefficients
oscillate. A value of ``1.0`` applies the full computed adjustment.

Bounds
-------

When ``min`` and/or ``max`` are specified, the candidate coefficient value is
clamped after the delta is applied. The iteration record tracks whether clamping
occurred via ``at_min`` and ``at_max`` flags.


Expression Context
==================

The ``model_value`` and ``target_value`` fields in the calibration spec can
contain Python expressions. These expressions are evaluated with access to the
following context variables:

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Variable
     - Type
     - Description
   * - ``state``
     - ``workflow.State``
     - The ActivitySim state object.
   * - ``np``
     - module
     - NumPy.
   * - ``pd``
     - module
     - pandas.
   * - ``households``
     - ``DataFrame``
     - The ``households`` pipeline table (if it exists).
   * - ``persons``
     - ``DataFrame``
     - The ``persons`` pipeline table (if it exists).
   * - ``tours``
     - ``DataFrame``
     - The ``tours`` pipeline table (if it exists).
   * - ``trips``
     - ``DataFrame``
     - The ``trips`` pipeline table (if it exists).
   * - *(other tables)*
     - ``DataFrame``
     - Any other registered pipeline table.
   * - ``network_los``
     - object
     - Network level of service (if available).
   * - ``skim_dict``
     - ``SkimDict``
     - Default skim dictionary (if available).
   * - ``component_output_dir``
     - ``Path``
     - Output directory for the current component (``output/calibration/<component>/``).
   * - ``component_settings``
     - object
     - The ``CalibrationComponentSettings`` for the current component.
   * - ``context``
     - ``dict``
     - Self-reference to the full context dict, for passing to helper functions.
   * - *(helper symbols)*
     - varies
     - All public names from the helper module (functions, variables, classes).

Writing Expressions
--------------------

**Inline expressions** work well for simple share computations:

.. code-block:: python

   len(households[households.auto_ownership == 0]) / len(households)

**Helper function calls** are preferred for complex logic:

.. code-block:: python

   summarize_model(context, min_dist=0.5, max_dist=1.0)

The ``context`` variable gives helper functions access to everything: pipeline
tables, skims, pandas, numpy, and other helpers.


Helper Modules
==============

A helper module is a Python file placed in the configs directory (or an
importable Python module) that provides functions for use in calibration spec
expressions and custom reports.

Loading
--------

Specify the helper module in ``calibration.yaml``:

.. code-block:: yaml

   model_settings:
     workplace_location:
       helper_module: workplace_location_calib_helper.py

File paths ending in ``.py`` are loaded from the config directory. Other values
are treated as Python import paths (e.g., ``mypackage.calibration_helpers``).

All public names (functions, variables, classes) from the module are injected
into the expression context.

Example Helper Module
----------------------

.. code-block:: python

   """Workplace location calibration helpers."""

   import pandas as pd


   def compute_distances(context, origins, destinations):
       """Compute distances between zones using the skim dictionary."""
       return context["skim_dict"].lookup(origins, destinations, "DIST")


   def summarize_model(context, min_dist=1, max_dist=2):
       """Compute the share of workers with workplace distance in [min_dist, max_dist)."""
       persons = context["persons"]
       workers = persons[persons["workplace_zone_id"] > 0]
       distances = compute_distances(
           context, workers["home_zone_id"], workers["workplace_zone_id"]
       )
       mask = (distances >= min_dist) & (distances < max_dist)
       return len(distances[mask]) / len(distances) if len(distances) > 0 else 0


   def report_workplace_location(context):
       """Custom bespoke report called after each iteration."""
       import matplotlib.pyplot as plt
       import os

       # Custom plotting logic here...
       plt.savefig(os.path.join(context["component_output_dir"], "custom_report.png"))
       plt.close()

Bespoke Reports
----------------

If ``reports.bespoke`` is set to a function name (e.g.,
``report_workplace_location``), that function is called after each component
iteration with the full expression context dict. Use this for custom plots,
tables, or dashboards beyond what the generic report provides.


Output Files
============

All calibration output is written under ``output/calibration/``.

Global Files
-------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - File
     - Description
   * - ``calibration_progress.json``
     - Tracks ``next_global_iteration`` for crash recovery. If a run is
       interrupted, restarting will resume from the last completed global
       iteration.
   * - ``calibration_iteration_records.csv``
     - Appended per-coefficient detail for every iteration across all
       components. Columns include ``global_iter``, ``component_iter``,
       ``coefficient``, ``target_value``, ``model_value``, ``difference``,
       ``pct_difference``, ``prev_coefficient``, ``next_coefficient``,
       ``converged``, ``at_min``, ``at_max``.
   * - ``calibration_iteration_summary.csv``
     - One row per component iteration with summary statistics:
       ``max_difference``, ``max_change``, ``num_converged``,
       ``num_unconverged``.
   * - ``final_calibrated_coefficients.csv``
     - Combined snapshot of all calibrated coefficients at the end of the run,
       with ``component`` and ``coefficient_name`` columns.

Per-Component Files (``calibration/<component_name>/``)
--------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - File
     - Description
   * - ``calibration_iteration_records.csv``
     - Component-specific iteration records (same schema as the global file).
   * - ``generic_report.csv``
     - Simplified iteration report with ``description``, ``difference``,
       ``pct_difference``, and ``converged``.
   * - ``coefficient_progress_set_N.png``
     - Line plot of coefficient values across iterations (up to 10 coefficients
       per plot).
   * - ``final_components_set_N.png``
     - Bar chart comparing final ``target_value`` vs ``model_value``.
   * - ``final_pct_change_set_N.png``
     - Bar chart of final percent difference between model and target.

Updated Config Files
---------------------

Coefficient CSV files in the configs directory are **updated in-place** after
each iteration. This means:

- The calibrated coefficients persist across runs.
- You can inspect intermediate coefficient values at any time.
- To reset, restore the original coefficient files from version control.


Crash Recovery
==============

Calibration progress is persisted to ``calibration_progress.json`` after each
completed global iteration. If a run is interrupted:

1. The coefficient files on disk reflect the state at the last completed iteration.
2. Restarting ``activitysim run`` with the same configuration will resume from
   the ``next_global_iteration`` recorded in the progress file.

To force a fresh start, delete ``output/calibration/calibration_progress.json``
and restore original coefficient files.


Multiprocess Mode
=================

Calibration works with ActivitySim's multiprocess execution mode. When
``multiprocess: True`` is set in ``settings.yaml``:

- Shared resources (skim buffers, shadow pricing buffers) are allocated **once**
  at calibration start and reused across all iterations, avoiding repeated
  expensive allocations.
- Precursor, intermediate, and subsequent model steps use ActivitySim's normal
  multiprocess orchestration.
- Calibrated component re-runs use direct apportion → simulate → coalesce
  orchestration with explicit checkpoint control, ensuring correct state
  restoration on each iteration.
- After each multiprocess run, the coalesced pipeline is loaded back into the
  parent process so calibration expressions can evaluate model outputs.

No special configuration is needed — the calibration framework automatically
respects ``num_processes``, ``multiprocess_steps``, ``slice``, and
``chunk_size`` from the existing ``settings.yaml``.


Convergence
===========

A coefficient is considered **converged** when:

.. math::

  \text{target_value} - \text{model_value} \leq \text{tolerance}

A component is converged when **all** of its coefficients are converged. The
component inner loop stops early upon convergence.

The overall calibration run completes after all ``global_iterations`` have
executed. Global convergence is tracked but does not currently trigger early
termination of the outer loop — use ``global_iterations`` to control the total
number of passes.


Coefficient Requirements
========================

Calibration coefficients must satisfy these requirements:

1. **Present in utility specification**: Every ``coefficient`` in the calibration
   spec must appear as a token in the component's utility expression CSV (the
   files referenced by settings keys ending in ``SPEC``). A validation error is
   raised at startup if any are missing.

2. **Present in coefficients file**: If a calibration coefficient is not found in
   the component's coefficients CSV, it is automatically added with an initial
   value of ``0.0`` and a warning is logged.

3. **Numeric values**: All coefficient values must be numeric. Non-numeric values
   raise an error.


Working Example
===============

A complete working example is included in the repository at::

   activitysim/examples/prototype_mtc/configs_calibration/

This example calibrates three components — ``workplace_location``,
``auto_ownership_simulate``, and ``tour_mode_choice_simulate`` — and includes:

- ``calibration.yaml`` — top-level configuration
- ``*_calibration.csv`` — calibration specs with inline expressions and helper
  function calls
- ``*_calib_helper.py`` — helper modules with custom summary functions and
  bespoke reports
- ``*_coefficients.csv`` — initial coefficient values

To run the example:

.. code-block:: bash

   activitysim run \
     -c activitysim/examples/prototype_mtc/configs_calibration \
     -c activitysim/examples/prototype_mtc/configs \
     -d activitysim/examples/prototype_mtc/data \
     -o output


Tips
====

- **Start with** ``submodel_max_iterations: 1`` and a small number of
  ``global_iterations`` to verify your setup before committing to a long
  calibration run.
- **Use** ``damping < 1.0`` if coefficients oscillate between iterations.
- **Set** ``hold_fast: True`` on reference-category coefficients that should be
  fixed (e.g., the base alternative in a logit model).
- **Comment out rows** in the calibration spec with ``#`` to temporarily exclude
  coefficients without modifying the file structure.
- **Check** ``calibration_iteration_records.csv`` to diagnose convergence issues
  — look for coefficients hitting bounds (``at_min``/``at_max``) or oscillating.
- **Use** ``complete_steps: true`` if you need full model outputs each iteration
  (e.g., for trip generation downstream of mode choice).
- **Version control your coefficient files** so you can diff changes and reset to
  initial values.
- **Use the** ``resume_after`` **setting** to skip expensive upstream steps (like
  skims loading or accessibility computation) that don't change across
  calibration iterations.
