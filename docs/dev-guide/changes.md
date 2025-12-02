# Change Log

This document describes significant changes to ActivitySim.  This includes
major new features that may require modifications to existing model configurations
or code to utilize, as well as breaking changes that may cause existing model
configurations or code to fail to run correctly.


## Upcoming Changes

This section describes changes that are implemented in current development 
branch (i.e., the main branch on GitHub), but not yet released in a stable version 
of ActivitySim.  See below under the various version headings for changes in 
released versions.


## v1.5.1

This release includes a handful of minor updates and fixes, as well as enhancements 
to the ActivtySim documentation. Users should generally not expect any breaking 
changes relative to v1.5.0, except that when running a simulation there will be a 
significant reduction in logging messages displayed on screen and written to the run log.


## v1.5

This release includes most of the new features and enhancements developed as part 
of the Phase 10 work.

### Preprocessing & Annotation

We have expanded preprocessing & annotation functionality, which is now standardized 
in formatting and available on most model components.  Existing model implementations 
may need to make minor upgrades to model configuration files to conform with the new 
standardized formatting. 

### Estimation Mode

Estimation mode has been updated to work with Larch v6.  This new version of Larch 
is modernized and more stable across platforms, and is more consistent with ActivitySim 
spec files (as both are now built on Sharrow).  The overall workflow for re-estimating 
model parameters is very similar to before, but users will need to use Larch v6 instead 
of Larch v5.  In addition, some new capabilities have been added for modifying model 
specifications in Larch (instead of re-running ActivitySim).

### Using UV for Dependency Management

Beginning with version 1.5, ActivitySim uses [UV](https://uv.dev/) for dependency
management.  UV is a modern dependency management tool that is designed to be
simple to use and easy to understand.  See [Installing ActivitySim](Installing-ActivitySim)
for details on how to install ActivitySim using UV.

### Skim Naming Conflict Resolution

The SkimDataset structure (required when using sharrow, optional in legacy mode) 
requires every skim variable to have a unique name. It also merges OMX variables 
based on time period, so that e.g. `BIKETIME__AM` and `BIKETIME__PM`, which would 
be 2-d arrays in the OMX file, become just two different parts of a 3-d array 
called `BIKETIME` in the SkimDataset. This is problematic when the skims also 
contain a 2-d array called `BIKETIME`, as that has no temporal dimension, and it 
gets loaded into a 2-d array in the SkimDataset, with the same name as the 3-d array, 
and thus one is overwritten and lost.

ActivitySim now includes a skims input check to identify this overwriting condition, 
and raise an error if it is happening, so that the user can correct the condition 
via (1) the `omx_ignore_patterns` setting, (2) revising the skim generation process 
to not create the overlapping named skims in the file in the first place, 
or (3) renaming one or both skims if the users actually wants both skims variables 
in the model. The error message generated includes a link to instructions and 
discussion of these alternatives.

### Settings Checker

A new settings checker has been added to validate model configuration files
before running the model.  This tool checks for common configuration errors,
such as missing required settings, incorrect data types, and invalid values.
This can help users identify and fix configuration issues before running
the model, which can save time and effort.  In prior versions, configuration
errors would often only be discovered when the model was run, which could
lead to long run times before the error was encountered.

### Expression Profiling (legacy mode only)

ActivitySim now includes a new performance profiling feature for expression evaluation
in the ActivitySim framework. The feature allows developers to track and log the
runtime of individual expressions, providing insights into potential bottlenecks
in complex models. Key changes include the integration of a performance timer,
updates to various core functions to support profiling, and new configuration
settings for controlling profiling behavior.
See [Expression Profiling](Expression-Profiling) for details.

### Telecommute Status Model

A new telecommute status model component has been added to ActivitySim. This component
models the telecommute status of workers, which can be used to determine
whether a worker telecommutes full-time, part-time, or not at all. A simple 
implementation of the telecommute status model can be based on the worker's telecommute 
frequency. For example, if a worker telecommutes 4 days a week, then there is 
a 80% probability for them to telecommute on the simulation day. The telecommute 
status model software can accommodate more complex model forms if needed. An example 
telecommute status model specification can be found in 
[ActivitySim/sandag-abm3-example#30](https://github.com/ActivitySim/sandag-abm3-example/pull/30).


## v1.4

### Improved Estimation Mode

Version 1.4 includes several improvements to the estimation mode, including:

- The ability to run estimation mode in parallel, which can significantly
  speed up the estimation process for large models.
- The ability to modify the model specification and coefficients file(s) for
  the estimated submodels without re-running ActivitySim, which allows for more
  flexibility in the estimation process.
- Other improvements to the estimation mode workflow, including better error
  handling, logging, and model evaluation tools.


## v1.3

### New Canonical Examples

Beginning with version 1.3, ActivitySim provides two supported "canonical" example
implementations:

- the [SANDAG Model](https://github.com/ActivitySim/sandag-abm3-example) is a two-zone
  model based on the SANDAG ABM3 model, and
- the [MTC Model](https://github.com/ActivitySim/activitysim-prototype-mtc) is a
  one-zone model based on the MTC's Travel Model One.

Each example implementation includes a complete set of model components, input data,
and configuration files, and is intended to serve as a reference for users to build
their own models. They are provided as stand-alone repositories, to highlight the
fact that model implementations are separate from the ActivitySim core codebase,
and to make it easier for users to fork and modify the examples for their own use
without needing to modify the ActivitySim core codebase. The examples are maintained
by the ActivitySim Consortium and are kept up-to-date with the latest version of
ActivitySim.

```{note}
The two example models are not identical to the original agency models from which
they were created. They are generally similar to those models, and have been calibrated
and validated to reproduce reasonable results. They are intended to demonstrate the
capabilities of ActivitySim and to provide a starting point for users to build their own
models. However, they are not intended to be used as-is for policy analysis or forecasting.
```

### Logging

The reading of YAML configuration files has been modified to use the "safe" reader,
which prohibits the use of arbitrary Python code in configuration files. This is a
security enhancement, but it requires some changes to the way logging is configured.

In previous versions, the logging configuration file could contain Python code to
place log files in various subdirectories of the output directory, which might
vary for different subprocesses of the model, like this:

```yaml
logging:
  handlers:
    logfile:
      class: logging.FileHandler
      filename: !!python/object/apply:activitysim.core.config.log_file_path ['activitysim.log']
      mode: w
      formatter: fileFormatter
      level: NOTSET
```

In the new version, the use of `!!python/object/apply` is prohibited. Instead of using
this directive, the `log_file_path` function can be invoked in the configuration file
by using the `get_log_file_path` key, like this:

```yaml
logging:
  handlers:
    logfile:
      class: logging.FileHandler
      filename:
        get_log_file_path: activitysim.log
      mode: w
      formatter: fileFormatter
      level: NOTSET
```

Similarly, previous use of the `if_sub_task` directive in the logging level
configuration like this:

```yaml
logging:
  handlers:
    console:
      class: logging.StreamHandler
      stream: ext://sys.stdout
      level: !!python/object/apply:activitysim.core.mp_tasks.if_sub_task [WARNING, NOTSET]
      formatter: elapsedFormatter
```

can be replaced with the `if_sub_task` and `if_not_sub_task` keys, like this:

```yaml
logging:
  handlers:
    console:
      class: logging.StreamHandler
      stream: ext://sys.stdout
      level:
        if_sub_task: WARNING
        if_not_sub_task: NOTSET
      formatter: elapsedFormatter
```

For more details, see [logging](Logging).

### Chunking

Version 1.3 introduces a new "[explicit](Explicit-Chunking)" chunking mechanism.

Explicit chunking is simpler to use and understand than dynamic chunking, and in
practice has been found to be more robust and reliable. It requires no "training"
and is activated in the top level model configuration file (typically `settings.yaml`):

```yaml
chunk_training_mode: explicit
```

Then, for model components that may stress the memory limits of the machine,
the user can specify the number of choosers in each chunk explicitly, either as an integer
number of choosers per chunk, or as a fraction of the overall number of choosers.
This is done by setting the `explicit_chunk` configuration setting in the model
component's settings.  For this setting, integer values greater than or equal to 1
correspond to the number of chooser rows in each explicit chunk. Fractional values
less than 1 correspond to the fraction of the total number of choosers.
If the `explicit_chunk` value is 0 or missing, then no chunking
is applied for that component.  The `explicit_chunk` values in each component's
settings are ignored if the `chunk_training_mode` is not set to `explicit`.
Refer to each model component's configuration documentation for details.

Refer to code updates that implement explicit chunking for accessibility in
[PR #759](https://github.com/ActivitySim/activitysim/pull/759), for
vehicle type choice, non-mandatory tour frequency, school escorting, and
joint tour frequency in  [PR #804](https://github.com/ActivitySim/activitysim/pull/804),
and all remaining interaction-simulate components in
[PR #870](https://github.com/ActivitySim/activitysim/pull/870).

### Automatic dropping of unused columns

Variables that are not used in a model component are now automatically dropped
from the chooser table before the component is run. Whether a variable is deemed
as "used" is determined by a text search of the model component code and specification
files for the variable name.  Dropping unused columns can be disabled by setting
[`drop_unused_columns`](activitysim.core.configuration.base.ComputeSettings.drop_unused_columns)
to `False` in the [`compute_settings`](activitysim.core.configuration.base.ComputeSettings)
for any model component, but by default this setting is `True`, as it can result in a
significant reduction in memory usage for large models.

Dropping columns may also cause problems if the model is not correctly configured.
If it is desired to use this feature, but some required columns are being dropped
incorrectly, the user can specify columns that should not be dropped by setting the
[`protect_columns`](activitysim.core.configuration.base.ComputeSettings.protect_columns)
setting under [`compute_settings`](activitysim.core.configuration.base.ComputeSettings).
This allows the user to specify columns that should not be dropped, even if they are
not apparently used in the model component.  For [example](https://github.com/ActivitySim/activitysim/blob/67820ad32789f59217608b5311e9a2a322d029ed/activitysim/examples/prototype_sandag_xborder/configs/tour_od_choice.yaml#L59-L61):

```yaml
compute_settings:
  protect_columns:
  - origin_destination
```

Code updates to drop unused columns are in
[PR #833](https://github.com/ActivitySim/activitysim/pull/833) and to protect
columns in [PR #871](https://github.com/ActivitySim/activitysim/pull/871).

### Automatic conversion of string data to categorical

Version 1.3 introduces a new feature that automatically converts string data
to categorical data.  This reduces memory usage and speeds up processing for
large models.  The conversion is done automatically for string columns
in most chooser tables.

To further reduce memory usage, there is also an optional downcasting of numeric
data available. For example, this allows storing integers that never exceed 255
as `int8` instead of `int64`.  This feature is controlled by the `downcast_int`
and `downcast_float` settings in the top level model configuration file (typically
`settings.yaml`).   The default value for these settings is `False`, meaning that
downcasting is not applied.  It is recommended to leave these settings at their
default values unless memory availability is severely constrained, as downcasting
can cause numerical instability in some cases.  First, changing the precision of
numeric data could cause results to change slightly and impact a previous calibrated
model result.  Second, downcasting to lower byte data types, e.g., int8, can cause
numeric overflow in downstream components if the numeric variable is used in
mathematical calculations that would result in values beyond the lower bit width
limit (e.g. squaring the value). If downcasting is desired, it is strongly recommended
to review all model specifications for compatability, and to review model results
to verify if the changes are acceptable.

See code updates in [PR #782](https://github.com/ActivitySim/activitysim/pull/782)
and [PR #863](https://github.com/ActivitySim/activitysim/pull/863)

### Alternatives preprocessors for trip destination.

Added alternatives preprocessor in
[PR #865](https://github.com/ActivitySim/activitysim/pull/869),
and converted to separate preprocessors for sample (at the TAZ level) and
simulate (at the MAZ level for 2 zone systems) in
[PR #869](https://github.com/ActivitySim/activitysim/pull/869).

### Per-component sharrow controls

This version adds a uniform interface for controlling sharrow optimizations
at the component level. This allows users to disable sharrow entirely,
or to disable the "fastmath" optimization for individual components.
Controls for sharrow are set in each component's settings under `compute_settings`.
For example, to disable sharrow entirely for a component, use:

```yaml
compute_settings:
  sharrow_skip: true
```

This overrides the global sharrow setting, and is useful if you want to skip
sharrow for particular components, either because their specifications are
not compatible with sharrow or if the sharrow performance is known to be
poor on this component.

When a component has multiple subcomponents, the `sharrow_skip` setting can be
a dictionary that maps the names of the subcomponents to boolean values.
For example, in the school escorting component, to skip sharrow for an
OUTBOUND and OUTBOUND_COND subcomponent but not the INBOUND subcomponent,
use the following settings:

```yaml
compute_settings:
  sharrow_skip:
    OUTBOUND: true
    INBOUND: false
    OUTBOUND_COND: true
```

The `compute_settings` can also be used to disable the "fastmath" optimization.
This is useful if the component is known to have numerical stability issues
with the fastmath optimization enabled, usually when the component potentially
works with data that includes `NaN` or `Inf` values. To disable fastmath for
a component, use:

```yaml
compute_settings:
  fastmath: false
```

Code updates that apply these settings are in
[PR #824](https://github.com/ActivitySim/activitysim/pull/824).

### Configuration validation

Version 1.3 adds a configuration validation system using the Pydantic library.
Previously, the YAML-based configuration files were allowed to contain arbitrary
keys and values, which could lead to errors if the configuration was not correctly
specified. The new validation system checks the configuration files for correctness,
and provides useful error messages if the configuration is invalid.  Invalid
conditions include missing required keys, incorrect data types, and the presence
of unexpected keys.  Existing models may need to be cleaned up (i.e. extraneous settings
in config files removed) to conform to the new validation system.

See [PR #758](https://github.com/ActivitySim/activitysim/pull/758) for code updates.

### Input checker

Version 1.3 adds an input checker that verifies that the input data is consistent
with expectations. This tool can help identify problems with the input data before
the model is run, and can be used to ensure that the input data is correctly
formatted and complete.

See [PR #753](https://github.com/ActivitySim/activitysim/pull/753) for code updates.

### Removal of orca dependency

This new version of ActivitySim does not use `orca` as a dependency, and thus does
not rely on orca’s global state to manage data. Instead, a new [`State`](activitysim.core.workflow.State)
class is introduced, which encapsulates the current state of a simulation including
all data tables. This is a significant change “under the hood”, which may be
particularly consequential for model that use “extensions” to the ActivitySim framework.
See [PR #654](https://github.com/ActivitySim/activitysim/pull/654) for code updates.

## v1.2

The [v1.2](https://github.com/ActivitySim/activitysim/releases/tag/v1.2.0) release
includes all updates and enhancements complete in the ActivitySim Consortium's Phase 7
development cycle, including:

- Sharrow performance enhancement
- Explicit school escorting
- Disaggregate accessibility
- Simulation-based shadow pricing
