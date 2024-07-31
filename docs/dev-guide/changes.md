# Change Log

This document describes significant changes to ActivitySim.  This includes
major new features that may require modifications to existing model configurations
or code to utilize, as well as breaking changes that may cause existing model
configurations or code to fail to run correctly.

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
