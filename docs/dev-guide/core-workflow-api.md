# State API

```{eval-rst}
.. currentmodule:: activitysim.core.workflow

.. autosummary::
    :toctree: _generated
    :recursive:

    State
```

## Constructors

```{eval-rst}
.. autosummary::
    :toctree: _generated
    :recursive:

    State.__init__
    State.make_default
    State.make_temp
    create_example
```

## Model Setup

```{eval-rst}
.. autosummary::

    State.init_state
    State.import_extensions
    State.initialize_filesystem
    State.default_settings
    State.load_settings
    State.settings
    State.filesystem
    State.network_settings
```

### Importing extensions

Register custom model packages with `State.import_extensions` before running
the model. Relative package paths are interpreted relative to the state's
working directory, which need not be the Python process's current directory:

```python
state = State.make_default("/path/to/model")
state.import_extensions("extensions")
state.run.all()
```

Absolute package paths, path-like objects, and lists of extensions are also
accepted. Python module names, including dotted names such as `my_package.models`,
are supported; for a single Python file, omit the `.py` suffix. Use a unique
package name for each extension, as Python caches imports by module name.

The CLI uses the same loader, for example
`activitysim run -w /path/to/model --ext extensions`, or
`activitysim run -c model/configs -d model/data -o output --ext model/extensions`.
Both interfaces retain importable module names in `imported_extensions`, so
existing consumers can use them with `importlib.import_module`, including to
load an extension's `settings_checker` submodule. Absolute import locations are
stored separately in internal worker metadata, allowing multiprocessing workers
to reimport extensions even from another current directory. Package paths may include `./` or a trailing directory separator.
These rules are the same for single-process and multiprocessing runs.

`append=False` replaces the registered extension list; it does not unload modules
already imported into the Python process. Importing extension code temporarily
adds its parent directory to `sys.path`, which is restored even if import fails.



## Basic Context Management

The most basic function of the `State` object is to serve as a defined
namespace for storing model-relevant variables.  This includes the top-level
model settings, data tables, skims, and any other Python variables
that represent the current state of a particular modeling system (or when
multiprocessing, sub-system).  Below are the basic methods to get and set values
in this context in their "raw" form, with minimal additional processing.

```{eval-rst}
.. autosummary::

    State.get
    State.set
    State.drop
    State.access
    State.get_injectable
    State.add_injectable
```


## Data Access and Manipulation

In addition to "raw" access to context variable, several methods are provided
to simplify different kinds access to the "tables" that represent the
simulation inputs and outputs of ActivitySim.  We say "tables" here in the
abstract sense -- historically these tables have been stored internally by
ORCA as `pandas.DataFrame`s, but the exact internal storage format is abstracted
away here in favor of providing access to the data in several specific formats.

```{eval-rst}

.. rubric:: Methods

.. autosummary::

    State.get_dataset
    State.get_dataframe
    State.get_dataarray
    State.get_dataframe_index_name
    State.get_pyarrow
    State.add_table
    State.is_table
    State.registered_tables
    State.get_table

.. rubric:: Accessor

.. autosummary::
    :toctree: _generated2
    :template: autosummary/accessor.rst

    State.dataset
```


## Run

Executing model components is handled by methods in the `run` accessor.

```{eval-rst}

.. rubric:: Accessor

.. autosummary::
    :toctree: _generated2
    :template: autosummary/accessor_callable.rst

    State.run



.. rubric:: Attributes

.. autosummary::
    :toctree: _generated
    :template: autosummary/accessor_attribute.rst

    State.run.heading_level



.. rubric:: Methods

.. autosummary::
    :toctree: _generated
    :template: autosummary/accessor_method.rst

    State.run.by_name
    State.run.all
```

(state-checkpoint)=
## Checkpoints

The `State` object provides access to [checkpointing](checkpointing.md) functions
within the `checkpoint` accessor.

```{eval-rst}

.. rubric:: Accessor

.. autosummary::
    :toctree: _generated2
    :template: autosummary/accessor.rst

    State.checkpoint


.. rubric:: Attributes

.. autosummary::
    :toctree: _generated
    :template: autosummary/accessor_attribute.rst

    State.checkpoint.last_checkpoint
    State.checkpoint.checkpoints
    State.checkpoint.store


.. rubric:: Methods

.. autosummary::
    :toctree: _generated
    :template: autosummary/accessor_method.rst

    State.checkpoint.store_is_open
    State.checkpoint.open_store
    State.checkpoint.close_store
    State.checkpoint.add
    State.checkpoint.list_tables
    State.checkpoint.load
    State.checkpoint.get_inventory
    State.checkpoint.restore
    State.checkpoint.restore_from
    State.checkpoint.check_against
    State.checkpoint.cleanup
    State.checkpoint.load_dataframe
    State.checkpoint.last_checkpoint_name
    State.checkpoint.is_readonly
    State.checkpoint.default_pipeline_file_path

```


## Tracing

```{eval-rst}

.. rubric:: Attributes

.. autosummary::
    :toctree: _generated
    :template: autosummary/accessor_attribute.rst

    State.tracing.traceable_tables
    State.tracing.traceable_table_ids
    State.tracing.traceable_table_indexes
    State.tracing.run_id
    State.tracing.validation_directory



.. rubric:: Methods

.. autosummary::
    :toctree: _generated
    :template: autosummary/accessor_method.rst

    State.tracing.initialize
    State.tracing.register_traceable_table
    State.tracing.deregister_traceable_table
    State.tracing.write_csv
    State.tracing.trace_df
    State.tracing.trace_interaction_eval_results
    State.tracing.get_trace_target
    State.tracing.trace_targets
    State.tracing.has_trace_targets
    State.tracing.dump_df
    State.tracing.delete_output_files
    State.tracing.delete_trace_files
```


## Logging

```{eval-rst}

.. rubric:: Methods

.. autosummary::
    :toctree: _generated
    :template: autosummary/accessor_method.rst

    State.logging.config_logger
    State.logging.rotate_log_directory
```


## Reporting

```{eval-rst}

.. rubric:: Accessor

.. autosummary::
    :toctree: _generated2
    :template: autosummary/accessor.rst

    State.report


.. rubric:: Methods

.. autosummary::
    :toctree: _generated
    :template: autosummary/accessor_method.rst

    State.report.nominal_distribution
    State.report.ordinal_distribution
    State.report.histogram
```


## Extending

Methods to extend ActivitySim's functionality are available under the `extend`
accessor.

```{eval-rst}

.. rubric:: Accessor

.. autosummary::
    :toctree: _generated2
    :template: autosummary/accessor.rst

    State.extend


.. rubric:: Methods

.. autosummary::
    :toctree: _generated
    :template: autosummary/accessor_method.rst

    State.extend.declare_table
```
