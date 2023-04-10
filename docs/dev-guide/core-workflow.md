# Core Workflow

An ActivitySim model is represented by a few fundamental elements:

- **Data tables**, sometimes referred to in older documentation sections as
    "pipeline tables".  These tables include households, persons, trips, tours,
    and potentially other tables that represent aspects of the simulated agents.
- **Network Level of Service**, defined by zones, network skims and representations
    of discrete time steps used for modeling.
- **Components**, which define the mathematical structure of various model steps.

The general framework of each ActivitySim model is defined within an encapsulated
`State` object.

## State API

```{eval-rst}
.. currentmodule:: activitysim.core.workflow
```

### Constructors

```{eval-rst}
.. autosummary::
    :toctree: _generated

    State
    State.make_default
    State.make_temp
```

### Model Setup

```{eval-rst}
.. autosummary::
    :toctree: _generated

    State.init_state
    State.import_extensions
    State.initialize_filesystem
    State.default_settings
    State.load_settings
    State.settings
```



### Data Access and Manipulation

```{eval-rst}
.. autosummary::
    :toctree: _generated

    State.get
    State.set
    State.drop
    State.access
    State.get_injectable
    State.add_injectable
    State.get_dataset
    State.get_dataframe
    State.get_dataarray
    State.get_dataframe_index_name
    State.get_pyarrow
    State.add_table
    State.is_table
    State.registered_tables
    State.get_table

```





### Tracing

```{eval-rst}

.. autosummary::
    :toctree: _generated
    :template: autosummary/accessor_method.rst

    State.tracing.register_traceable_table
```
