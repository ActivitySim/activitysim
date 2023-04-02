# Core Workflow

An ActivitySim model is represented by a few fundamental elements:

- **Data tables**, sometimes referred to in older documentation sections as
    "pipeline tables".  These tables include households, persons, trips, tours,
    and potentially other tables that represent aspects of the simulated agents.
- **Network Level of Service**, defined by zones, network skims and representations
    of discrete time steps used for modeling.
- **Components**, which define the mathematical structure of various model steps.

## Workflow State

The general framework of each ActivitySim model is defined within an encapsulated
`State` object.

## Workflow API

```{eval-rst}
.. autosummary::
    :toctree: _generated
    :recursive:

    activitysim.core.workflow
```

### Tracing

```{eval-rst}
.. autosummary::
    :toctree: _generated
    :recursive:

    activitysim.core.workflow.tracing
```
