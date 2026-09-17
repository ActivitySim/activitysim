# Workflow State

The general framework of each ActivitySim model is defined within an encapsulated
[`State`](core-workflow-api) object.  This object maintains references to data and
model structures in a well-defined context, and allow the user to pass that context
around to the various functions and methods that progressively build up the simulated
activity patterns.

The [`State`](core-workflow-api) object replaces the ORCA framework, and allows for data
from multiple models, or multiple versions of the same model, to co-exist in a single
Python instance simultaneously.  The state contains references for overall model
settings, the network level of service features, as well as the state of the
simulated households, persons, etc.  Extensive documentation on the
[API](core-workflow-api.md) for working with the state is available.

The [`State`](core-workflow-api) class for ActivitySim also offers hooks for a
few fundamental elements:

- [**Steps**](core-workflow-steps), also referred to as "model components",
    which represent the fundamental mathematical building blocks of an ActivitySim
    model.  Each component contains instructions for incrementally augmenting the
    state of the model, generally by adding columns or rows to an existing table,
    although components are not limited to that and can potentially do other things
    as well.
- [**Data Tables**](core-workflow-table), sometimes referred to in older
    documentation sections as "pipeline tables".  These tables include households,
    persons, trips, tours, and potentially other tables that represent aspects of
    the simulated agents.
- [**Other Cached Objects**](core-workflow-table.md#other-cached-objects), which can
    be any arbitrary Python object that can be created programmatically and stored
    in the state's context dictionary.


```{eval-rst}
.. toctree::
   :maxdepth: 1
   :hidden:

   core-workflow-api
   core-workflow-steps
   core-workflow-table
```
