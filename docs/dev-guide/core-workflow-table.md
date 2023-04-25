# Workflow Tables

## Standard Tables

An ActivitySim table definition is written as a Python function with a
`workflow.table` decorator:

```python
import pandas as pd
from activitysim.core import workflow

@workflow.table
def households(state: workflow.State) -> pd.DataFrame:
    df = pd.DataFrame(...)
    # do something to set up table here
    return df
```

Similar to a typical Python class method, the first argument to a workflow table
function is always a reference to a [`State`](core-workflow-api.md)
object named `state`.  Unlike a typical Python class method, this is rigorously
enforced -- if you decorate a function as a `workflow.table` and the first
argument is not named `state` a `TypeError` will be raised.

For most tables, the initialization of the table will be defined by values in
the {py:class}`Settings <activitysim.core.configuration.Settings>`, and there will
be no other function arguments.

If table initialization does require access to other tables (e.g. the *vehicles*
table needs access to the *households* table to be initialized) then other tables
can be provided as matching-named arguments, in the same way as
[`workflow.step`](core-workflow-steps) functions.

The `workflow.table` function should return a `pandas.DataFrame` or
`xarray.Dataset` representation of the table.  When this function is called
automatically by the processes that orchestrate execution, this object will be
stored in the state's context as the name of the table.


## Temporary Tables

In addition to the main `workflow.table` decorator, there is also a similar
`workflow.temp_table` decorator for temporary tables.

```python
import pandas as pd
from activitysim.core import workflow

@workflow.temp_table
def households_merged(
    state: workflow.State,
    households: pd.DataFrame,
    land_use: pd.DataFrame,
    accessibility: pd.DataFrame,
) -> pd.DataFrame:
    df = pd.DataFrame(...)
    # do something to set up table here
    return df
```

There are two main differences between regular tables and temporary tables:

1. Temporary tables are never checkpointed.

    The supposition for temporary tables is that they are generally large, and
    easy to re-create on the fly, so storing them to disk is wasteful.  Most
    temporary tables in ActivitySim are simply merges of other existing tables,
    although that is not formally a requirement of a temporary tables.

2. Temporary tables are dropped when any predicate argument is changed in the same `State`.

    The *predicates* are all the named arguments of the `workflow.temp_table`
    wrapped function after the `state`.  If another ActivitySim instruction
    triggers an update to *any* of these predicate arguments, the temporary
    table is dropped from the state's context.  It can (presumably) be recreated
    easily from the (now different) predicate values if/when needed for later steps.


(core-workflow-cached-objects)=
## Other Cached Objects

Other arbitrary Python objects can also be generated by functions that are
handled by the same automatic system as tables, using the `workflow.cached_object`
decorator.

```python
from activitysim.core import workflow

@workflow.cached_object
def name_of_object(
    state: workflow.State,
    other_thing: bool = False,
):
    obj = [1,2,3] if other_thing else [7,8,9]  # or any python object
    return obj
```

Similar to temporary tables, these objects are not stored in checkpoint files.
Unlike temporary tables, they are not formally predicated on their arguments, so
for example in the `cached_object` above, a change in the value of `other_thing`
will cause `name_of_object` to be regenerated if it already exists in the state's
context.