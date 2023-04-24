# Workflow Steps

An ActivitySim component is written as a Python function with a `workflow.step`
decorator:

```python
import pandas as pd
from activitysim.core import workflow

@workflow.step
def component_name(
    state: workflow.State,
    named_temp_table: pd.DataFrame,
    named_table: pd.DataFrame,
    cached_object: bool = False,
):
    ... # do something
```

Similar to a typical Python class method, the first argument to a workflow step
is always a reference to a [`State`](core-workflow-api.md) object named `state`.  Unlike a typical Python
class method, this is rigorously enforced -- if you decorate a function as a
`workflow.step` and the first argument is not named `state` a `TypeError` will
be raised.

Similar to the legacy ORCA-based implementation of ActivitySim, when called by
the automated processes that orchestrate component execution, the names of all
subsequent arguments should generally match objects that are expected to be
already stored as keys in the `state`'s context, or have decorated constructors
declared elsewhere in the imported codebase.
