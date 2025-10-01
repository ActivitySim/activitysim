(workflow-steps)=
# Workflow Steps

An ActivitySim component is written as a Python function with a `@workflow.step`
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
) -> None:
    ... # do something
```

Similar to a typical Python class method, the first argument to a
workflow step must be a reference to a [`State`](core-workflow-api.md)
object named `state`.  Unlike a typical Python class method, this is
rigorously enforced -- if you decorate a function as a `workflow.step`
and the first argument is not named `state` a `TypeError` will be raised.

Similar to the legacy ORCA-based implementation of ActivitySim, when called
by the automated processes that orchestrate component execution, the names
of all subsequent arguments should generally match objects that are expected
to be already stored as keys in the `state` context, or have decorated
constructors declared elsewhere in the imported codebase. However, if an
argument is provided with a default value, then the default value is used
unless it is explicitly overloaded in the function call; i.e. the default
value in the function signature takes precedence over any value stored in the
state's context.

Unlike typical Python functions, the type annotations for the decorated
function's arguments are *not* totally ignored, at least when the function is
called via the [`State.run`](activitysim.core.workflow.State.run) mechanisms.
When asking for a data table to be provided, the type annotation is respected
if it is `pandas.DataFrame` or `xarray.Dataset`; the caller will receive the
table in the indicated format.

The decorator will spin off a reference of the decorated function in the
`_RUNNABLE_STEPS` class attribute for `State`, facilitating the automatic
discovery and/or execution of this function via the
[`State.run`](activitysim.core.workflow.State.run) mechanisms.
The original function also remains available to import and use without
changes.

The decorated function may mutate the `state` argument by adding or removing
things from the state's context. Most existing workflow steps operate in this
manner. The return type annotation can be given as "None" to flag that
this mutation behavior is indeed baked in to the decorated function -- indeed,
by implication it must be as there is no other pathway to output a result,
although that is not otherwise checked.

Alternatively, the wrapped function can return a `Mapping[str, Any]` that
will be used to update the state's context. This happens automatically when
the step is called via the `State.run` accessor, or can (must) be handled
separately by the caller if the function is executed directly. (Future work
may migrate ActivitySim to favor or require this "pure" function behavior.)


## API

```{eval-rst}
.. currentmodule:: activitysim.core.workflow

.. autosummary::
    :toctree: _generated
    :template: autosummary/class_decorator.rst
    :recursive:

    step
```
