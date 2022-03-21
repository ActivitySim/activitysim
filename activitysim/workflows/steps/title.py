from pypyr.errors import KeyNotInContextError
from pypyr.steps.py import run_step as _run_step
from .progression import progress, progress_overall, progress_step

def run_step(context):
    """Set the overall progress title.

    Takes two forms of input:
        py: exec contents as dynamically interpreted python statements, with
            contents of context available as vars.
        pycode: exec contents as dynamically interpreted python statements,
            with the context object itself available as a var.

    Args:
        context (pypyr.context.Context): Mandatory.
            Context is a dictionary or dictionary-like.
            Context must contain key 'py' or 'pycode'
    """
    try:
        label = context.get_formatted('label')
    except KeyNotInContextError:
        label = None
    if label is None:
        label = "ActivitySim Workflow"
    progress.update(progress_overall, description=label)
