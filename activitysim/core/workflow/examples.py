from __future__ import annotations

from pathlib import Path

from activitysim.core import workflow


def create_example(
    example_name, directory: Path = None, temp: bool = False
) -> "workflow.State":
    """
    Create an example model.

    Parameters
    ----------
    example_name : str
    directory : Path-like, optional

    Returns
    -------
    State
    """
    if temp:
        if directory is not None:
            raise ValueError("cannot give `directory` and also `temp`")
        import tempfile

        temp_dir = tempfile.TemporaryDirectory()
        directory = temp_dir.name
    else:
        temp_dir = None
    if directory is None:
        directory = Path.cwd()

    # import inside function to prevent circular references.
    from activitysim.examples import get_example

    state = workflow.State.make_default(
        get_example(example_name, destination=directory)
    )
    if temp:
        state.set("_TEMP_DIR_", temp_dir)
    return state
