from __future__ import annotations

from pathlib import Path

from activitysim.core import workflow


def create_example(
    example_name: str,
    directory: Path | str = None,
    temp: bool = False,
) -> workflow.State:
    """
    Create an example model.

    Parameters
    ----------
    example_name : str
    directory : Path-like, optional
        Install the example into this directory.
    temp : bool, default False
        Install the example into a temporary directory tied to the returned
        State object. Cannot be set to True if `directory` is given.

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

    directory = Path(directory)

    # import inside function to prevent circular references.
    from activitysim.examples import get_example

    installed_to, subdirs = get_example(
        example_name, destination=directory, with_subdirs=True
    )
    state = workflow.State.make_default(installed_to, **subdirs)
    if temp:
        state.set("_TEMP_DIR_", temp_dir)
    return state
