from pathlib import Path

from activitysim.core.workflow.state import Whale
from activitysim.examples import get_example


def create_example(example_name, directory: Path = None, temp: bool = False):
    """
    Create an example model.

    Parameters
    ----------
    example_name : str
    directory : Path-like, optional

    Returns
    -------
    Whale
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
    whale = Whale.make_default(get_example(example_name, destination=directory))
    if temp:
        whale.context["_TEMP_DIR_"] = temp_dir
    return whale
