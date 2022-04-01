import os
from contextlib import contextmanager
from pathlib import Path

@contextmanager
def chdir(path: Path):
    """
    Sets the cwd within the context

    Args:
        path (Path): The path to the cwd

    Yields:
        None
    """

    cwd = Path().absolute()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(cwd)
