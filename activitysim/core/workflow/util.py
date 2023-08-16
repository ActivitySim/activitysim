from __future__ import annotations

import logging
import os

from pypyr.context import Context, KeyNotInContextError

logger = logging.getLogger(__name__)


def get_formatted_or_raw(self: Context, key: str):
    try:
        return self.get_formatted(key)
    except TypeError:
        return self.get(key)
    except Exception as err:
        raise ValueError(f"extracting {key} from context") from err


def get_formatted_or_default(self: Context, key: str, default):
    try:
        return self.get_formatted(key)
    except (KeyNotInContextError, KeyError):
        return default
    except TypeError:
        return self.get(key)
    except Exception as err:
        raise ValueError(f"extracting {key} from context") from err


def get_override_or_formatted_or_default(
    overrides: dict, self: Context, key: str, default
):
    if key in overrides:
        return overrides[key]
    else:
        return get_formatted_or_default(self, key, default)


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def write_notebook_heading(text: str, heading_level: int | None = None) -> None:
    """
    If running in a jupyter-like environment, display a heading.

    Parameters
    ----------
    text : str
        The heading to display
    heading_level : int, optional
        The heading level to use.  Should be an integer from 1 to 6.
        If omitted or zero, no heading is not displayed.
    """
    if heading_level and is_notebook():
        if heading_level < 0:
            raise ValueError("negative heading levels not allowed")
        if heading_level > 6:
            # heading levels greater than 6 are not allowed
            heading_level = 6
        import IPython.display

        IPython.display.display_markdown("#" * heading_level + f" {text}", raw=True)


def remove_empty_folders(path_abs):
    walk = list(os.walk(path_abs))
    for path, _, _ in walk[::-1]:
        if len(os.listdir(path)) == 0:
            os.remove(path)
