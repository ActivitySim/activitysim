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


# def _create_step(step_name, step_func):
#     # the module version of each step is for pypyr, and it always mutates
#     # context in-place instead of making updates to copies
#     from .steps import _create_module, _STEP_LIBRARY
#     _create_module(f"{__package__}.{step_name}", {"run_step": step_func})
#     _STEP_LIBRARY[step_name] = step_func
#
#
# def run_named_step(name, context):
#     from .steps import _STEP_LIBRARY
#     try:
#         step_func = _STEP_LIBRARY[name]
#     except KeyError:
#         logger.error(f"Unknown step {name}, the known steps are:")
#         for n in sorted(_STEP_LIBRARY.keys()):
#             logger.error(f" - {n}")
#         raise
#     step_func(context)
#     return context


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


def remove_empty_folders(path_abs):
    walk = list(os.walk(path_abs))
    for path, _, _ in walk[::-1]:
        if len(os.listdir(path)) == 0:
            os.remove(path)
