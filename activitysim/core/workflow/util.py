import logging

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
