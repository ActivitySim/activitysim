import importlib
import importlib.machinery
import importlib.util
import logging
from inspect import getfullargspec
from typing import Mapping

from pypyr.context import Context
from pypyr.errors import KeyNotInContextError

_STEP_LIBRARY = {}


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


def error_logging(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as err:
            logging.error(f"===== ERROR IN {func.__name__} =====")
            logging.exception(f"{err}")
            logging.error(f"===== / =====")
            raise

    return wrapper


def _new_module(mod_name):
    spec = importlib.machinery.ModuleSpec(mod_name, None)
    return importlib.util.module_from_spec(spec)


def _create_module(mod_name, content):
    mod = _new_module(mod_name)
    for k, v in content.items():
        setattr(mod, k, v)
    return mod


def _create_step(step_name, step_func):
    _create_module(f"{__package__}.{step_name}", {"run_step": step_func})
    _STEP_LIBRARY[step_name] = step_func


def run_named_step(name, context):
    context.update(_STEP_LIBRARY[name](context))
    return context


class workflow_step:
    """
    Decorator for functions that update a context variable.

    The decorator will generate a `run_step` function in the same module,
    wrapped with additional arguments and appropriately annotated for use
    with the pypyr workflow model.  The original function also remains
    available to import and use without changes.

    When called as a step inside a pypyr workflow, the following context
    variables are potentially accessed:

    report : xmle.Reporter
        The active report into which new figures or tables are added.
    caption : str
        A caption for the item being processed.  This is used both in
        writing out the output (if any) in the report and for logging
        step progression during a run.
    caption_type : str
        The caption type (typically, 'fig' for figures or 'tab'
        for tables).
    progress_tag : str
        Use this instead of `caption` to log step progression during a run.

    If the function returns values that should update the context, that
    can be done in one of three ways:

    - Set `updates_context` to True and return a `dict`, and use that
      dict to update the context directly.
    - Return a single object, and set `returns_names` to a string
      giving the name that object should take in the context.
    - Return a sequence of objects, and set `returns_names` to a
      matching sequence of names that those objects should take
      in the context.

    Otherwise, the return value is appended to the report.  To declare that
    there is no return value and no reporting should be done, you must
    explicitly annotate the function with a return value of `-> None`.

    Important: there can be only one `workstep` in
    each module.  If you need more than one, make another separate module.

    Parameters
    ----------
    wrapped_func : Callable
    returns_names : str or tuple[str], optional
    updates_context : bool, default False

    Returns
    -------
    wrapped_func : Callable
        The original wrapped function

    """

    def __new__(cls, wrapped_func=None, *, step_name=None):
        """
        Initialize a work step wrapper.

        Parameters
        ----------
        wrapped_func : Callable
            The function being decorated.
        """
        if isinstance(wrapped_func, str):
            # the step_name is provided instead of the wrapped func
            step_name = wrapped_func
            wrapped_func = None
        if step_name is None and wrapped_func is not None:
            step_name = wrapped_func.__name__
        self = super().__new__(cls)
        self._step_name = step_name
        if wrapped_func is not None:
            return self(wrapped_func)
        else:
            return self

    def __call__(self, wrapped_func):
        """
        Initialize a workflow_step wrapper.

        Parameters
        ----------
        wrapped_func : Callable
            The function being decorated.  It should return a dictionary
            of context updates.
        """
        (
            _args,
            _varargs,
            _varkw,
            _defaults,
            _kwonlyargs,
            _kwonlydefaults,
            _annotations,
        ) = getfullargspec(wrapped_func)

        def run_step(context: Context = None) -> None:
            caption = get_formatted_or_default(context, "caption", None)
            progress_tag = get_formatted_or_default(context, "progress_tag", caption)
            # if progress_tag is not None:
            #     reset_progress_step(description=progress_tag)

            return_type = _annotations.get("return", "<missing>")

            caption_type = get_formatted_or_default(context, "caption_type", "fig")
            caption_maker = get_formatted_or_default(context, caption_type, None)
            # parse and run function itself
            if _defaults is None:
                ndefault = 0
                _required_args = _args
            else:
                ndefault = len(_defaults)
                _required_args = _args[:-ndefault]
            args = []
            for arg in _required_args:
                context.assert_key_has_value(key=arg, caller=wrapped_func.__module__)
                try:
                    args.append(get_formatted_or_raw(context, arg))
                except Exception as err:
                    raise ValueError(f"extracting {arg} from context") from err
            if ndefault:
                for arg, default in zip(_args[-ndefault:], _defaults):
                    args.append(get_formatted_or_default(context, arg, default))
            kwargs = {}
            for karg in _kwonlyargs:
                if karg in _kwonlydefaults:
                    kwargs[karg] = get_formatted_or_default(
                        context, karg, _kwonlydefaults[karg]
                    )
                else:
                    context.assert_key_has_value(
                        key=karg, caller=wrapped_func.__module__
                    )
                    try:
                        kwargs[karg] = get_formatted_or_raw(context, karg)
                    except Exception as err:
                        raise ValueError(f"extracting {karg} from context") from err
            if _varkw:
                kwargs.update(context)
                for arg in _required_args:
                    if arg in kwargs:
                        kwargs.pop(arg)
            outcome = error_logging(wrapped_func)(*args, **kwargs)
            if not isinstance(outcome, Mapping):
                raise ValueError(
                    f"{wrapped_func.__name__} is marked as updates_context, "
                    f"it should return a mapping"
                )
            context.update(outcome)

        # module = importlib.import_module(wrapped_func.__module__)
        # if hasattr(module, "run_step"):
        #     raise ValueError(
        #         f"{wrapped_func.__module__}.run_step exists, there can be only one per module"
        #     )
        # setattr(module, "run_step", run_step)
        _create_step(self._step_name, run_step)

        return wrapped_func
