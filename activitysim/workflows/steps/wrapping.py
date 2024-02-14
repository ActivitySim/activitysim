from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping
from inspect import getfullargspec

from pypyr.context import Context

from activitysim.workflows.steps import get_formatted_or_default
from activitysim.workflows.steps.error_handler import error_logging
from activitysim.workflows.steps.progression import reset_progress_step

logger = logging.getLogger(__name__)


class MockReport:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type or exc_val or exc_tb:
            logger.exception(repr(exc_val))

    def __lshift__(self, other):
        logger.info(str(other))


class workstep:
    """
    Decorator for functions that write report sections or update context.

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

    def __new__(cls, wrapped_func=None, *, returns_names=None, updates_context=False):
        """
        Initialize a work step wrapper.

        Parameters
        ----------
        wrapped_func : Callable
            The function being decorated.
        """
        if isinstance(wrapped_func, str | tuple | list):
            # the returns_names are provided instead of the wrapped func
            returns_names = wrapped_func
            wrapped_func = None
        self = super().__new__(cls)
        self._returns_names = returns_names
        self._updates_context = updates_context
        if wrapped_func is not None:
            return self(wrapped_func)
        else:
            return self

    def __call__(self, wrapped_func):
        returns_names = self._returns_names
        updates_context = self._updates_context
        (
            _args,
            _varargs,
            _varkw,
            _defaults,
            _kwonlyargs,
            _kwonlydefaults,
            _annotations,
        ) = getfullargspec(wrapped_func)

        if isinstance(returns_names, str):
            returns_names = (returns_names,)

        def run_step(context: Context = None) -> None:
            caption = get_formatted_or_default(context, "caption", None)
            progress_tag = get_formatted_or_default(context, "progress_tag", caption)
            if progress_tag is not None:
                reset_progress_step(description=progress_tag)

            return_type = _annotations.get("return", "<missing>")
            _updates_context = updates_context or return_type in {
                dict,
                Context,
                "dict",
                "Context",
            }
            if return_type not in {None, dict, Context, "None", "dict", "Context"}:
                if returns_names is None and not _updates_context:
                    context.assert_key_has_value(
                        key="report", caller=wrapped_func.__module__
                    )
            report = context.get("report", MockReport())
            with report:
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
                    context.assert_key_has_value(
                        key=arg, caller=wrapped_func.__module__
                    )
                    try:
                        args.append(context.get_formatted_or_raw(arg))
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
                            kwargs[karg] = context.get_formatted_or_raw(karg)
                        except Exception as err:
                            raise ValueError(f"extracting {karg} from context") from err
                if _varkw:
                    kwargs.update(context)
                    for arg in _required_args:
                        if arg in kwargs:
                            kwargs.pop(arg)
                outcome = error_logging(wrapped_func)(*args, **kwargs)
                if returns_names:
                    if len(returns_names) == 1:
                        context[returns_names[0]] = outcome
                    else:
                        for returns_name, out in zip(returns_names, outcome):
                            context[returns_name] = out
                elif updates_context:
                    if not isinstance(outcome, Mapping):
                        raise ValueError(
                            f"{wrapped_func.__name__} is marked as updates_context, "
                            f"it should return a mapping"
                        )
                    context.update(outcome)
                elif outcome is not None:
                    caption_level = get_formatted_or_default(
                        context, "caption_level", None
                    )
                    if caption is not None:
                        report << caption_maker(caption, level=caption_level)
                    report << outcome

        module = importlib.import_module(wrapped_func.__module__)
        if hasattr(module, "run_step"):
            raise ValueError(
                f"{wrapped_func.__module__}.run_step exists, there can be only one per module"
            )
        setattr(module, "run_step", run_step)

        return wrapped_func
