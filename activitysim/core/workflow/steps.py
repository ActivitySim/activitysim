import importlib
import importlib.machinery
import importlib.util
import logging
import time
from inspect import get_annotations, getfullargspec
from typing import Callable, Mapping, NamedTuple

from pypyr.context import Context
from pypyr.errors import KeyNotInContextError

from activitysim.core.exceptions import (
    DuplicateWorkflowNameError,
    DuplicateWorkflowTableError,
)
from activitysim.core.workflow.util import (
    get_formatted_or_default,
    get_formatted_or_raw,
)

logger = logging.getLogger(__name__)

_STEP_LIBRARY = {}


class TableInfo(NamedTuple):
    factory: Callable
    predicates: tuple[str]


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
    # the module version of each step is for pypyr, and it always mutates
    # context in-place instead of making updates to copies
    _create_module(f"{__package__}.{step_name}", {"run_step": step_func})
    _STEP_LIBRARY[step_name] = step_func


def run_named_step(name, context):
    try:
        step_func = _STEP_LIBRARY[name]
    except KeyError:
        logger.error(f"Unknown step {name}, the known steps are:")
        for n in sorted(_STEP_LIBRARY.keys()):
            logger.error(f" - {n}")
        raise
    step_func(context)
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

    def __new__(
        cls,
        wrapped_func=None,
        *,
        step_name=None,
        cache=False,
        inplace=False,
        kind="step",
    ):
        """
        Initialize a work step wrapper.

        Parameters
        ----------
        wrapped_func : Callable
            The function being decorated.
        step_name : str
            Use this name for the function being decorated, if not given
            the existing name is used.
        cache : bool, default False
            If true, this function is only run if the named value is not
            already stored in the context.  Also, the return value should
            not be a mapping but instead just a single Python object that
            will be stored in the context with a key given by the step_name.
        """
        if wrapped_func is not None and not isinstance(wrapped_func, Callable):
            raise TypeError("workflow step must decorate a callable")
        if step_name is None and wrapped_func is not None:
            step_name = wrapped_func.__name__
        self = super().__new__(cls)
        self._step_name = step_name
        self._cache = cache
        self._inplace = inplace
        self._kind = kind
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
        from activitysim.core.workflow import Whale

        _validate_workflow_function(wrapped_func)
        if self._step_name is None:
            self._step_name = wrapped_func.__name__
        logger.debug(f"found workflow_{self._kind}: {self._step_name}")

        # check for duplicate workflow function names
        if self._step_name in Whale._LOADABLE_OBJECTS:
            raise DuplicateWorkflowNameError(self._step_name)
        if self._step_name in Whale._LOADABLE_TABLES:
            raise DuplicateWorkflowNameError(self._step_name)
        if self._step_name in Whale._RUNNABLE_STEPS:
            raise DuplicateWorkflowNameError(self._step_name)

        (
            _args,
            _varargs,
            _varkw,
            _defaults,
            _kwonlyargs,
            _kwonlydefaults,
            _annotations,
        ) = getfullargspec(wrapped_func)
        if _defaults is None:
            _ndefault = 0
            _required_args = _args
        else:
            _ndefault = len(_defaults)
            _required_args = _args[:-_ndefault]

        if not _required_args or _required_args[0] != "whale":
            raise TypeError(
                f"the first argument of a workflow_{self._kind} must be the whale"
            )

        def run_step(context: Context = None) -> None:
            if self._cache and (context is not None) and (self._step_name in context):
                return context.get_formatted(self._step_name)
            assert isinstance(context, Context)
            whale = Whale(context)
            caption = get_formatted_or_default(context, "caption", None)
            progress_tag = get_formatted_or_default(context, "progress_tag", caption)
            # if progress_tag is not None:
            #     reset_progress_step(description=progress_tag)

            return_type = _annotations.get("return", "<missing>")

            caption_type = get_formatted_or_default(context, "caption_type", "fig")
            caption_maker = get_formatted_or_default(context, caption_type, None)
            # parse and run function itself
            args = []
            for arg in _required_args:
                if arg == "whale":
                    args.append(whale)
                else:
                    try:
                        context.assert_key_has_value(
                            key=arg, caller=wrapped_func.__module__
                        )
                    except KeyNotInContextError:
                        # The desired key does not yet exist.  We will attempt
                        # to create it using the whale.
                        if arg in whale._LOADABLE_TABLES:
                            arg_value = whale._LOADABLE_TABLES[arg](context)
                        elif arg in whale._LOADABLE_OBJECTS:
                            arg_value = whale._LOADABLE_OBJECTS[arg](context)
                        else:
                            raise
                    else:
                        arg_value = get_formatted_or_raw(context, arg)
                    try:
                        args.append(arg_value)
                    except Exception as err:
                        raise ValueError(f"extracting {arg} from context") from err
            if _ndefault:
                for arg, default in zip(_args[-_ndefault:], _defaults):
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
            if self._kind == "table":
                context[self._step_name] = outcome
                if "_salient_tables" not in context:
                    context["_salient_tables"] = {}
                context["_salient_tables"][self._step_name] = time.time()
                return outcome
            elif self._kind == "temp_table":
                context[self._step_name] = outcome
                return outcome
            elif self._kind == "cached_object":
                context[self._step_name] = outcome
                return outcome
            elif self._kind == "step":
                if outcome is not None:
                    if not isinstance(outcome, Mapping):
                        raise ValueError(
                            f"workflow step {wrapped_func.__name__} should return a mapping or None"
                        )
                    context.update(outcome)

        _create_step(self._step_name, run_step)

        def update_with_cache(whale, *args, **kwargs):
            ignore_cache = kwargs.pop("_ignore_cache_", False)
            if self._step_name not in whale.context or ignore_cache:
                whale.context[self._step_name] = wrapped_func(whale, *args, **kwargs)
            return whale.context[self._step_name]

        if self._kind == "cached_object":
            Whale._LOADABLE_OBJECTS[self._step_name] = run_step
            return update_with_cache
        elif self._kind == "table":
            Whale._LOADABLE_TABLES[self._step_name] = run_step
            return update_with_cache
        elif self._kind == "temp_table":
            Whale._TEMP_NAMES.add(self._step_name)
            Whale._LOADABLE_TABLES[self._step_name] = run_step
            for i in _args[1:]:
                if i not in Whale._PREDICATES:
                    Whale._PREDICATES[i] = {self._step_name}
                else:
                    Whale._PREDICATES[i].add(self._step_name)
            return update_with_cache
        elif self._kind == "step":
            Whale._RUNNABLE_STEPS[self._step_name] = run_step
            return wrapped_func
        else:
            raise ValueError(self._kind)


class workflow_cached_object(workflow_step):
    def __new__(cls, wrapped_func=None, *, step_name=None):
        return super().__new__(
            cls, wrapped_func, step_name=step_name, cache=True, kind="cached_object"
        )


class workflow_table(workflow_step):
    def __new__(cls, wrapped_func=None, *, step_name=None):
        return super().__new__(
            cls, wrapped_func, step_name=step_name, cache=True, kind="table"
        )


class workflow_temp_table(workflow_step):
    def __new__(cls, wrapped_func=None, *, step_name=None):
        return super().__new__(
            cls, wrapped_func, step_name=step_name, cache=True, kind="temp_table"
        )


def _validate_workflow_function(f):
    from activitysim.core.workflow import Whale

    argspec = getfullargspec(f)
    if argspec.args[0] != "whale":
        raise SyntaxError("workflow.func must have `whale` as the first argument")
    if argspec.annotations.get("whale") is not Whale:
        raise SyntaxError(
            "workflow.func must have `Whale` as the first argument annotation"
        )


def func(function):
    """
    Wrapper for a simple workflow function.
    """
    from activitysim.core.workflow import Whale

    _validate_workflow_function(function)

    def wrapper(whale, *args, **kwargs):
        if not isinstance(whale, Whale):
            raise TypeError(
                "workflow functions must have a Whale as the first argument"
            )
        return function(whale, *args, **kwargs)

    return wrapper


# def workflow_table(func):
#     """
#     Decorator for functions that initialize tables.
#
#     The function being decorated should have a single argument: `whale`.
#
#     Parameters
#     ----------
#     func
#
#     Returns
#     -------
#     func
#     """
#     from ..pipeline import Whale
#     name = func.__name__
#     logger.debug(f"found loadable table {name}")
#     if name in Whale._LOADABLE_TABLES:
#         raise DuplicateWorkflowTableError(name)
#     Whale._LOADABLE_TABLES[name] = func
#     return func
