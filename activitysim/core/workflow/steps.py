from __future__ import annotations

import abc
import importlib
import importlib.machinery
import importlib.util
import logging
import time
from collections import namedtuple
from collections.abc import Container
from inspect import get_annotations, getfullargspec
from typing import Callable, Collection, Mapping, NamedTuple

import numpy as np  # noqa: 401
import pandas as pd  # noqa: 401
import xarray as xr  # noqa: 401
from pypyr.context import Context
from pypyr.errors import KeyNotInContextError

from activitysim.core import workflow
from activitysim.core.exceptions import (
    DuplicateWorkflowNameError,
    DuplicateWorkflowTableError,
)
from activitysim.core.workflow.util import (
    get_formatted_or_default,
    get_formatted_or_raw,
    get_override_or_formatted_or_default,
    is_notebook,
)

logger = logging.getLogger(__name__)

_STEP_LIBRARY = {}

ExtendedArgSpec = namedtuple(
    "ExtendedArgSpec",
    "args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, "
    "annotations, ndefault, required_args",
)


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


def run_named_step(name, context, **kwargs):
    try:
        step_func = _STEP_LIBRARY[name]
    except KeyError:
        logger.error(f"Unknown step {name}, the known steps are:")
        for n in sorted(_STEP_LIBRARY.keys()):
            logger.error(f" - {n}")
        raise
    step_func(context, **kwargs)
    return context


class StepArgInit(abc.ABC):
    """
    Base class for things that initialize default workflow.step args from state.
    """

    @abc.abstractmethod
    def __call__(self, state: workflow.State, **other_overrides):
        raise NotImplementedError


class ModelSettingsFromYaml(StepArgInit):
    def __init__(self, model_settings_file_name):
        self.model_settings_file_name = model_settings_file_name

    def __call__(self, state: workflow.State, **other_overrides):
        return state.filesystem.read_model_settings(self.model_settings_file_name)


class step:
    """
    Decorator for ActivitySim model components and related functions.

    See the documentation on :ref:`workflow-steps` for more details.

    Parameters
    ----------
    wrapped_func : Callable
        The function being wrapped.
    step_name : str, optional
        The name of the step.  This is usually just inferred from the name of
        the function being wrapped, but it can be explicitly set to some other
        value if needed.
    cache : bool, default False
        If true, this function is only run if the named value is not
        already stored in the context.  Also, the return value should
        not be a mapping but instead just a single Python object that
        will be stored in the context with a key given by the step_name.
    kind : {"step", "table", "temp_table", "cached_object"}
        The kind of workflow function being wrapped.
    copy_tables : bool or Container[str], default True
        If this evaluates to true, access to tables as a DataFrame is
        always via a copy operation on any registered table instead of the
        original. If given as a container, only table names in the container
        are copied.
    overloading : bool, default False
        Allow this step definition to overload an existing wrapped workflow
        function. This permits a user to overload ActivitySim functions with
        bespoke alternatives.  To ensure that the reverse never occurs (i.e.
        the user creates a bespoke alternative implementation and then allows
        it to be overwritten by ActivitySim's default by importing things in
        the wrong order) steps defined and delivered within the ActivitySim
        package itself should never set this flag.

    Returns
    -------
    Callable
    """

    def __new__(
        cls,
        wrapped_func=None,
        *,
        step_name=None,
        cache=False,
        kind="step",
        copy_tables=True,
        overloading=False,
    ):
        if wrapped_func is not None and not isinstance(wrapped_func, Callable):
            raise TypeError("workflow step must decorate a callable")
        if step_name is None and wrapped_func is not None:
            step_name = wrapped_func.__name__
        self = super().__new__(cls)
        self._step_name = step_name
        self._cache = cache
        self._kind = kind
        self._copy_tables = copy_tables
        self._overloading = overloading
        if wrapped_func is not None:
            return self(wrapped_func)
        else:
            return self

    def __call__(self, wrapped_func):
        """
        Initialize a workflow.step wrapper.

        Parameters
        ----------
        wrapped_func : Callable
            The function being decorated.  It should return a dictionary
            of context updates.
        """
        from activitysim.core.workflow import State

        _validate_workflow_function(wrapped_func)
        if self._step_name is None:
            self._step_name = wrapped_func.__name__
        logger.debug(f"found workflow_{self._kind}: {self._step_name}")
        docstring = wrapped_func.__doc__

        # overloading of existing steps is only allowed when the user
        # sets overloading=True, which should never be done for steps
        # defined and delivered within the ActivitySim package itself
        def warn_overload():
            if self._overloading:
                logger.warning(
                    f"workflow.step {wrapped_func.__module__}.{self._step_name} "
                    f"overloading existing {self._step_name}"
                )
            else:
                raise DuplicateWorkflowNameError(self._step_name)

        # check for duplicate workflow function names
        if self._step_name in State._LOADABLE_OBJECTS:
            warn_overload()
        if self._step_name in State._LOADABLE_TABLES:
            warn_overload()
        if self._step_name in State._RUNNABLE_STEPS:
            warn_overload()

        (
            _args,
            _varargs,
            _varkw,
            _defaults,
            _kwonlyargs,
            _kwonlydefaults,
            _annotations,
        ) = getfullargspec(wrapped_func)

        # getfullargspec does not eval stringized annotations, so re-get those
        _annotations = get_annotations(wrapped_func, eval_str=True)

        if _defaults is None:
            _ndefault = 0
            _required_args = _args
        else:
            _ndefault = len(_defaults)
            _required_args = _args[:-_ndefault]

        self._fullargspec = ExtendedArgSpec(
            _args,
            _varargs,
            _varkw,
            _defaults,
            _kwonlyargs,
            _kwonlydefaults,
            _annotations,
            _ndefault,
            _required_args,
        )

        if not _required_args or _required_args[0] != "state":
            raise TypeError(
                f"the first argument of a workflow_{self._kind} must be the state"
            )

        def run_step(context: Context = None, **override_kwargs) -> None:
            if (
                self._cache
                and (context is not None)
                and (self._step_name in context)
                and len(override_kwargs) == 0
            ):
                return context.get_formatted(self._step_name)
            assert isinstance(context, Context)
            state = State(context)

            # initialize step-specific arguments if they are not provided in override_kwargs
            if _ndefault:
                for arg, default in zip(_args[-_ndefault:], _defaults):
                    if isinstance(default, StepArgInit):
                        override_kwargs[arg] = default(state, **override_kwargs)
                    else:
                        override_kwargs[arg] = default
            if _kwonlydefaults:
                for karg in _kwonlyargs:
                    karg_default = _kwonlydefaults.get(karg, None)
                    if isinstance(karg_default, StepArgInit):
                        override_kwargs[karg] = karg_default(state, **override_kwargs)
                    else:
                        override_kwargs[karg] = karg_default

            caption = get_override_or_formatted_or_default(
                override_kwargs, context, "caption", None
            )
            progress_tag = get_override_or_formatted_or_default(
                override_kwargs, context, "progress_tag", caption
            )
            # if progress_tag is not None:
            #     reset_progress_step(description=progress_tag)

            return_type = _annotations.get("return", "<missing>")

            caption_type = get_override_or_formatted_or_default(
                override_kwargs, context, "caption_type", "fig"
            )
            caption_maker = get_override_or_formatted_or_default(
                override_kwargs, context, caption_type, None
            )
            # parse and run function itself
            args = []
            for arg in _required_args[1:]:
                # first arg is always state
                if arg in override_kwargs:
                    arg_value = override_kwargs[arg]
                elif arg in context:
                    arg_value = context.get(arg)
                else:
                    if arg in state._LOADABLE_TABLES:
                        arg_value = state._LOADABLE_TABLES[arg](context)
                    elif arg in state._LOADABLE_OBJECTS:
                        arg_value = state._LOADABLE_OBJECTS[arg](context)
                    else:
                        context.assert_key_has_value(
                            key=arg, caller=wrapped_func.__module__
                        )
                        raise KeyError(arg)
                if (
                    self._copy_tables
                    and arg in state.existing_table_status
                    and arg not in override_kwargs
                ):
                    is_df = _annotations.get(arg) is pd.DataFrame
                    if is_df:
                        if isinstance(self._copy_tables, Container):
                            if arg in self._copy_tables:
                                arg_value = arg_value.copy()
                        else:
                            # copy_tables is truthy
                            arg_value = arg_value.copy()
                if _annotations.get(arg) is pd.DataFrame and isinstance(
                    arg_value, xr.Dataset
                ):
                    # convert to dataframe if asking for that
                    arg_value = arg_value.single_dim.to_pandas()
                if _annotations.get(arg) is xr.Dataset and isinstance(
                    arg_value, pd.DataFrame
                ):
                    # convert to dataset if asking for that
                    from sharrow.dataset import construct

                    arg_value = construct(arg_value)
                try:
                    args.append(arg_value)
                except Exception as err:
                    raise ValueError(f"extracting {arg} from context") from err
            if _ndefault:
                # step arguments with defaults are never taken from the context
                # they use the defaults always unless overridden manually
                for arg, default in zip(_args[-_ndefault:], _defaults):
                    if arg in override_kwargs:
                        args.append(override_kwargs[arg])
                    else:
                        args.append(default)
            kwargs = {}
            for karg in _kwonlyargs:
                if karg in _kwonlydefaults:
                    # step arguments with defaults are never taken from the context
                    # they use the defaults always unless overridden manually
                    kwargs[karg] = override_kwargs.get(karg, _kwonlydefaults[karg])
                else:
                    if karg in override_kwargs:
                        kwargs[karg] = override_kwargs[karg]
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
            try:
                state.this_step = self
                outcome = error_logging(wrapped_func)(state, *args, **kwargs)
            finally:
                del state.this_step
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

        run_step.__doc__ = docstring
        _create_step(self._step_name, run_step)

        def update_with_cache(state: State, *args, **kwargs):
            ignore_cache = kwargs.pop("_ignore_cache_", False)
            if self._step_name not in state._context or ignore_cache:
                state._context[self._step_name] = wrapped_func(state, *args, **kwargs)
            return state._context[self._step_name]

        update_with_cache.__doc__ = docstring
        update_with_cache.__name__ = self._step_name

        if self._kind == "cached_object":
            State._LOADABLE_OBJECTS[self._step_name] = run_step
            return update_with_cache
        elif self._kind == "table":
            State._LOADABLE_TABLES[self._step_name] = run_step
            return update_with_cache
        elif self._kind == "temp_table":
            State._TEMP_NAMES.add(self._step_name)
            State._LOADABLE_TABLES[self._step_name] = run_step
            for i in _args[1:]:
                if i not in State._PREDICATES:
                    State._PREDICATES[i] = {self._step_name}
                else:
                    State._PREDICATES[i].add(self._step_name)
            return update_with_cache
        elif self._kind == "step":
            State._RUNNABLE_STEPS[self._step_name] = run_step
            return wrapped_func
        else:
            raise ValueError(self._kind)


class cached_object(step):
    """
    Decorator for functions that deliver objects that should be cached.

    The function is called to initialize or otherwise generate the value of
    an object to be cached, but only if the matching name is not already stored
    in the state's context.

    :py:class:`@workflow.cached_object <activitysim.core.workflow.cached_object>` is equivalent to
    :py:class:`@workflow.step(cache=True, kind="cached_object") <activitysim.core.workflow.step>`.
    """

    def __new__(cls, wrapped_func=None, *, step_name=None):
        return super().__new__(
            cls, wrapped_func, step_name=step_name, cache=True, kind="cached_object"
        )


class table(step):
    """
    Decorator for functions that deliver a data table.

    The function is called to initialize or otherwise generate the content of
    a named data table, but only if the matching name is not already stored
    in the state's context.

    :py:class:`@workflow.table <activitysim.core.workflow.table>` is equivalent to
    :py:class:`@workflow.step(cache=True, kind="table") <activitysim.core.workflow.step>`.
    """

    def __new__(cls, wrapped_func=None, *, step_name=None):
        return super().__new__(
            cls, wrapped_func, step_name=step_name, cache=True, kind="table"
        )


class temp_table(step):
    """
    Decorator for functions that deliver a temporary data table.

    The function is called to initialize or otherwise generate the content of
    a named temp table, but only if the matching name is not already stored
    in the state's context.

    :py:class:`@workflow.temp_table <activitysim.core.workflow.temp_table>` is equivalent to
    :py:class:`@workflow.step(cache=True, kind="temp_table") <activitysim.core.workflow.step>`.
    """

    def __new__(cls, wrapped_func=None, *, step_name=None):
        return super().__new__(
            cls, wrapped_func, step_name=step_name, cache=True, kind="temp_table"
        )


def _validate_workflow_function(f):
    annot = get_annotations(f, eval_str=True)
    argspec = getfullargspec(f)
    if argspec.args[0] != "state":
        raise SyntaxError("workflow.func must have `state` as the first argument")
    if not issubclass(annot.get("state"), workflow.State):
        raise SyntaxError(
            "workflow.func must have `State` as the first argument annotation"
        )


def func(function):
    """
    Wrapper for a simple workflow function.
    """
    _validate_workflow_function(function)

    def wrapper(state, *args, **kwargs):
        if not isinstance(state, workflow.State):
            raise TypeError(
                "workflow functions must have a State as the first argument"
            )
        return function(state, *args, **kwargs)

    return wrapper
