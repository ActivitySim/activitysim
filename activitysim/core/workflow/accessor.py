from __future__ import annotations

import inspect
import textwrap
import warnings

from activitysim.core import workflow
from activitysim.core.exceptions import StateAccessError

NO_DEFAULT = "< no default >"


class StateAccessor:
    """
    Boilerplate code for accessors.

    Accessors consolidate groups of related functions in a common interface,
    without requiring the main State class to become bloated by including all
    relevant functionality.  They also allow setting and storing attributes
    without worrying about conflicting with similarly named attributes of
    other accessors.
    """

    def __set_name__(self, owner, name):
        self._name = name

    def __init__(self, state: "workflow.State" = None):
        self._obj = state

    def __get__(self, instance, objtype=None):
        if instance is None:
            return self
        cached_accessor = getattr(instance, f"_cached_accessor_{self._name}", None)
        if cached_accessor is not None:
            return cached_accessor
        from .state import State

        assert isinstance(instance, State)
        accessor_obj = self.__class__(instance)
        object.__setattr__(instance, self._name, accessor_obj)
        return accessor_obj

    def __set__(self, instance, value):
        if isinstance(value, self.__class__):
            setattr(instance, f"_cached_accessor_{self._name}", value)
        else:
            raise ValueError(f"cannot directly set accessor {self._name}")

    def __delete__(self, instance):
        raise ValueError(f"cannot delete accessor {self._name}")


class FromState:
    def __init__(
        self,
        member_type=None,
        default_init=False,
        default_value=NO_DEFAULT,
        doc: str | None = None,
    ):
        """
        Creates a property to access an element from the current context.

        Parameters
        ----------
        member_type : type
            The data type for this attribute.  Basic type validation may be
            applied when setting this value, but validation could be disabled
            when optimizing models for production and this type checking should
            not be relied on as a runtime feature. If not given, member_type is
            read from the accessor's type annotation (if applicable).
        default_init : bool or callable
            When set to true, if this context value is accessed and it has not
            already been set, it is automatically initialized with the default
            value (i.e. via a no-argument constructor) for the given type.
        default_value : Any, optional
            When set to some value, if this context value is accessed and it has
            not already been set, it is automatically initialized with this
            default value.
        doc : str, optional
            Attribute documentation.
        """
        self.member_type = member_type
        self._default_init = default_init
        self._default_value = default_value
        if self._default_init and self._default_value != NO_DEFAULT:
            raise ValueError("cannot use both default_init and default_value")
        if doc:
            self.__doc__ = textwrap.dedent(doc).strip()

    def __set_name__(self, owner, name):
        self.name = f"{owner.__name__.lower()}_{name}"
        # set member type based on annotation
        if self.member_type is None:
            annot = inspect.get_annotations(owner, eval_str=True)
            if name in annot:
                self.member_type = annot[name]

    def __get__(self, instance: StateAccessor, objtype=None):
        try:
            return instance._obj._context[self.name]
        except (KeyError, AttributeError):
            if instance is None or instance._obj is None:
                # a freestanding accessor not bound to a parent State is not
                # typical but does happen when Sphinx generates documentation
                return self
            if self._default_init:
                if callable(self._default_init):
                    instance._obj._context[self.name] = self._default_init()
                else:
                    instance._obj._context[self.name] = self.member_type()
                return instance._obj._context[self.name]
            elif self._default_value != NO_DEFAULT:
                instance._obj._context[self.name] = self._default_value
                return instance._obj._context[self.name]
            raise StateAccessError(
                f"{self.name} not initialized for this state"
            ) from None

    def __set__(self, instance: StateAccessor, value):
        if not self.__validate_type(value):
            raise TypeError(f"{self.name} must be {self.member_type} not {type(value)}")
        instance._obj._context[self.name] = value

    def __delete__(self, instance):
        self.__set__(instance, None)

    def __validate_type(self, value):
        # type validation is only done at the top level for now.
        try:
            type_ok = isinstance(value, self.member_type)
        except (TypeError, AttributeError):
            from typing import get_args, get_origin

            type_ok = isinstance(value, get_origin(self.member_type))
        return type_ok
