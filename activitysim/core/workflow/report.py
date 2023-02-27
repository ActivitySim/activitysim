from __future__ import annotations

import logging
import os
import sys
from collections.abc import Mapping, MutableMapping

import yaml

from activitysim.core.contrast import NominalTarget, compare_nominal
from activitysim.core.contrast.continuous import compare_histogram
from activitysim.core.workflow._state import BasicState
from activitysim.core.workflow.accessor import StateAccessor

logger = logging.getLogger(__name__)


class Reporting(StateAccessor):
    """
    Tools for reporting and visualization
    """

    def __get__(self, instance, objtype=None) -> "Reporting":
        # derived __get__ changes annotation, aids in type checking
        return super().__get__(instance, objtype)

    def nominal_distribution(
        self, tablename, nominal_col, *args, target=None, **kwargs
    ):
        states = {"model": self._obj}
        if target is not None:
            if not isinstance(target, (NominalTarget, BasicState)):
                target = NominalTarget(counts=target)
            states["target"] = target
        return compare_nominal(states, tablename, nominal_col, *args, **kwargs)

    def histogram(self, table_name, column_name, *args, target=None, **kwargs):
        states = {"model": self._obj}
        if target is not None:
            states["target"] = target
        return compare_histogram(states, table_name, column_name, *args, **kwargs)
