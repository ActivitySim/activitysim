from __future__ import annotations

import logging

from activitysim.core.workflow.accessor import StateAccessor

logger = logging.getLogger(__name__)


class Reporting(StateAccessor):
    """
    Tools for reporting and visualization
    """

    def __get__(self, instance, objtype=None) -> "Reporting":
        # derived __get__ changes annotation, aids in type checking
        return super().__get__(instance, objtype)
