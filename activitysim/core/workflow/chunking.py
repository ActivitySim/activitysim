from __future__ import annotations

import logging.config
import threading

from activitysim.core.workflow.accessor import FromState, StateAccessor

logger = logging.getLogger(__name__)


def _init_historian():
    from activitysim.core.chunk import ChunkHistorian

    return ChunkHistorian()


class Chunking(StateAccessor):
    """
    This accessor provides chunking tools.
    """

    def __get__(self, instance, objtype=None) -> Chunking:
        # derived __get__ changes annotation, aids in type checking
        return super().__get__(instance, objtype)

    CHUNK_LEDGERS: list = FromState(default_init=True)
    CHUNK_SIZERS: list = FromState(default_init=True)
    ledger_lock: threading.Lock = FromState(default_init=True)
    HISTORIAN = FromState(default_init=_init_historian)
