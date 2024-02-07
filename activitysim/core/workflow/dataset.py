from __future__ import annotations

import logging
from collections.abc import Iterable

import xarray as xr
from sharrow.dataset import construct

from .accessor import StateAccessor

logger = logging.getLogger(__name__)


class Datasets(StateAccessor):
    """
    This accessor provides easy access to state tables and datasets.

    Named datasets are available as attributes on this accessor, for
    example `State.dataset.persons`, which is the equivalent of calling
    `State.get_dataset("persons")`.

    Otherwise, there are no "normal" methods for this accessor, but the
    `__dir__` method is implemented to provide tab-completion capabilities
    inside compatible interactive interpreters (including Jupyter notebooks).
    """

    def __get__(self, instance, objtype=None) -> Datasets:
        # derived __get__ changes annotation, aids in type checking
        return super().__get__(instance, objtype)

    def __dir__(self) -> Iterable[str]:
        return (
            self._obj.existing_table_status.keys() | self._obj._LOADABLE_TABLES.keys()
        )

    def __getattr__(self, item) -> xr.Dataset:
        if item in self._obj.existing_table_status:
            return self._obj.get_dataset(item)
        elif item in self._obj._LOADABLE_TABLES:
            arg_value = self._obj._LOADABLE_TABLES[item](self._obj._context)
            return construct(arg_value)
        raise AttributeError(item)
