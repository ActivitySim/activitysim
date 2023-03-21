from __future__ import annotations

import logging
import shutil

import pandas as pd
import xarray as xr
from sharrow.dataset import construct

from activitysim.core.datastore.parquet import ParquetStore
from activitysim.core.exceptions import ReadOnlyError

logger = logging.getLogger(__name__)


class ZarrStore(ParquetStore):
    def write_data(
        self,
        name: str,
        checkpoint_name: str,
        data: xr.Dataset | pd.DataFrame,
        overwrite: bool = True,
    ):
        if self.is_readonly:
            raise ReadOnlyError("store is read-only")
        if isinstance(data, pd.DataFrame):
            data = construct(data)
        # zarr is used for any ndim
        target = self._zarr_subdir(name, checkpoint_name)
        if overwrite and target.is_dir():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        data.to_zarr_with_attr(target)
