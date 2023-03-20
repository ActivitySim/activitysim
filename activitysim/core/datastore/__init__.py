from __future__ import annotations

from ._base import CheckpointStore

__all__ = [
    "CheckpointStore",
    "new_store",
]


def new_store(*args, storage_format="parquet", **kwargs):
    if storage_format.lower() == "parquet":
        from .parquet import ParquetStore

        return ParquetStore(*args, **kwargs)
    elif storage_format.lower() == "zarr":
        from .zarr import ZarrStore

        return ZarrStore(*args, **kwargs)
    elif storage_format.lower() in {"hdf", "hdf5"}:
        from .hdf import HdfStore

        return HdfStore(*args, **kwargs)
