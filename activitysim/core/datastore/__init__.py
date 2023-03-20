from __future__ import annotations

from ._base import (
    CHECKPOINT_NAME,
    CHECKPOINT_TABLE_NAME,
    NON_TABLE_COLUMNS,
    CheckpointStore,
)

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


def copy_hdf(source_filename, dest_filename, mode="a", storage_format="parquet"):
    from .hdf import HdfStore

    hdf_store = HdfStore(source_filename, "r")
    output_store = new_store(dest_filename, mode=mode, storage_format=storage_format)
    checkpoint_df = hdf_store.read_dataframe(CHECKPOINT_TABLE_NAME, None).set_index(
        CHECKPOINT_NAME
    )

    for checkpoint_name in checkpoint_df.index:
        hdf_store.restore_checkpoint(checkpoint_name)

        for table_name in checkpoint_df.columns:
            if table_name in NON_TABLE_COLUMNS:
                continue
            checkpoint_name_ = checkpoint_df.loc[checkpoint_name, table_name]
            if checkpoint_name_ == checkpoint_name:
                data = hdf_store.get_dataset(table_name)
                output_store.set_data(table_name, data)

        output_store.make_checkpoint(checkpoint_name)

    return output_store
