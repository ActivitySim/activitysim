from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import pandas as pd
import xarray as xr
from sharrow.dataset import construct

from activitysim.core.datastore._base import CheckpointStore
from activitysim.core.exceptions import (
    CheckpointFileNotFoundError,
    CheckpointNameNotFoundError,
    ReadOnlyError,
    TableNameNotFound,
)

from ._base import (
    CHECKPOINT_NAME,
    CHECKPOINT_TABLE_NAME,
    LAST_CHECKPOINT,
    NON_TABLE_COLUMNS,
    TIMESTAMP,
)

logger = logging.getLogger(__name__)


class HdfStore(CheckpointStore):
    """Storage interface for HDF5-based table storage."""

    extension = ".h5"

    def __init__(self, filename: Path, mode="a", complib: str = None):
        super().__init__(mode=mode)
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        if filename.suffix != self.extension:
            filename = filename.with_suffix(self.extension)
        self._filename = filename
        # self._hdf5 = pd.HDFStore(str(filename), mode=mode)
        self.complib = complib
        self.last_checkpoint = {}
        self.checkpoints: list[dict] = []

    @property
    def filename(self) -> Path:
        return self._filename

    def _store_table_key(self, table_name, checkpoint_name):
        if checkpoint_name:
            key = f"{table_name}/{checkpoint_name}"
        else:
            key = f"/{table_name}"
        return key

    def _get_store_checkpoint_from_named_checkpoint(
        self, name: str, checkpoint_name: str = LAST_CHECKPOINT
    ):
        f"""
        Get the name of the checkpoint where a table is actually written.

        Checkpoint tables are not re-written if the content has not changed, so
        retrieving a particular table at a given checkpoint can involve back-tracking
        to find where the file was last actually written.

        Parameters
        ----------
        name : str
        checkpoint_name : str, default {LAST_CHECKPOINT!r}
            The name of the checkpoint to load.  If not given, {LAST_CHECKPOINT!r}
            is assumed, indicating that this function should load the last stored
            checkpoint value.

        Returns
        -------
        str
            The checkpoint to actually load.
        """
        cp_df = self.read_dataframe(
            CHECKPOINT_TABLE_NAME, checkpoint_name=None
        ).set_index(CHECKPOINT_NAME)
        if checkpoint_name == LAST_CHECKPOINT:
            checkpoint_name = cp_df.index[-1]
        try:
            return cp_df.loc[checkpoint_name, name]
        except KeyError:
            if checkpoint_name not in cp_df.index:
                raise CheckpointNameNotFoundError(checkpoint_name)
            elif name not in cp_df.columns:
                raise TableNameNotFound(name)
            else:
                raise

    def read_dataframe(
        self,
        name: str,
        checkpoint_name: str | None = LAST_CHECKPOINT,
    ) -> pd.DataFrame:
        """
        Read in a dataset from persistent storage.

        Parameters
        ----------
        name : str
        checkpoint_name : str, optional
            The checkpoint version name to use for this read operation.  If not
            provided, the last available checkpoint for this dataset is read.
        """
        key = self._store_table_key(name, checkpoint_name)
        with pd.HDFStore(str(self.filename), mode=self._mode) as store:
            return store.get(key)

    def write_data(
        self,
        name: str,
        checkpoint_name: str | None,
        data: xr.Dataset | pd.DataFrame,
        overwrite: bool = True,
    ):
        """
        Write out a particular dataset to persistent storage.

        Parameters
        ----------
        name : str
        checkpoint_name : str
            The checkpoint version name to use for this write operation.  For
            consistency in data management and organization, all write
            operations must have a checkpoint name.
        data : xarray.Dataset or pandas.DataFrame
        """
        key = self._store_table_key(name, checkpoint_name)
        if isinstance(data, xr.Dataset):
            if len(data.dims) == 1:
                df = data.single_dim.to_pandas()
            else:
                df = data.to_pandas()
        else:
            df = data
        with pd.HDFStore(str(self.filename), mode=self._mode) as store:
            if self.complib is None or len(df.columns) == 0:
                # tables with no columns can't be compressed successfully, so to
                # avoid them getting just lost and dropped they are instead written
                # in fixed format with no compression, which should be just fine
                # since they have no data anyhow.
                try:
                    store.put(key, df)
                except NotImplementedError:
                    store.put(key, df, "table")
            else:
                store.put(key, df, "table", complib=self.complib)
            store.flush()

    def make_checkpoint(self, checkpoint_name: str, overwrite: bool = True) -> None:
        if self.is_readonly:
            raise ReadOnlyError
        to_be_checkpointed = self.to_be_checkpointed(everything=True)

        try:
            checkpoint_df = self.get_dataframe(CHECKPOINT_TABLE_NAME)
        except KeyError:
            checkpoint_df = pd.DataFrame(index=[], columns=NON_TABLE_COLUMNS)

        if checkpoint_name in checkpoint_df.index:
            if overwrite:
                checkpoint_df = checkpoint_df.drop(index=checkpoint_name)
            else:
                raise KeyError(f"duplicate checkpoint {checkpoint_name!r}")

        this_timestamp = dt.datetime.now()

        logger.debug(
            "add_checkpoint %s timestamp %s" % (checkpoint_name, this_timestamp)
        )

        for table_name in to_be_checkpointed:
            data = self.get_dataset(table_name)
            logger.debug(f"add_checkpoint {checkpoint_name!r} table {table_name!r}")
            self.write_data(table_name, checkpoint_name, data)

            # remember which checkpoint it was last written
            self.last_checkpoint[table_name] = checkpoint_name
            self._update_dataset(table_name, data, last_checkpoint=checkpoint_name)

        self.last_checkpoint[CHECKPOINT_NAME] = checkpoint_name
        self.last_checkpoint[TIMESTAMP] = this_timestamp

        # append to the array of checkpoint history
        self.checkpoints.append(self.last_checkpoint.copy())

        # create a pandas dataframe of the checkpoint history, one row per checkpoint
        checkpoints = pd.DataFrame(self.checkpoints)

        # convert empty values to str so PyTables doesn't pickle object types
        for c in checkpoints.columns:
            checkpoints[c] = checkpoints[c].fillna("")

        # write it to the store, overwriting any previous version (no way to simply extend)
        self.write_data(CHECKPOINT_TABLE_NAME, None, checkpoints)

    def restore_checkpoint(self, checkpoint_name: str) -> None:
        logger.info("load_checkpoint %s" % (checkpoint_name))

        try:
            checkpoints = self.read_dataframe(CHECKPOINT_TABLE_NAME, None)
        except FileNotFoundError as err:
            raise CheckpointFileNotFoundError(err)

        if checkpoint_name == LAST_CHECKPOINT:
            checkpoint_name = checkpoints[CHECKPOINT_NAME].iloc[-1]
            logger.info("loading checkpoint '%s'" % checkpoint_name)

        try:
            # truncate rows after target checkpoint
            i = checkpoints[checkpoints[CHECKPOINT_NAME] == checkpoint_name].index[0]
            checkpoints = checkpoints.loc[:i]

            # if the store is not open in read-only mode,
            # write it to the store to ensure so any subsequent checkpoints are forgotten
            if not self.is_readonly:
                self.write_data(CHECKPOINT_TABLE_NAME, None, checkpoints)

        except IndexError:
            msg = "Couldn't find checkpoint '%s' in checkpoints" % (checkpoint_name,)
            print(checkpoints[CHECKPOINT_NAME])
            logger.error(msg)
            raise RuntimeError(msg)

        # convert pandas dataframe back to array of checkpoint dicts
        checkpoints = checkpoints.to_dict(orient="records")

        # drop tables with empty names
        for checkpoint in checkpoints:
            for key in list(checkpoint.keys()):
                if key not in NON_TABLE_COLUMNS and not checkpoint[key]:
                    del checkpoint[key]

        # patch _CHECKPOINTS array of dicts
        self.checkpoints = checkpoints

        # patch _CHECKPOINTS dict with latest checkpoint info
        self.last_checkpoint.clear()
        self.last_checkpoint.update(self.checkpoints[-1])

        logger.info(
            "load_checkpoint %s timestamp %s"
            % (checkpoint_name, self.last_checkpoint["timestamp"])
        )

        last_checkpoint = self.last_checkpoint

        tables = [
            name
            for name, checkpoint_name in last_checkpoint.items()
            if checkpoint_name and name not in NON_TABLE_COLUMNS
        ]

        for table_name in tables:
            # read dataframe from pipeline store
            df = self.read_dataframe(
                table_name,
                checkpoint_name=last_checkpoint[table_name],
            )
            logger.info("load_checkpoint table %s %s" % (table_name, df.shape))

            self._set_dataset(table_name, construct(df), last_checkpoint[table_name])
