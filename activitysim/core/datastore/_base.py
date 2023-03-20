from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Collection
from pathlib import Path

import pandas as pd
import xarray as xr
from sharrow import DataTree, Relationship
from sharrow.dataset import construct

from activitysim.core.exceptions import ReadOnlyError

logger = logging.getLogger(__name__)

# name of the checkpoint dict keys
# (which are also columns in the checkpoints dataframe stored in hte pipeline store)
TIMESTAMP = "timestamp"
CHECKPOINT_NAME = "checkpoint_name"
NON_TABLE_COLUMNS = [CHECKPOINT_NAME, TIMESTAMP]

# name used for storing the checkpoints dataframe to the pipeline store
CHECKPOINT_TABLE_NAME = "checkpoints"

LAST_CHECKPOINT = "_"

# name of the first step/checkpoint created when the pipeline is started
INITIAL_CHECKPOINT_NAME = "init"
FINAL_CHECKPOINT_NAME = "final"


def timestamp():
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


class CheckpointStore:
    def __init__(self, mode="a"):
        self._mode = mode
        self._tree = DataTree(root_node_name=False)
        self._keep_digitized = False

    def write_data(
        self,
        name: str,
        checkpoint_name: str,
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
        data : xarray.Dataset
        overwrite : bool, default True
            It True, any existing persisted data at this location is deleted
            before new data is written.
        """

    def read_data(
        self,
        name: str,
        checkpoint_name: str = LAST_CHECKPOINT,
    ):
        """
        Read in a dataset from persistent storage.

        Parameters
        ----------
        name : str
        checkpoint_name : str, optional
            The checkpoint version name to use for this read operation.  If not
            provided, the last available checkpoint for this dataset is read.
        """
        raise NotImplementedError(f"cannot read data with {type(self)}")

    def get_dataframe(
        self,
        name: str,
        variables: Collection[str] = None,
    ) -> pd.DataFrame:
        """
        Retrieve some or all of a named dataset as a pandas DataFrame.

        For maximum performance, the dataset should have only a single dimension.

        Parameters
        ----------
        name : str
        variables : Collection[str], optional
            Get only these variables of the dataset.

        Returns
        -------
        pd.DataFrame
        """
        dataset = self.get_dataset(name, variables)
        if len(dataset.dims) == 1:
            return dataset.single_dim.to_pandas()
        else:
            return dataset.to_pandas()

    def get_dataset(
        self,
        name: str,
        variables: Collection[str] = None,
    ) -> xr.Dataset:
        """
        Retrieve some or all of a named dataset.

        Parameters
        ----------
        name : str
        variables : Collection[str], optional
            Get only these variables of the dataset.
        """
        if variables is None:
            return self._tree.get_subspace(name)
        else:
            return xr.Dataset({c: self._tree[f"{name}.{c}"] for c in variables})

    def _set_dataset(
        self,
        name: str,
        data: xr.Dataset,
        last_checkpoint: str = None,
    ) -> None:
        data_vars = {}
        coords = {}
        for k, v in data.coords.items():
            coords[k] = v.assign_attrs(last_checkpoint=last_checkpoint)
        for k, v in data.items():
            if k in coords:
                continue
            data_vars[k] = v.assign_attrs(last_checkpoint=last_checkpoint)
        data = xr.Dataset(data_vars=data_vars, coords=coords, attrs=data.attrs)
        self._tree.add_dataset(name, data)

    def set_data(
        self,
        name: str,
        data: xr.Dataset | pd.DataFrame,
    ) -> None:
        """
        Set the content of a named dataset in memory.

        This completely overwrites any existing data with the same name.
        It does not write anything to persistent storage, it only changes
        the current state of the named dataset in memory.  To persist the
        changes to disk, use `make_checkpoint`.

        Parameters
        ----------
        name : str
        data : Dataset or DataFrame
        """
        assert isinstance(name, str)
        if isinstance(data, xr.Dataset):
            self._set_dataset(name, data)
        elif isinstance(data, pd.DataFrame):
            self._set_dataset(name, construct(data))
        else:
            raise TypeError(f"cannot set_data with {type(data)}")

    def __getitem__(self, item):
        return self.get_dataset(item)

    def __setitem__(self, key, value):
        return self.set_data(key, value)

    def _update_dataset(
        self,
        name: str,
        data: xr.Dataset,
        last_checkpoint=None,
    ) -> xr.Dataset:
        if not isinstance(data, xr.Dataset):
            raise TypeError(type(data))
        partial_update = self._tree.get_subspace(name, default_empty=True)
        for k, v in data.items():
            if k in data.coords:
                continue
            assert v.name == k
            partial_update = self._update_dataarray(
                name, v, last_checkpoint, partial_update=partial_update
            )
        for k, v in data.coords.items():
            assert v.name == k
            partial_update = self._update_dataarray(
                name, v, last_checkpoint, as_coord=True, partial_update=partial_update
            )
        return partial_update

    def _update_dataarray(
        self,
        name: str,
        data: xr.DataArray,
        last_checkpoint=None,
        as_coord=False,
        partial_update=None,
    ) -> xr.Dataset:
        if partial_update is None:
            base_data = self._tree.get_subspace(name, default_empty=True)
        else:
            base_data = partial_update
        if isinstance(data, xr.DataArray):
            if as_coord:
                updated_dataset = base_data.assign_coords(
                    {data.name: data.assign_attrs(last_checkpoint=last_checkpoint)}
                )
            else:
                updated_dataset = base_data.assign(
                    {data.name: data.assign_attrs(last_checkpoint=last_checkpoint)}
                )
            self._tree = self._tree.replace_datasets(
                {name: updated_dataset}, redigitize=self._keep_digitized
            )
            return updated_dataset
        else:
            raise TypeError(type(data))

    def add_data(
        self,
        name: str,
        data: xr.Dataset | pd.DataFrame | xr.DataArray | pd.Series,
    ) -> None:
        """
        Augment the content of a named dataset in memory.

        This method adds to existing dataset with the same name, and it does not
        necessarily overwrite existing variables unless they are provided as
        input to the `data` argument.  It does not write anything to persistent
        storage, it only changes the current state of the named dataset in memory.
        To persist the changes to disk, use `make_checkpoint`.

        Parameters
        ----------
        name : str
        data : Dataset or DataFrame or DataArray or Series
        """
        if self.is_readonly:
            raise ReadOnlyError()
        # promote from pandas to xarray if needed
        if isinstance(data, pd.DataFrame):
            data = construct(data)
        elif isinstance(data, pd.Series):
            data = xr.DataArray.from_series(data)
        # make updates
        if isinstance(data, xr.Dataset):
            self._update_dataset(name, data, last_checkpoint=None)
        elif isinstance(data, xr.DataArray):
            self._update_dataarray(name, data, last_checkpoint=None)
        else:
            raise TypeError(type(data))

    def make_checkpoint(self, checkpoint_name: str, overwrite: bool = True) -> None:
        """
        Write all modified datasets to disk.

        Only new data (since the last time a checkpoint was made) is actually
        written out.

        Parameters
        ----------
        checkpoint_name : str
        overwrite : bool, default True
            If true, data from an existing checkpoint with the same name is
            overwritten.
        """
        raise NotImplementedError(f"cannot make checkpoint with {type(self)}")

    def restore_checkpoint(self, checkpoint_name: str) -> None:
        """
        Restore all tables to their state as of the named checkpoint.

        Parameters
        ----------
        checkpoint_name : str
        """
        raise NotImplementedError(f"cannot restore checkpoint with {type(self)}")

    @property
    def is_readonly(self) -> bool:
        """This store is read-only."""
        return self._mode == "r"

    @property
    def filename(self) -> Path | None:
        """Location of the persistent backing for this store."""
        # this base does not implement persistent storage, so it has no filename
        return None

    def list_checkpoint_names(self) -> list[str]:
        """Get a list of all checkpoint names in this store."""
        return []

    def add_relationship(self, relationship: str | Relationship):
        self._tree.add_relationship(relationship)

    def digitize_relationships(self, redigitize=True):
        """
        Convert all label-based relationships into position-based.

        Parameters
        ----------
        redigitize : bool, default True
            Re-compute position-based relationships from labels, even
            if the relationship had previously been digitized.
        """
        self._keep_digitized = True
        self._tree.digitize_relationships(inplace=True, redigitize=redigitize)

    @property
    def relationships_are_digitized(self) -> bool:
        """bool : Whether all relationships are digital (by position)."""
        return self._tree.relationships_are_digitized

    def to_be_checkpointed(self, everything=False) -> dict[str, xr.Dataset]:
        """
        The data that has been modified and needs to be checkpointed.

        Parameters
        ----------
        everything : bool, default False
            Whether to include the complete content of any table that has any
            modifications (legacy ActivitySim format).

        Returns
        -------
        dict[str, xr.Dataset]
        """
        result = {}
        for table_name, table_data in self._tree.subspaces_iter():
            # any data elements that were created without a
            # last_checkpoint attr get one now
            for _k, v in table_data.variables.items():
                if "last_checkpoint" not in v.attrs:
                    v.attrs["last_checkpoint"] = None
            # collect everything not checkpointed
            uncheckpointed = table_data.filter_by_attrs(last_checkpoint=None)
            if uncheckpointed:
                result[table_name] = table_data if everything else uncheckpointed
        return result

    # @classmethod
    # def from_hdf(cls, source_filename, dest_filename, mode="a"):
    #     hdf_store = HdfStore(source_filename, "r")
    #     output_store = cls(dest_filename, mode)
    #     checkpoint_df = hdf_store.get_dataframe(CHECKPOINT_TABLE_NAME)
    #     output_store.put(CHECKPOINT_TABLE_NAME, checkpoint_df)
    #     for table_name in checkpoint_df.columns:
    #         if table_name in NON_TABLE_COLUMNS:
    #             continue
    #         checkpoints_written = set()
    #         for checkpoint_name in checkpoint_df[table_name]:
    #             if checkpoint_name:
    #                 df = hdf_store.get_dataframe(table_name, checkpoint_name)
    #                 if checkpoint_name and checkpoint_name not in checkpoints_written:
    #                     output_store.put(
    #                         table_name, df, checkpoint_name=checkpoint_name
    #                     )
    #                     checkpoints_written.add(checkpoint_name)
    #     return output_store
    #
    # def _get_store_checkpoint_from_named_checkpoint(
    #     self, table_name: str, checkpoint_name: str = LAST_CHECKPOINT
    # ):
    #     f"""
    #     Get the name of the checkpoint where a table is actually written.
    #
    #     Checkpoint tables are not re-written if the content has not changed, so
    #     retrieving a particular table at a given checkpoint can involve back-tracking
    #     to find where the file was last actually written.
    #
    #     Parameters
    #     ----------
    #     table_name : str
    #     checkpoint_name : str, default {LAST_CHECKPOINT!r}
    #         The name of the checkpoint to load.  If not given, {LAST_CHECKPOINT!r}
    #         is assumed, indicating that this function should load the last stored
    #         checkpoint value.
    #
    #     Returns
    #     -------
    #     str
    #         The checkpoint to actually load.
    #     """
    #     cp_df = self.get_dataframe(CHECKPOINT_TABLE_NAME).set_index(CHECKPOINT_NAME)
    #     if checkpoint_name == LAST_CHECKPOINT:
    #         checkpoint_name = cp_df.index[-1]
    #     try:
    #         return cp_df.loc[checkpoint_name, table_name]
    #     except KeyError:
    #         if checkpoint_name not in cp_df.index:
    #             raise CheckpointNameNotFoundError(checkpoint_name)
    #         elif table_name not in cp_df.columns:
    #             raise TableNameNotFound(table_name)
    #         else:
    #             raise
