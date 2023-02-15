import datetime as dt
import logging
import os
import warnings
from builtins import map, next
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
import pyarrow as pa
import xarray as xr
from pypyr.context import Context, KeyNotInContextError

from activitysim.core.configuration import FileSystem, NetworkSettings, Settings
from activitysim.core.exceptions import WhaleAccessError
from activitysim.core.workflow.steps import run_named_step

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


class Checkpoints:
    def __init__(self):
        self.last_checkpoint = {}
        self.checkpoints = []
        self._checkpoint_store = None
        self.is_open = False

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, objtype=None):
        from .state import Whale

        assert isinstance(instance, Whale)
        self.obj = instance
        return self

    def __set__(self, instance, value):
        raise ValueError(f"cannot set {self.name}")

    def __delete__(self, instance):
        raise ValueError(f"cannot delete {self.name}")

    @property
    def store(self) -> Union[pd.HDFStore, Path]:
        if self._checkpoint_store is None:
            self.open_store()
        return self._checkpoint_store

    @property
    def default_pipeline_file_path(self):
        prefix = self.obj.get("pipeline_file_prefix", None)
        if prefix is None:
            return self.obj.filesystem.get_pipeline_filepath()
        else:
            pipeline_file_name = str(self.obj.filesystem.pipeline_file_name)
            pipeline_file_name = f"{prefix}-{pipeline_file_name}"
            return self.obj.filesystem.get_output_dir().joinpath(pipeline_file_name)

    def open_store(
        self, pipeline_file_name: Optional[Path] = None, overwrite=False, mode="a"
    ):
        """
        Open the checkpoint store.

        If the pipeline_file_name setting ends in ".h5", then the pandas
        HDFStore file format is used, otherwise pipeline files are stored
        as parquet files organized in regular file system directories.

        Parameters
        ----------
        pipeline_file_name : Path-like, optional
            An explicit pipeline file path.  If not given, the default pipeline
            file path is opened.
        overwrite : bool, default False
            delete file before opening (unless resuming)
        mode : {'a', 'w', 'r', 'r+'}, default 'a'
            ``'r'``
                Read-only; no data can be modified.
            ``'w'``
                Write; a new file is created (an existing file with the same
                name would be deleted).
            ``'a'``
                Append; an existing file is opened for reading and writing,
                and if the file does not exist it is created.
            ``'r+'``
                It is similar to ``'a'``, but the file must already exist.
        """

        if self._checkpoint_store is not None:
            raise RuntimeError("Pipeline store is already open!")

        if pipeline_file_name is None:
            pipeline_file_path = self.default_pipeline_file_path
        else:
            pipeline_file_path = Path(pipeline_file_name)

        if pipeline_file_path.suffix == ".h5":
            if overwrite:
                try:
                    if os.path.isfile(pipeline_file_path):
                        logger.debug("removing pipeline store: %s" % pipeline_file_path)
                        os.unlink(pipeline_file_path)
                except Exception as e:
                    print(e)
                    logger.warning("Error removing %s: %s" % (pipeline_file_path, e))

            self._checkpoint_store = pd.HDFStore(str(pipeline_file_path), mode=mode)
        else:
            self._checkpoint_store = Path(pipeline_file_path)

        self.is_open = True
        logger.debug(f"opened checkpoint.store {pipeline_file_path}")

    def close_store(self):
        """
        Close any known open files
        """

        assert self.is_open, f"Pipeline is not open."

        self.obj.close_open_files()

        if not isinstance(self.store, Path):
            self.store.close()

        self.obj.init_state()  # TODO no?

        logger.debug("close_pipeline")

    def is_readonly(self):
        if self.is_open:
            store = self.store
            if store and not isinstance(store, Path) and store._mode == "r":
                return True
        return False

    def add(self, checkpoint_name: str):
        """
        Create a new checkpoint with specified name.

        Adding a checkpoint will write into the checkpoint store
        all the data required to restore the simulation to its
        current state.

        Parameters
        ----------
        checkpoint_name : str
        """
        timestamp = dt.datetime.now()

        logger.debug("add_checkpoint %s timestamp %s" % (checkpoint_name, timestamp))

        for table_name in self.obj.uncheckpointed_table_names():
            df = self.obj.get_dataframe(table_name)
            logger.debug(f"add_checkpoint {checkpoint_name!r} table {table_name!r}")
            self.write_df(df, table_name, checkpoint_name)

            # remember which checkpoint it was last written
            self.last_checkpoint[table_name] = checkpoint_name
            self.obj.existing_table_status[table_name] = False

        self.last_checkpoint[CHECKPOINT_NAME] = checkpoint_name
        self.last_checkpoint[TIMESTAMP] = timestamp

        # append to the array of checkpoint history
        self.checkpoints.append(self.last_checkpoint.copy())

        # create a pandas dataframe of the checkpoint history, one row per checkpoint
        checkpoints = pd.DataFrame(self.checkpoints)

        # convert empty values to str so PyTables doesn't pickle object types
        for c in checkpoints.columns:
            checkpoints[c] = checkpoints[c].fillna("")

        # write it to the store, overwriting any previous version (no way to simply extend)
        self.write_df(checkpoints, CHECKPOINT_TABLE_NAME)

    def read_df(self, table_name, checkpoint_name=None):
        """
        Read a pandas dataframe from the pipeline store.

        We store multiple versions of all simulation tables, for every checkpoint in which they change,
        so we need to know both the table_name and the checkpoint_name of hte desired table.

        The only exception is the checkpoints dataframe, which just has a table_name

        An error will be raised by HDFStore if the table is not found

        Parameters
        ----------
        table_name : str
        checkpoint_name : str

        Returns
        -------
        df : pandas.DataFrame
            the dataframe read from the store

        """
        store = self.store
        if isinstance(store, Path):
            df = pd.read_parquet(
                store.joinpath(table_name, f"{checkpoint_name}.parquet"),
            )
        else:
            df = store[self.obj.pipeline_table_key(table_name, checkpoint_name)]

        return df

    def write_df(self, df, table_name, checkpoint_name=None):
        """
        Write a pandas dataframe to the pipeline store.

        We store multiple versions of all simulation tables, for every checkpoint in which they change,
        so we need to know both the table_name and the checkpoint_name to label the saved table

        The only exception is the checkpoints dataframe, which just has a table_name,
        although when using the parquet storage format this file is stored as "None.parquet"
        to maintain a simple consistent file directory structure.

        Parameters
        ----------
        df : pandas.DataFrame
            dataframe to store
        table_name : str
            also conventionally the injected table name
        checkpoint_name : str
            the checkpoint at which the table was created/modified
        """

        # coerce column names to str as unicode names will cause PyTables to pickle them
        df.columns = df.columns.astype(str)

        store = self.store
        if isinstance(store, Path):
            store.joinpath(table_name).mkdir(parents=True, exist_ok=True)
            df.to_parquet(store.joinpath(table_name, f"{checkpoint_name}.parquet"))
        else:
            complib = self.obj.settings.pipeline_complib
            if complib is None or len(df.columns) == 0:
                # tables with no columns can't be compressed successfully, so to
                # avoid them getting just lost and dropped they are instead written
                # in fixed format with no compression, which should be just fine
                # since they have no data anyhow.
                store.put(
                    self.obj.pipeline_table_key(table_name, checkpoint_name),
                    df,
                )
            else:
                store.put(
                    self.obj.pipeline_table_key(table_name, checkpoint_name),
                    df,
                    "table",
                    complib=complib,
                )
            store.flush()

    def list_tables(self):
        """
        Return a list of the names of all checkpointed tables
        """
        return [
            name
            for name, checkpoint_name in self.last_checkpoint.items()
            if checkpoint_name and name not in NON_TABLE_COLUMNS
        ]

    def load(self, checkpoint_name: str):
        """
        Load dataframes and restore random number channel state from pipeline hdf5 file.
        This restores the pipeline state that existed at the specified checkpoint in a prior simulation.
        This allows us to resume the simulation after the specified checkpoint

        Parameters
        ----------
        checkpoint_name : str
            model_name of checkpoint to load (resume_after argument to open_pipeline)
        """

        logger.info("load_checkpoint %s" % (checkpoint_name))

        checkpoints = self.read_df(CHECKPOINT_TABLE_NAME)

        if checkpoint_name == LAST_CHECKPOINT:
            checkpoint_name = checkpoints[CHECKPOINT_NAME].iloc[-1]
            logger.info("loading checkpoint '%s'" % checkpoint_name)

        try:
            # truncate rows after target checkpoint
            i = checkpoints[checkpoints[CHECKPOINT_NAME] == checkpoint_name].index[0]
            checkpoints = checkpoints.loc[:i]

            # if the store is not open in read-only mode,
            # write it to the store to ensure so any subsequent checkpoints are forgotten
            if not self.is_readonly() and isinstance(self.store, pd.HDFStore):
                self.write_df(checkpoints, CHECKPOINT_TABLE_NAME)

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

        tables = self.list_tables()

        loaded_tables = {}
        for table_name in tables:
            # read dataframe from pipeline store
            df = self.read_df(
                table_name, checkpoint_name=self.last_checkpoint[table_name]
            )
            logger.info("load_checkpoint table %s %s" % (table_name, df.shape))
            # register it as an orca table
            self.obj.add_table(table_name, df)
            loaded_tables[table_name] = df
            if table_name == "land_use" and "_original_zone_id" in df.columns:
                # The presence of _original_zone_id indicates this table index was
                # decoded to zero-based, so we need to disable offset
                # processing for legacy skim access.
                # TODO: this "magic" column name should be replaced with a mechanism
                #       to write and recover particular settings from the pipeline
                #       store, but we don't have that mechanism yet
                self.obj.settings.offset_preprocessing = True

        # register for tracing in order that tracing.register_traceable_table wants us to register them
        traceable_tables = self.obj.get_injectable("traceable_tables", [])

        from activitysim.core.tracing import register_traceable_table

        for table_name in traceable_tables:
            if table_name in loaded_tables:
                register_traceable_table(
                    self.obj, table_name, loaded_tables[table_name]
                )

        # add tables of known rng channels
        rng_channels = self.obj.get_injectable("rng_channels", [])
        if rng_channels:
            logger.debug("loading random channels %s" % rng_channels)
            for table_name in rng_channels:
                if table_name in loaded_tables:
                    logger.debug("adding channel %s" % (table_name,))
                    self.obj.rng().add_channel(table_name, loaded_tables[table_name])

    def get_inventory(self):
        """
        Get pandas dataframe of info about all checkpoints stored in pipeline

        pipeline doesn't have to be open

        Returns
        -------
        checkpoints_df : pandas.DataFrame

        """

        store = self.store

        if store is not None:
            if isinstance(store, Path):
                df = pd.read_parquet(
                    store.joinpath(CHECKPOINT_TABLE_NAME, "None.parquet")
                )
            else:
                df = store[CHECKPOINT_TABLE_NAME]
        else:
            pipeline_file_path = self.obj.filesystem.get_pipeline_filepath()
            if pipeline_file_path.suffix == ".h5":
                df = pd.read_hdf(pipeline_file_path, CHECKPOINT_TABLE_NAME)
            else:
                df = pd.read_parquet(
                    pipeline_file_path.joinpath(CHECKPOINT_TABLE_NAME, "None.parquet")
                )

        # non-table columns first (column order in df is random because created from a dict)
        table_names = [
            name for name in df.columns.values if name not in NON_TABLE_COLUMNS
        ]

        df = df[NON_TABLE_COLUMNS + table_names]

        return df
