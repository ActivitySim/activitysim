from __future__ import annotations

import abc
import datetime as dt
import logging
import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from activitysim.core.workflow.accessor import WhaleAccessor

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


class GenericCheckpointStore:
    @abc.abstractmethod
    def put(
        self,
        table_name: str,
        df: pd.DataFrame,
        complib: str = None,
        checkpoint_name: str = None,
    ):
        """
        Store a table.

        Parameters
        ----------
        table_name : str
        df : pd.DataFrame
        complib : str
            Name of compression library to use.
        checkpoint_name : str, optional
            The checkpoint version name to use for this table.
        """

    @abc.abstractmethod
    def get_dataframe(
        self, table_name: str, checkpoint_name: str = None
    ) -> pd.DataFrame:
        """
        Load table from store as a pandas DataFrame.

        Parameters
        ----------
        key : str
        checkpoint_name : str, optional
            The checkpoint version name to use for this table.

        Returns
        -------
        pd.DataFrame
        """

    @property
    @abc.abstractmethod
    def is_readonly(self) -> bool:
        """This store is read-only."""

    @property
    @abc.abstractmethod
    def is_open(self) -> bool:
        """This store is open."""

    @abc.abstractmethod
    def close(self) -> None:
        """Close this store."""


class HdfStore(GenericCheckpointStore):
    """Storage interface for HDF5-based table storage."""

    def __init__(self, filename: Path, mode="a"):
        self._hdf5 = pd.HDFStore(str(filename), mode=mode)

    def _store_table_key(self, table_name, checkpoint_name):
        if checkpoint_name:
            key = f"{table_name}/{checkpoint_name}"
        else:
            key = f"/{table_name}"
        return key

    def put(
        self,
        table_name: str,
        df: pd.DataFrame,
        complib: str = None,
        checkpoint_name: str = None,
    ):
        key = self._store_table_key(table_name, checkpoint_name)
        if complib is None or len(df.columns) == 0:
            # tables with no columns can't be compressed successfully, so to
            # avoid them getting just lost and dropped they are instead written
            # in fixed format with no compression, which should be just fine
            # since they have no data anyhow.
            self._hdf5.put(key, df)
        else:
            self._hdf5.put(key, df, "table", complib=complib)
        self._hdf5.flush()

    def get_dataframe(
        self, table_name: str, checkpoint_name: str = None
    ) -> pd.DataFrame:
        key = self._store_table_key(table_name, checkpoint_name)
        return self._hdf5[key]

    @property
    def is_readonly(self) -> bool:
        return self._hdf5._mode == "r"

    @property
    def is_open(self) -> bool:
        return self._hdf5.is_open

    def close(self) -> None:
        """Close this store."""
        self._hdf5.close()


class ParquetStore(GenericCheckpointStore):
    """Storage interface for parquet-based table storage."""

    def __init__(self, directory: Path, mode: str = "a"):
        self._directory = directory
        self._mode = mode

    def _store_table_path(self, table_name, checkpoint_name):
        if checkpoint_name:
            return self._directory.joinpath(table_name, f"{checkpoint_name}.parquet")
        else:
            return self._directory.joinpath(f"{table_name}.parquet")

    def put(
        self,
        table_name: str,
        df: pd.DataFrame,
        complib: str = "NOTSET",
        checkpoint_name: str = None,
    ):
        if self.is_readonly:
            raise ValueError("store is read-only")
        filepath = self._store_table_path(table_name, checkpoint_name)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if complib == "NOTSET":
            pd.DataFrame(df).to_parquet(
                path=filepath,
            )
        else:
            pd.DataFrame(df).to_parquet(path=filepath, compression=complib)

    def get_dataframe(
        self, table_name: str, checkpoint_name: str = None
    ) -> pd.DataFrame:
        return pd.read_parquet(self._store_table_path(table_name, checkpoint_name))

    @property
    def is_readonly(self) -> bool:
        return self._mode == "r"

    @property
    def is_open(self) -> bool:
        return self._directory is not None and self._directory.is_dir()

    def close(self) -> None:
        """Close this store."""
        pass


class Checkpoints(WhaleAccessor):
    def __init__(self):
        self.initialize()

    def initialize(self):
        self.last_checkpoint = {}
        self.checkpoints: list[dict] = []
        self._checkpoint_store = None

    @property
    def store(self) -> GenericCheckpointStore:
        if self._checkpoint_store is None:
            self.open_store()
        return self._checkpoint_store

    def store_is_open(self) -> bool:
        if self._checkpoint_store is None:
            return False
        return self._checkpoint_store.is_open

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

            self._checkpoint_store = HdfStore(pipeline_file_path, mode=mode)
        else:
            self._checkpoint_store = ParquetStore(pipeline_file_path, mode=mode)

        logger.debug(f"opened checkpoint.store {pipeline_file_path}")

    def close_store(self):
        """
        Close any known open files
        """
        if self._checkpoint_store is not None:
            self.store.close()
        self.__init__()
        logger.debug("close_store")

    @property
    def is_readonly(self):
        if self._checkpoint_store is not None:
            return self._checkpoint_store.is_readonly
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
        return self.store.get_dataframe(table_name, checkpoint_name)

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

        self.store.put(
            table_name,
            df,
            complib=self.obj.settings.pipeline_complib,
            checkpoint_name=checkpoint_name,
        )

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
            if not self.is_readonly and isinstance(self.store, pd.HDFStore):
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
        traceable_tables = self.obj.tracing.traceable_tables

        for table_name in traceable_tables:
            if table_name in loaded_tables:
                self.obj.tracing.register_traceable_table(
                    table_name, loaded_tables[table_name]
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
        df = self.store.get_dataframe(CHECKPOINT_TABLE_NAME)
        # non-table columns first (column order in df is random because created from a dict)
        table_names = [
            name for name in df.columns.values if name not in NON_TABLE_COLUMNS
        ]

        df = df[NON_TABLE_COLUMNS + table_names]

        return df

    def restore(self, resume_after=None, mode="a"):
        """
        Restore state from checkpoints.

        This can be used with "resume_after" to get the correct checkpoint,
        or for a new run.

        If resume_after, then we expect the pipeline hdf5 file to exist and contain
        checkpoints from a previous run, including a checkpoint with name specified in resume_after

        Parameters
        ----------
        resume_after : str or None
            name of checkpoint to load from pipeline store
        mode : {'a', 'w', 'r', 'r+'}, default 'a'
            same as for typical opening of H5Store.  Ignored unless resume_after
            is not None.  This is here to allow read-only pipeline for benchmarking.
        """

        self.obj.init_state()

        if resume_after:
            # open existing pipeline
            logger.debug("checkpoint.restore - open existing pipeline")
            if self._checkpoint_store is None:
                self.open_store(overwrite=False, mode=mode)
            try:
                self.load(resume_after)
            except KeyError as err:
                if "checkpoints" in err.args[0]:
                    # no checkpoints initialized, fall back to restart
                    self.last_checkpoint[CHECKPOINT_NAME] = INITIAL_CHECKPOINT_NAME
                    self.add(INITIAL_CHECKPOINT_NAME)
                else:
                    raise
            logger.debug(f"restore from checkpoint {resume_after} complete")
        else:
            # open new, empty pipeline
            logger.debug("checkpoint.restore - new, empty pipeline")
            if self._checkpoint_store is None:
                self.open_store(overwrite=True)
            # - not sure why I thought we needed this?
            # could have exogenous tables or prng instantiation under some circumstance??
            self.last_checkpoint[CHECKPOINT_NAME] = INITIAL_CHECKPOINT_NAME
            # empty table, in case they have turned off all checkpointing
            self.add(INITIAL_CHECKPOINT_NAME)

            logger.debug(f"restore from tabula rasa complete")
