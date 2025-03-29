from __future__ import annotations

import abc
import datetime as dt
import logging
import os
import warnings
from pathlib import Path
from typing import Optional, TypeVar

import pandas as pd
import pyarrow as pa

from activitysim.core.exceptions import (
    CheckpointFileNotFoundError,
    CheckpointNameNotFoundError,
    StateAccessError,
    TableNameNotFound,
)
from activitysim.core.workflow.accessor import FromState, StateAccessor

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


CheckpointStore = TypeVar("CheckpointStore", bound="GenericCheckpointStore")


class GenericCheckpointStore:
    """Abstract base class defining interface for table storage."""

    @abc.abstractmethod
    def put(
        self,
        table_name: str,
        df: pd.DataFrame,
        complib: str = None,
        checkpoint_name: str = None,
    ) -> None:
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
        table_name : str
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

    @property
    @abc.abstractmethod
    def filename(self) -> Path:
        """Location of this store."""

    def list_checkpoint_names(self) -> list[str]:
        """Get a list of all checkpoint names in this store."""
        try:
            df = self.get_dataframe(CHECKPOINT_TABLE_NAME)
        except Exception:
            return []
        else:
            return list(df.checkpoint_name)

    @classmethod
    def from_hdf(
        cls: CheckpointStore,
        source_filename: Path,
        dest_filename: Path,
        mode: str = "a",
    ) -> CheckpointStore:
        """
        Create a new checkpoint store from an existing HdfStore.

        Parameters
        ----------
        source_filename : path-like
            The filename of the source HDF5 checkpoint file.  This file should
            be the output of an ActivitySim run (or constructed alike).
        dest_filename : path-like
            The filename or directory where a new checkpoint storage will be
            created.
        mode : str
            The file mode used to open the destination.  Must not be a read-only
            mode or this operation will fail.

        Returns
        -------
        CheckpointStore
        """
        hdf_store = HdfStore(source_filename, "r")
        output_store = cls(dest_filename, mode)
        checkpoint_df = hdf_store.get_dataframe(CHECKPOINT_TABLE_NAME)
        output_store.put(CHECKPOINT_TABLE_NAME, checkpoint_df)
        for table_name in checkpoint_df.columns:
            if table_name in NON_TABLE_COLUMNS:
                continue
            checkpoints_written = set()
            for checkpoint_name in checkpoint_df[table_name]:
                if checkpoint_name:
                    df = hdf_store.get_dataframe(table_name, checkpoint_name)
                    if checkpoint_name and checkpoint_name not in checkpoints_written:
                        output_store.put(
                            table_name, df, checkpoint_name=checkpoint_name
                        )
                        checkpoints_written.add(checkpoint_name)
        return output_store

    def _get_store_checkpoint_from_named_checkpoint(
        self, table_name: str, checkpoint_name: str = LAST_CHECKPOINT
    ):
        """
        Get the name of the checkpoint where a table is actually written.

        Checkpoint tables are not re-written if the content has not changed, so
        retrieving a particular table at a given checkpoint can involve back-tracking
        to find where the file was last actually written.

        Parameters
        ----------
        table_name : str
        checkpoint_name : str, optional
            The name of the checkpoint to load.  If not given this function
            will load the last stored checkpoint value.

        Returns
        -------
        str
            The checkpoint to actually load.
        """
        cp_df = self.get_dataframe(CHECKPOINT_TABLE_NAME).set_index(CHECKPOINT_NAME)
        if checkpoint_name == LAST_CHECKPOINT:
            checkpoint_name = cp_df.index[-1]
        try:
            return cp_df.loc[checkpoint_name, table_name]
        except KeyError:
            if checkpoint_name not in cp_df.index:
                raise CheckpointNameNotFoundError(checkpoint_name)
            elif table_name not in cp_df.columns:
                raise TableNameNotFound(table_name)
            else:
                raise


class HdfStore(GenericCheckpointStore):
    """Storage interface for HDF5-based table storage."""

    def __init__(self, filename: Path, mode="a"):
        self._hdf5 = pd.HDFStore(str(filename), mode=mode)

    @property
    def filename(self) -> Path:
        return Path(self._hdf5.filename)

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
    ) -> None:
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
    """Storage interface for parquet-based table storage.

    This store will store each saved table in a parquet-format archive,
    resulting in a hierarchy of separate files in a defined structure, as
    opposed to a single monolithic repository files containing all the data.

    This interface will fall back to storing tables in a gzipped pickle if
    the parquet format fails (as might happen if datatypes for some columns
    are not homogenous and values are stored as "object").
    """

    extension = ".parquetpipeline"

    @staticmethod
    def _to_parquet(df: pd.DataFrame, filename, *args, **kwargs):
        try:
            df.to_parquet(filename, *args, **kwargs)
        except (pa.lib.ArrowInvalid, pa.lib.ArrowTypeError) as err:
            logger.error(
                f"Problem writing to {filename}\n" f"{err}\n" f"falling back to pickle"
            )
            # fallback to pickle, compatible with more dtypes
            df.to_pickle(Path(filename).with_suffix(".pickle.gz"))

    def __init__(self, directory: Path, mode: str = "a", gitignore: bool = True):
        """Initialize a storage interface for parquet-based table storage.

        Parameters
        ----------
        directory : Path
            The file directory for this ParquetStore. If this location does not
            include a ".parquetpipeline" or ".zip" suffix, one is added.
        mode : {"a", "r"}, default "a"
            Mode to open this store, "a"ppend or "r"ead-only.  Zipped stores
            can only be opened in read-only mode.
        gitignore : bool, default True
            If not opened in read-only mode, should a ".gitignore" file be added
            with a global wildcard (**)?  Doing so will help prevent this store
            from being accidentally committed to git.
        """
        directory = Path(directory)
        if directory.suffix == ".zip":
            if mode != "r":
                raise ValueError("can only open a Zip parquet store as read-only.")
        elif directory.suffix != self.extension:
            directory = directory.with_suffix(self.extension)
        self._directory = directory
        self._mode = mode
        if self._mode != "r":
            self._directory.mkdir(parents=True, exist_ok=True)
            if gitignore and not self._directory.joinpath(".gitignore").exists():
                self._directory.joinpath(".gitignore").write_text("**\n")

    @property
    def filename(self) -> Path:
        """The directory location of this ParquetStore."""
        return self._directory

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
    ) -> None:
        if self.is_readonly:
            raise ValueError("store is read-only")
        filepath = self._store_table_path(table_name, checkpoint_name)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if complib == "NOTSET":
            self._to_parquet(pd.DataFrame(df), filepath)
        else:
            self._to_parquet(pd.DataFrame(df), filepath, compression=complib)

    def get_dataframe(
        self, table_name: str, checkpoint_name: str = None
    ) -> pd.DataFrame:
        if table_name != CHECKPOINT_TABLE_NAME and checkpoint_name is None:
            checkpoint_name = LAST_CHECKPOINT
        if self._directory.suffix == ".zip":
            import io
            import zipfile

            zip_internal_filename = self._store_table_path(
                table_name, checkpoint_name
            ).relative_to(self._directory)
            with zipfile.ZipFile(self._directory, mode="r") as zipf:
                namelist = set(zipf.namelist())
                if zip_internal_filename.as_posix() in namelist:
                    with zipf.open(zip_internal_filename.as_posix()) as zipo:
                        return pd.read_parquet(zipo)
                elif (
                    zip_internal_filename.with_suffix(".pickle.gz").as_posix()
                    in namelist
                ):
                    with zipf.open(
                        zip_internal_filename.with_suffix(".pickle.gz").as_posix()
                    ) as zipo:
                        return pd.read_pickle(zipo, compression="gzip")
                checkpoint_name_ = self._get_store_checkpoint_from_named_checkpoint(
                    table_name, checkpoint_name
                )
                if checkpoint_name_ != checkpoint_name:
                    return self.get_dataframe(table_name, checkpoint_name_)
                raise FileNotFoundError(str(zip_internal_filename))
        target_path = self._store_table_path(table_name, checkpoint_name)
        if target_path.exists():
            return pd.read_parquet(target_path)
        elif target_path.with_suffix(".pickle.gz").exists():
            return pd.read_pickle(target_path.with_suffix(".pickle.gz"))
        else:
            # the direct-read failed, check for backtracking checkpoint
            if checkpoint_name is not None:
                checkpoint_name_ = self._get_store_checkpoint_from_named_checkpoint(
                    table_name, checkpoint_name
                )
                if checkpoint_name_ != checkpoint_name:
                    return self.get_dataframe(table_name, checkpoint_name_)
            raise FileNotFoundError(target_path)

    @property
    def is_readonly(self) -> bool:
        return self._mode == "r"

    @property
    def is_open(self) -> bool:
        return self._directory is not None and self._directory.is_dir()

    def close(self) -> None:
        """Close this store."""
        pass

    def make_zip_archive(self, output_filename) -> Path:
        """
        Compress this pipeline into a zip archive.

        Parameters
        ----------
        output_filename

        Returns
        -------
        Path
            Filename of the resulting zipped store.
        """
        output_filename = Path(output_filename)
        import zipfile

        if output_filename.suffix != ".zip":
            output_filename = output_filename.with_suffix(".zip")
        with zipfile.ZipFile(output_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(self._directory):
                files = [f for f in files if not f[0] == "."]
                for f in files:
                    arcname = Path(root).joinpath(f).relative_to(self._directory)
                    zipf.write(Path(root).joinpath(f), arcname=arcname)

        return output_filename

    def wipe(self):
        """
        Remove this store, including all subdirectories.
        """
        if self.is_readonly:
            raise ValueError("store is readonly")
        walked = list(os.walk(self._directory))
        while walked:
            root, dirs, files = walked.pop(-1)
            for f in files:
                if f.endswith(".parquet"):
                    os.unlink(os.path.join(root, f))
            # after removing all parquet files, is this directory basically empty?
            should_drop_root = True
            file_list = {f for f in Path(root).glob("**/*") if f.is_file()}
            for f in file_list:
                if f not in {".gitignore", ".DS_Store"}:
                    should_drop_root = False
            if should_drop_root:
                os.rmdir(root)


class NullStore(GenericCheckpointStore):
    """
    A NullStore is a dummy that emulates a checkpoint store object.

    It never writes anything to disk and is primarily used to for
    temporary data to prevent accidentally overwriting content in
    a "real" store.
    """

    def put(
        self,
        table_name: str,
        df: pd.DataFrame,
        complib: str = "NOTSET",
        checkpoint_name: str = None,
    ) -> None:
        pass

    def get_dataframe(
        self, table_name: str, checkpoint_name: str = None
    ) -> pd.DataFrame:
        raise ValueError("no data is actually stored in NullStore")

    @property
    def is_readonly(self) -> bool:
        return False

    @property
    def is_open(self) -> bool:
        return True

    def close(self) -> None:
        """Close this store."""
        pass


class Checkpoints(StateAccessor):
    """
    State accessor for checkpointing operations.

    See :ref:`State.checkpoint <state-checkpoint>` for more detailed
    documentation.
    """

    last_checkpoint: dict = FromState(
        default_init=True,
        doc="""
    Metadata about the last saved checkpoint.

    This dictionary contains the name of the checkpoint, a timestamp, and
    the checkpoint-lookup for all relevant tables.
    """,
    )
    checkpoints: list[dict] = FromState(
        default_init=True,
        doc="""
    Metadata about various saved checkpoint(s).

    Each item in this list is a dictionary similar to the `last_checkpoint`.
    """,
    )
    _checkpoint_store: GenericCheckpointStore | None = FromState(
        default_value=None,
        doc="""
    The store where checkpoints are written.
    """,
    )

    def __get__(self, instance, objtype=None) -> Checkpoints:
        # derived __get__ changes annotation, aids in type checking
        return super().__get__(instance, objtype)

    def initialize(self):
        self.last_checkpoint = {}
        self.checkpoints: list[dict] = []
        self._checkpoint_store = None

    @property
    def store(self) -> GenericCheckpointStore:
        """The store where checkpoints are written."""
        if self._checkpoint_store is None:
            self.open_store()
        return self._checkpoint_store

    def store_is_open(self) -> bool:
        """Whether this checkpoint store is open."""
        if self._checkpoint_store is None:
            return False
        return self._checkpoint_store.is_open

    def default_pipeline_file_path(self):
        if self._obj is None:
            # a freestanding accessor not bound to a parent State is not
            # typical but does happen when Sphinx generates documentation
            return self
        prefix = self._obj.get("pipeline_file_prefix", None)
        if prefix is None:
            return self._obj.filesystem.get_pipeline_filepath()
        else:
            pipeline_file_name = str(self._obj.filesystem.pipeline_file_name)
            pipeline_file_name = f"{prefix}-{pipeline_file_name}"
            return self._obj.filesystem.get_output_dir().joinpath(pipeline_file_name)

    def open_store(
        self, pipeline_file_name: Optional[Path] = None, overwrite=False, mode="a"
    ):
        """
        Open the checkpoint store.

        The format for the checkpoint store is determined by the
        `checkpoint_format` setting in the top-level Settings.

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
            pipeline_file_path = self.default_pipeline_file_path()
        else:
            pipeline_file_path = Path(pipeline_file_name)

        if self._obj.settings.checkpoint_format == "hdf":
            if overwrite:
                try:
                    if os.path.isfile(pipeline_file_path):
                        logger.debug("removing pipeline store: %s" % pipeline_file_path)
                        os.unlink(pipeline_file_path)
                except Exception as e:
                    print(e)
                    logger.warning(f"Error removing {pipeline_file_path}: {e}")

            self._checkpoint_store = HdfStore(pipeline_file_path, mode=mode)
        else:
            self._checkpoint_store = ParquetStore(pipeline_file_path, mode=mode)

        logger.debug(f"opened checkpoint.store {pipeline_file_path}")

    def close_store(self):
        """
        Close the checkpoint storage.
        """
        if self._checkpoint_store is not None:
            self.store.close()
            self._checkpoint_store = None
        logger.debug("checkpoint.close_store")

    def is_readonly(self):
        if self._checkpoint_store is not None:
            try:
                return self._checkpoint_store.is_readonly
            except AttributeError:
                return None
        return False

    def last_checkpoint_name(self):
        if self.last_checkpoint:
            try:
                return self.last_checkpoint.get("checkpoint_name", None)
            except AttributeError:
                return None
        else:
            return None

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

        for table_name in self._obj.uncheckpointed_table_names():
            df = self._obj.get_dataframe(table_name)
            logger.debug(f"add_checkpoint {checkpoint_name!r} table {table_name!r}")
            self._write_df(df, table_name, checkpoint_name)

            # remember which checkpoint it was last written
            self.last_checkpoint[table_name] = checkpoint_name
            self._obj.existing_table_status[table_name] = False

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
        self._write_df(checkpoints, CHECKPOINT_TABLE_NAME)

    def _read_df(
        self, table_name, checkpoint_name=None, store: GenericCheckpointStore = None
    ):
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
        if store is None:
            store = self.store
        return store.get_dataframe(table_name, checkpoint_name)

    def _write_df(
        self,
        df: pd.DataFrame,
        table_name: str,
        checkpoint_name: str = None,
        store: GenericCheckpointStore = None,
    ):
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
        store : GenericCheckpointStore, optional
            Write to this store instead of the default store.
        """
        if store is None:
            store = self.store

        # coerce column names to str as unicode names will cause PyTables to pickle them
        df.columns = df.columns.astype(str)

        store.put(
            table_name,
            df,
            complib=self._obj.settings.pipeline_complib,
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

    def load(self, checkpoint_name: str, store=None):
        """
        Load dataframes and restore random number channel state from pipeline hdf5 file.
        This restores the pipeline state that existed at the specified checkpoint in a prior simulation.
        This allows us to resume the simulation after the specified checkpoint

        Parameters
        ----------
        checkpoint_name : str
            model_name of checkpoint to load (resume_after argument to open_pipeline)
        """

        logger.info(f"load_checkpoint {checkpoint_name} from {self.store.filename}")

        try:
            checkpoints = self._read_df(CHECKPOINT_TABLE_NAME, store=store)
        except FileNotFoundError as err:
            raise CheckpointFileNotFoundError(err) from None

        if checkpoint_name == LAST_CHECKPOINT:
            checkpoint_name = checkpoints[CHECKPOINT_NAME].iloc[-1]
            logger.info(f"loading checkpoint '{checkpoint_name}'")

        try:
            # truncate rows after target checkpoint
            i = checkpoints[checkpoints[CHECKPOINT_NAME] == checkpoint_name].index[0]
            checkpoints = checkpoints.loc[:i]

            # if the store is not open in read-only mode,
            # write it to the store to ensure so any subsequent checkpoints are forgotten
            if (
                store is None
                and not self.is_readonly
                and isinstance(self.store, pd.HDFStore)
            ):
                self._write_df(checkpoints, CHECKPOINT_TABLE_NAME)

        except IndexError:
            msg = f"Couldn't find checkpoint '{checkpoint_name}' in checkpoints"
            print(checkpoints[CHECKPOINT_NAME])
            logger.error(msg)
            raise RuntimeError(msg) from None

        # convert pandas dataframe back to array of checkpoint dicts
        checkpoints = checkpoints.to_dict(orient="records")

        # drop tables with empty names
        for checkpoint in checkpoints:
            for key in list(checkpoint.keys()):
                if key not in NON_TABLE_COLUMNS and not checkpoint[key]:
                    del checkpoint[key]

        if store is None:
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
            last_checkpoint = self.last_checkpoint

        else:
            last_checkpoint = checkpoints[-1]
            tables = [
                name
                for name, checkpoint_name in last_checkpoint.items()
                if checkpoint_name and name not in NON_TABLE_COLUMNS
            ]

        loaded_tables = {}
        for table_name in tables:
            # read dataframe from pipeline store
            df = self._read_df(
                table_name, checkpoint_name=last_checkpoint[table_name], store=store
            )
            logger.info("load_checkpoint table %s %s" % (table_name, df.shape))
            # register it as an workflow table
            self._obj.add_table(table_name, df)
            loaded_tables[table_name] = df
            if table_name == "land_use" and "_original_zone_id" in df.columns:
                # The presence of _original_zone_id indicates this table index was
                # decoded to zero-based, so we need to disable offset
                # processing for legacy skim access.
                # TODO: this "magic" column name should be replaced with a mechanism
                #       to write and recover particular settings from the pipeline
                #       store, but we don't have that mechanism yet
                try:
                    self._obj.settings.offset_preprocessing = True
                except StateAccessError:
                    pass
                    # self.obj.default_settings()
                    # self.obj.settings.offset_preprocessing = True

        # register for tracing in order that tracing.register_traceable_table wants us to register them
        traceable_tables = self._obj.tracing.traceable_tables

        for table_name in traceable_tables:
            if table_name in loaded_tables:
                self._obj.tracing.register_traceable_table(
                    table_name, loaded_tables[table_name]
                )

        # add tables of known rng channels
        rng_channels = self._obj.get_injectable("rng_channels", [])
        if rng_channels:
            logger.debug("loading random channels %s" % rng_channels)
            for table_name in rng_channels:
                if table_name in loaded_tables:
                    logger.debug("adding channel %s" % (table_name,))
                    self._obj.rng().add_channel(table_name, loaded_tables[table_name])

        if store is not None:
            # we have loaded from an external store, so we make a new checkpoint
            # with the same name as the one we just loaded.
            self.add(checkpoint_name)

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

        self._obj.init_state()

        if resume_after:
            # open existing pipeline
            logger.debug("checkpoint.restore - open existing pipeline")
            if self._checkpoint_store is None:
                self.open_store(overwrite=False, mode=mode)
            try:
                self.load(resume_after)
            except (KeyError, CheckpointFileNotFoundError) as err:
                if (
                    isinstance(err, CheckpointFileNotFoundError)
                    or "checkpoints" in err.args[0]
                ):
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

    def restore_from(self, location: Path, checkpoint_name: str = LAST_CHECKPOINT):
        """
        Restore state from an alternative pipeline store.

        The checkpoint history is collapsed when reading out of an alternative
        store location, given the presumption that if the use wanted to load a
        prior intermediate state, that could be done so from the same outside
        store, and the history does not need to be also preserved in the active
        checkpoint store.

        Parameters
        ----------
        location : Path-like
            Location of pipeline store to load.
        checkpoint_name : str
            name of checkpoint to load from pipeline store
        """
        self._obj.init_state()
        logger.debug(f"checkpoint.restore_from - opening {location}")
        if isinstance(location, str):
            location = Path(location)
        if self._obj.settings.checkpoint_format == "hdf":
            from_store = HdfStore(location, mode="r")
        else:
            from_store = ParquetStore(location, mode="r")
        self.load(checkpoint_name, store=from_store)
        logger.debug(f"checkpoint.restore_from of {checkpoint_name} complete")

    def check_against(
        self,
        location: Path,
        checkpoint_name: str,
        strict_categoricals: bool = False,
        rtol: float = 1.0e-5,
        atol: float = 1.0e-8,
    ):
        """
        Check that the tables in this State match those in an archived pipeline.

        Parameters
        ----------
        location : Path-like
        checkpoint_name : str
        strict_categoricals : bool, default False
            If True, check that categorical columns have the same categories
            in both the current state and the checkpoint.  Otherwise, the dtypes
            of categorical columns are ignored, and only the values themselves are
            checked to confirm they match.
        rtol : float, default 1e-5
            Relative tolerance. Passed through to `assert_frame_equal`.
        atol : float, default 1e-8
            Absolute tolerance. Passed through to `assert_frame_equal`.

        Raises
        ------
        AssertionError
            If any registered table does not match.
        """
        __tracebackhide__ = True  # don't show this code in pytest outputs

        for table_name in self._obj.registered_tables():
            local_table = self._obj.get_dataframe(table_name)
            logger.info(f"table {table_name!r}: shalpe1 {local_table.shape}")

        from .state import State

        ref_state = State()
        ref_state.default_settings()
        ref_state.checkpoint._checkpoint_store = NullStore()

        if isinstance(location, str):
            location = Path(location)
        if self._obj.settings.checkpoint_format == "hdf":
            from_store = HdfStore(location, mode="r")
        else:
            from_store = ParquetStore(location, mode="r")
        ref_state.checkpoint.load(checkpoint_name, store=from_store)
        registered_tables = ref_state.registered_tables()
        if len(registered_tables) == 0:
            logger.warning("no tables checked")
        for table_name in registered_tables:
            local_table = self._obj.get_dataframe(table_name)
            ref_table = ref_state.get_dataframe(table_name)
            cols_in_run_but_not_ref = set(local_table.columns) - set(ref_table.columns)
            cols_in_ref_but_not_run = set(ref_table.columns) - set(local_table.columns)
            if cols_in_ref_but_not_run:
                msg = f"checkpoint {checkpoint_name!r} table {table_name!r} column names mismatch"
                if cols_in_run_but_not_ref:
                    msg += (
                        f"\ncolumns found but not expected: {cols_in_run_but_not_ref}"
                    )
                if cols_in_ref_but_not_run:
                    msg += (
                        f"\ncolumns expected but not found: {cols_in_ref_but_not_run}"
                    )
                raise AssertionError(msg)
            elif cols_in_run_but_not_ref:
                # if there are extra columns output that were not expected, but
                # we at least have all the column names that were expected, just
                # warn, not error
                warnings.warn(
                    f"checkpoint {checkpoint_name!r} table {table_name!r}\n"
                    f"columns found but not expected: {cols_in_run_but_not_ref}"
                )
            if len(ref_table.columns) == 0:
                try:
                    pd.testing.assert_index_equal(local_table.index, ref_table.index)
                except Exception as err:
                    raise AssertionError(
                        f"checkpoint {checkpoint_name!r} table {table_name!r}, {str(err)}"
                    )
                else:
                    logger.info(f"table {table_name!r}: ok")
            else:
                try:
                    pd.testing.assert_frame_equal(
                        local_table[ref_table.columns],
                        ref_table,
                        check_dtype=False,
                        rtol=rtol,
                        atol=atol,
                    )
                except Exception as err:
                    if not strict_categoricals:
                        try:
                            pd.testing.assert_frame_equal(
                                local_table[ref_table.columns],
                                ref_table,
                                check_dtype=False,
                                check_categorical=False,
                                rtol=rtol,
                                atol=atol,
                            )
                        except Exception as err2:
                            raise AssertionError(
                                f"checkpoint {checkpoint_name!r} table {table_name!r}, {str(err)}\nfrom: {str(err2)}"
                            )
                        else:
                            warnings.warn(
                                f"checkpoint {checkpoint_name!r} table {table_name!r}, "
                                f"values match but categorical dtype does not"
                            )
                    else:
                        raise AssertionError(
                            f"checkpoint {checkpoint_name!r} table {table_name!r}, {str(err)}"
                        )
                else:
                    logger.info(f"table {table_name!r}: ok")

    def cleanup(self):
        """
        Remove intermediate checkpoints from pipeline.

        These are the steps to clean up:

        - Open main pipeline if not already open (it may be closed if
          running with multiprocessing),
        - Create a new single-checkpoint pipeline file with the latest
          version of all checkpointed tables,
        - Delete the original main pipeline and any subprocess pipelines

        This method is generally called at the end of a successful model
        run, as it removes the intermediate checkpoint files.

        Called if cleanup_pipeline_after_run setting is True

        """
        # we don't expect to be called unless cleanup_pipeline_after_run setting is True
        if not self._obj.settings.cleanup_pipeline_after_run:
            logger.warning("will not clean up, `cleanup_pipeline_after_run` is False")
            return

        if not self.store_is_open():
            self.restore(LAST_CHECKPOINT)

        assert self.store_is_open(), f"Pipeline is not open."

        FINAL_PIPELINE_FILE_NAME = f"final_{self._obj.filesystem.pipeline_file_name}"
        FINAL_CHECKPOINT_NAME = "final"

        if self._obj.settings.checkpoint_format == "hdf":
            # constructing the path manually like this will not create a
            # subdirectory that competes with the HDF5 filename.
            final_pipeline_file_path = self._obj.filesystem.get_output_dir().joinpath(
                FINAL_PIPELINE_FILE_NAME
            )
        else:
            # calling for a subdir ensures that the subdirectory exists.
            final_pipeline_file_path = self._obj.filesystem.get_output_dir(
                subdir=FINAL_PIPELINE_FILE_NAME
            )

        # keep only the last row of checkpoints and patch the last checkpoint name
        checkpoints_df = self.get_inventory().tail(1).copy()
        checkpoints_df["checkpoint_name"] = FINAL_CHECKPOINT_NAME

        if self._obj.settings.checkpoint_format == "hdf":
            with pd.HDFStore(
                str(final_pipeline_file_path), mode="w"
            ) as final_pipeline_store:
                for table_name in self.list_tables():
                    # patch last checkpoint name for all tables
                    checkpoints_df[table_name] = FINAL_CHECKPOINT_NAME

                    table_df = self._obj.get_table(table_name)
                    logger.debug(
                        f"cleanup_pipeline - adding table {table_name} {table_df.shape}"
                    )

                    final_pipeline_store[table_name] = table_df

                final_pipeline_store[CHECKPOINT_TABLE_NAME] = checkpoints_df
            self.close_store()
        else:
            for table_name in self.list_tables():
                # patch last checkpoint name for all tables
                checkpoints_df[table_name] = FINAL_CHECKPOINT_NAME

                table_df = self._obj.get_table(table_name)
                logger.debug(
                    f"cleanup_pipeline - adding table {table_name} {table_df.shape}"
                )
                table_dir = final_pipeline_file_path.joinpath(table_name)
                if not table_dir.exists():
                    table_dir.mkdir(parents=True)
                ParquetStore._to_parquet(
                    table_df, table_dir.joinpath(f"{FINAL_CHECKPOINT_NAME}.parquet")
                )
            final_pipeline_file_path.joinpath(CHECKPOINT_TABLE_NAME).mkdir(
                parents=True, exist_ok=True
            )
            ParquetStore._to_parquet(
                checkpoints_df,
                final_pipeline_file_path.joinpath(
                    CHECKPOINT_TABLE_NAME, "None.parquet"
                ),
            )

        logger.debug(f"deleting all pipeline files except {final_pipeline_file_path}")
        self._obj.tracing.delete_output_files("h5", ignore=[final_pipeline_file_path])

        # delete all ParquetStore except final
        pqps = list(
            self._obj.filesystem.get_output_dir().glob(f"**/*{ParquetStore.extension}")
        )
        for pqp in pqps:
            if pqp.name != final_pipeline_file_path.name:
                ParquetStore(pqp).wipe()

    def load_dataframe(self, table_name, checkpoint_name=None):
        """
        Return pandas dataframe corresponding to table_name

        if checkpoint_name is None, return the current (most recent) version of the table.
        The table can be a checkpointed table or any registered table (e.g. function table)

        if checkpoint_name is specified, return table as it was at that checkpoint
        (the most recently checkpointed version of the table at or before checkpoint_name)

        Parameters
        ----------
        table_name : str
        checkpoint_name : str or None

        Returns
        -------
        df : pandas.DataFrame
        """

        if table_name not in self.last_checkpoint and self._obj.is_table(table_name):
            if checkpoint_name is not None:
                raise RuntimeError(
                    f"checkpoint.dataframe: checkpoint_name ({checkpoint_name!r}) not "
                    f"supported for non-checkpointed table {table_name!r}"
                )

            return self._obj.get_dataframe(table_name)

        # if there is no checkpoint name given, do not attempt to read from store
        if checkpoint_name is None:
            if table_name not in self.last_checkpoint:
                raise RuntimeError("table '%s' never checkpointed." % table_name)

            if not self.last_checkpoint[table_name]:
                raise RuntimeError("table '%s' was dropped." % table_name)

            return self._obj.get_dataframe(table_name)

        # find the requested checkpoint
        checkpoint = next(
            (x for x in self.checkpoints if x["checkpoint_name"] == checkpoint_name),
            None,
        )
        if checkpoint is None:
            raise RuntimeError("checkpoint '%s' not in checkpoints." % checkpoint_name)

        # find the checkpoint that table was written to store
        last_checkpoint_name = checkpoint.get(table_name, None)

        if not last_checkpoint_name:
            raise RuntimeError(
                "table '%s' not in checkpoint '%s'." % (table_name, checkpoint_name)
            )

        # if this version of table is same as current
        if self.last_checkpoint.get(table_name, None) == last_checkpoint_name:
            return self._obj.get_dataframe(table_name)

        return self._read_df(table_name, last_checkpoint_name)
