# ActivitySim
# See full license in LICENSE.txt.
import datetime as dt
import logging
import os
from builtins import map, next
from pathlib import Path
from typing import Any

import pandas as pd
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

# name of the first step/checkpoint created when the pipeline is started
INITIAL_CHECKPOINT_NAME = "init"
FINAL_CHECKPOINT_NAME = "final"

# special value for resume_after meaning last checkpoint
LAST_CHECKPOINT = "_"

# single character prefix for run_list model name to indicate that no checkpoint should be saved
NO_CHECKPOINT_PREFIX = "_"

NO_DEFAULT = "throw error if missing"


def split_arg(s, sep, default=""):
    """
    split str s in two at first sep, returning empty string as second result if no sep
    """
    r = s.split(sep, 2)
    r = list(map(str.strip, r))

    arg = r[0]

    if len(r) == 1:
        val = default
    else:
        val = r[1]
        val = {"true": True, "false": False}.get(val.lower(), val)

    return arg, val


class WhaleAttr:
    def __init__(self, member_type, default_init=False):
        self.member_type = member_type
        self._default_init = default_init

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, objtype=None):
        try:
            return instance.context[self.name]
        except (KeyError, AttributeError):
            if self._default_init:
                instance.context[self.name] = self.member_type()
                return instance.context[self.name]
            raise WhaleAccessError(f"{self.name} not initialized for this whale")

    def __set__(self, instance, value):
        if not isinstance(value, self.member_type):
            raise TypeError(f"{self.name} must be {self.member_type} not {type(value)}")
        instance.context[self.name] = value

    def __delete__(self, instance):
        self.__set__(instance, None)


class Whale:
    def __init__(self, context=None):
        if context is None:
            self.context = Context()
            self.init_state()
        elif isinstance(context, Context):
            self.context = context
        else:
            raise TypeError(f"cannot init Whale with {type(context)}")

    def init_state(self, pipeline_file_format="parquet"):
        # most recent checkpoint
        self.last_checkpoint = {}

        # array of checkpoint dicts
        self.checkpoints = []

        from activitysim.core.random import Random  # TOP?

        self.context["prng"] = Random()

        self.open_files = {}

        self.pipeline_store = None

        self._is_open = False
        from activitysim.core.tracing import initialize_traceable_tables  # TOP?

        initialize_traceable_tables(self)

        self.context["_salient_tables"] = {}

    filesystem = WhaleAttr(FileSystem)
    settings = WhaleAttr(Settings)
    network_settings = WhaleAttr(NetworkSettings)
    predicates = WhaleAttr(dict, default_init=True)

    # @property
    # def filesystem(self) -> FileSystem:
    #     try:
    #         return self.context["filesystem"]
    #     except KeyError:
    #         raise WhaleAccessError("filesystem not initialized for this pipeline")
    #
    # @filesystem.setter
    # def filesystem(self, fs: FileSystem):
    #     if not isinstance(fs, FileSystem):
    #         raise TypeError(f"filesystem must be FileSystem not {type(fs)}")
    #     self.context["filesystem"] = fs
    #
    # @property
    # def settings(self) -> Settings:
    #     try:
    #         return self.context["settings"]
    #     except KeyError:
    #         raise WhaleAccessError("settings not initialized for this pipeline")
    #
    # @settings.setter
    # def settings(self, s: Settings):
    #     if not isinstance(s, Settings):
    #         raise TypeError(f"settings must be Settings not {type(s)}")
    #     self.context["settings"] = s
    #
    # @property
    # def network_settings(self) -> NetworkSettings:
    #     try:
    #         return self.context["network_settings"]
    #     except KeyError:
    #         raise WhaleAccessError(
    #             "network_settings not initialized for this pipeline"
    #         )
    #
    # @network_settings.setter
    # def network_settings(self, s: NetworkSettings):
    #     if not isinstance(s, NetworkSettings):
    #         raise TypeError(f"settings must be NetworkSettings not {type(s)}")
    #     self.context["network_settings"] = s

    _RUNNABLE_STEPS = {}
    _LOADABLE_TABLES = {}
    _LOADABLE_OBJECTS = {}
    _PREDICATES = {}

    @property
    def known_table_names(self):
        return self._LOADABLE_TABLES.keys() | self.existing_table_names

    @property
    def existing_table_names(self):
        return self.existing_table_status.keys()

    @property
    def existing_table_status(self) -> dict:
        return self.context["_salient_tables"]

    def uncheckpointed_table_names(self):
        uncheckpointed = []
        for tablename, table_status in self.existing_table_status.items():
            if table_status:
                uncheckpointed.append(tablename)
        return uncheckpointed

    def load_table(self, tablename, overwrite=False, swallow_errors=False):
        """
        Load a table from disk or otherwise programmatically create it.

        Parameters
        ----------
        tablename : str
        overwrite : bool
        swallow_errors : bool

        Returns
        -------
        pandas.DataFrame or xarray.Dataset
        """
        if tablename in self.existing_table_names and not overwrite:
            if swallow_errors:
                return
            raise ValueError(f"table {tablename} already loaded")
        if tablename not in self._LOADABLE_TABLES:
            if swallow_errors:
                return
            raise ValueError(f"table {tablename} has no loading function")
        logger.debug(f"loading table {tablename}")
        try:
            t = self._LOADABLE_TABLES[tablename](self.context)
        except WhaleAccessError:
            if not swallow_errors:
                raise
            else:
                t = None
        if t is not None:
            self.add_table(tablename, t)
        return t

    def get_dataframe(self, tablename):
        t = self.context.get(tablename, None)
        if t is None:
            t = self.load_table(tablename, swallow_errors=False)
        if t is None:
            raise KeyError(tablename)
        if isinstance(t, pd.DataFrame):
            return t
        raise TypeError(f"cannot convert {tablename} to DataFrame")

    def access(self, key, initializer):
        if key not in self.context:
            self.set(key, initializer)
        return self.context[key]

    def get(self, key, default: Any = NO_DEFAULT):
        if default == NO_DEFAULT:
            try:
                result = self.context[key]
            except (KeyError, KeyNotInContextError):
                result = getattr(self.filesystem, key, None)
                if result is None:
                    if key in self._LOADABLE_TABLES:
                        result = self._LOADABLE_TABLES[key](self.context)
                    elif key in self._LOADABLE_OBJECTS:
                        result = self._LOADABLE_OBJECTS[key](self.context)
                if result is None:
                    raise
        else:
            result = self.context.get(key, default)
        if not isinstance(result, (xr.Dataset, xr.DataArray, pd.DataFrame, pd.Series)):
            result = self.context.get_formatted_value(result)
        return result

    def set(self, key, value):
        self.context[key] = value
        for i in self._PREDICATES.get(key, []):
            if i in self.context:
                logger.critical(f"update of {key} clears cached {i}")
                del self.context[i]

    def extract(self, func):
        return func(self)

    get_injectable = get  # legacy function name
    add_injectable = set  # legacy function name

    def rng(self):
        return self.context["prng"]

    @property
    def is_open(self):
        return self._is_open

    @is_open.setter
    def is_open(self, x):
        self._is_open = bool(x)

    def is_readonly(self):
        if self.is_open:
            store = self.get_pipeline_store()
            if store and not isinstance(store, Path) and store._mode == "r":
                return True
        return False

    def pipeline_table_key(self, table_name, checkpoint_name):
        if checkpoint_name:
            key = f"{table_name}/{checkpoint_name}"
        else:
            key = f"/{table_name}"
        return key

    def close_on_exit(self, file, name):
        assert name not in self.open_files
        self.open_files[name] = file

    def close_open_files(self):
        for name, file in self.open_files.items():
            print("Closing %s" % name)
            file.close()
        self.open_files.clear()

    def open_pipeline_store(self, pipeline_file_name, overwrite=False, mode="a"):
        """
        Open the pipeline checkpoint store.

        If the pipeline_file_name setting ends in ".h5", then the pandas
        HDFStore file format is used, otherwise pipeline files are stored
        as parquet files organized in regular file system directories.

        Parameters
        ----------
        overwrite : bool
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

        if self.pipeline_store is not None:
            raise RuntimeError("Pipeline store is already open!")

        pipeline_file_path = self.filesystem.get_pipeline_filepath()

        if pipeline_file_path.suffix == ".h5":
            if overwrite:
                try:
                    if os.path.isfile(pipeline_file_path):
                        logger.debug("removing pipeline store: %s" % pipeline_file_path)
                        os.unlink(pipeline_file_path)
                except Exception as e:
                    print(e)
                    logger.warning("Error removing %s: %s" % (pipeline_file_path, e))

            self.pipeline_store = pd.HDFStore(str(pipeline_file_path), mode=mode)
        else:
            self.pipeline_store = Path(pipeline_file_path)

        logger.debug(f"opened pipeline_store {pipeline_file_path}")

    def get_pipeline_store(self):
        """
        Return the open pipeline hdf5 checkpoint store or return None if it not been opened

        If the pipeline filename ends in ".h5" then the legacy HDF5 pipeline
        is used, otherwise the faster parquet format is used, and the value
        returned here is just the path to the pipeline directory.

        """
        return self.pipeline_store

    def get_rn_generator(self):
        """
        Return the singleton random number object

        Returns
        -------
        activitysim.random.Random
        """
        return self.rng()

    def get_global_constants(self):
        """
        Read global constants from settings file

        Returns
        -------
        constants : dict
            dictionary of constants to add to locals for use by expressions in model spec
        """
        return self.filesystem.read_settings_file("constants.yaml", mandatory=False)

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

        store = self.get_pipeline_store()
        if isinstance(store, Path):
            df = pd.read_parquet(
                store.joinpath(table_name, f"{checkpoint_name}.parquet"),
            )
        else:
            df = store[self.pipeline_table_key(table_name, checkpoint_name)]

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

        store = self.get_pipeline_store()
        if isinstance(store, Path):
            store.joinpath(table_name).mkdir(parents=True, exist_ok=True)
            df.to_parquet(store.joinpath(table_name, f"{checkpoint_name}.parquet"))
        else:
            complib = self.settings.pipeline_complib
            if complib is None or len(df.columns) == 0:
                # tables with no columns can't be compressed successfully, so to
                # avoid them getting just lost and dropped they are instead written
                # in fixed format with no compression, which should be just fine
                # since they have no data anyhow.
                store.put(
                    self.pipeline_table_key(table_name, checkpoint_name),
                    df,
                )
            else:
                store.put(
                    self.pipeline_table_key(table_name, checkpoint_name),
                    df,
                    "table",
                    complib=complib,
                )
            store.flush()

    def add_table(self, name, content, salient=True):
        if salient:
            # mark this salient table as edited, so it can be checkpointed
            # at some later time if desired.
            self.existing_table_status[name] = True
        self.set(name, content)

    def is_table(self, name):
        return name in self.existing_table_status

    # def rewrap(self, table_name, df=None):
    #     """
    #     Add or replace an orca registered table as a unitary DataFrame-backed DataFrameWrapper table
    #
    #     if df is None, then get the dataframe from orca (table_name should be registered, or
    #     an error will be thrown) which may involve evaluating added columns, etc.
    #
    #     If the orca table already exists, deregister it along with any associated columns before
    #     re-registering it.
    #
    #     The net result is that the dataframe is a registered orca DataFrameWrapper table with no
    #     computed or added columns.
    #
    #     Parameters
    #     ----------
    #     table_name
    #     df
    #
    #     Returns
    #     -------
    #         the underlying df of the rewrapped table
    #     """
    #
    #     logger.debug("rewrap table %s inplace=%s" % (table_name, (df is None)))
    #
    #     if self.is_table(table_name):
    #         if df is None:
    #             # # logger.debug("rewrap - orca.get_table(%s)" % (table_name,))
    #             # t = orca.get_table(table_name)
    #             # df = t.to_frame()
    #             df = self.context.get(table_name)
    #         else:
    #             # logger.debug("rewrap - orca.get_raw_table(%s)" % (table_name,))
    #             # don't trigger function call of TableFuncWrapper
    #             # t = orca.get_raw_table(table_name)
    #             df = self.context.get(table_name)
    #
    #     assert df is not None
    #
    #     self.add_table(table_name, df)
    #
    #     return df

    def add_checkpoint(self, checkpoint_name):
        """
        Create a new checkpoint with specified name, write all data required to restore the simulation
        to its current state.

        Detect any changed tables , re-wrap them and write the current version to the pipeline store.
        Write the current state of the random number generator.

        Parameters
        ----------
        checkpoint_name : str
        """
        timestamp = dt.datetime.now()

        logger.debug("add_checkpoint %s timestamp %s" % (checkpoint_name, timestamp))

        for table_name in self.uncheckpointed_table_names():
            df = self.get_dataframe(table_name)
            logger.debug(f"add_checkpoint {checkpoint_name!r} table {table_name!r}")
            self.write_df(df, table_name, checkpoint_name)

            # remember which checkpoint it was last written
            self.last_checkpoint[table_name] = checkpoint_name
            self.existing_table_status[table_name] = False

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

    def registered_tables(self):
        """
        Return a list of the names of all currently registered dataframe tables
        """
        return [name for name in self.existing_table_status if name in self.context]

    def checkpointed_tables(self):
        """
        Return a list of the names of all checkpointed tables
        """

        return [
            name
            for name, checkpoint_name in self.last_checkpoint.items()
            if checkpoint_name and name not in NON_TABLE_COLUMNS
        ]

    def load_checkpoint(self, checkpoint_name):
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
            if not self.is_readonly():
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

        tables = self.checkpointed_tables()

        loaded_tables = {}
        for table_name in tables:
            # read dataframe from pipeline store
            df = self.read_df(
                table_name, checkpoint_name=self.last_checkpoint[table_name]
            )
            logger.info("load_checkpoint table %s %s" % (table_name, df.shape))
            # register it as an orca table
            self.add_table(table_name, df)
            loaded_tables[table_name] = df
            if table_name == "land_use" and "_original_zone_id" in df.columns:
                # The presence of _original_zone_id indicates this table index was
                # decoded to zero-based, so we need to disable offset
                # processing for legacy skim access.
                # TODO: this "magic" column name should be replaced with a mechanism
                #       to write and recover particular settings from the pipeline
                #       store, but we don't have that mechanism yet
                self.settings.offset_preprocessing = True

        # register for tracing in order that tracing.register_traceable_table wants us to register them
        traceable_tables = self.get_injectable("traceable_tables", [])

        from activitysim.core.tracing import register_traceable_table

        for table_name in traceable_tables:
            if table_name in loaded_tables:
                register_traceable_table(self, table_name, loaded_tables[table_name])

        # add tables of known rng channels
        rng_channels = self.get_injectable("rng_channels", [])
        if rng_channels:
            logger.debug("loading random channels %s" % rng_channels)
            for table_name in rng_channels:
                if table_name in loaded_tables:
                    logger.debug("adding channel %s" % (table_name,))
                    self.rng().add_channel(table_name, loaded_tables[table_name])

    @property
    def current_model_name(self) -> str:
        """Name of the currently running model."""
        return self.rng().step_name

    def run_model(self, model_name):
        """
        Run the specified model and add checkpoint for model_name

        Since we use model_name as checkpoint name, the same model may not be run more than once.

        Parameters
        ----------
        model_name : str
            model_name is assumed to be the name of a registered orca step
        """

        if not self.is_open:
            raise RuntimeError("Pipeline not initialized! Did you call open_pipeline?")

        # can't run same model more than once
        if model_name in [
            checkpoint[CHECKPOINT_NAME] for checkpoint in self.checkpoints
        ]:
            raise RuntimeError("Cannot run model '%s' more than once" % model_name)

        self.rng().begin_step(model_name)

        # check for args
        if "." in model_name:
            step_name, arg_string = model_name.split(".", 1)
            args = dict(
                (k, v)
                for k, v in (
                    split_arg(item, "=", default=True) for item in arg_string.split(";")
                )
            )
        else:
            step_name = model_name
            args = {}

        # check for no_checkpoint prefix
        if step_name[0] == NO_CHECKPOINT_PREFIX:
            step_name = step_name[1:]
            checkpoint = False
        else:
            checkpoint = self.intermediate_checkpoint(model_name)

        self.add_injectable("step_args", args)

        self.trace_memory_info(f"pipeline.run_model {model_name} start")

        from activitysim.core.tracing import print_elapsed_time

        t0 = print_elapsed_time()
        logger.info(f"#run_model running step {step_name}")

        instrument = self.settings.instrument
        if instrument is not None:
            try:
                from pyinstrument import Profiler
            except ImportError:
                instrument = False
        if isinstance(instrument, (list, set, tuple)):
            if step_name not in instrument:
                instrument = False
            else:
                instrument = True

        if instrument:
            from pyinstrument import Profiler

            with Profiler() as profiler:
                self.context = run_named_step(step_name, self.context)
            out_file = self.filesystem.get_profiling_file_path(f"{step_name}.html")
            with open(out_file, "wt") as f:
                f.write(profiler.output_html())
        else:
            self.context = run_named_step(step_name, self.context)

        t0 = print_elapsed_time(
            "#run_model completed step '%s'" % model_name, t0, debug=True
        )
        self.trace_memory_info(f"pipeline.run_model {model_name} finished")

        self.add_injectable("step_args", None)

        self.rng().end_step(model_name)
        if checkpoint:
            self.add_checkpoint(model_name)
        else:
            logger.info("##### skipping %s checkpoint for %s" % (step_name, model_name))

    def open_pipeline(self, resume_after=None, mode="a"):
        """
        Start pipeline, either for a new run or, if resume_after, loading checkpoint from pipeline.

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

        if self.is_open:
            raise RuntimeError("Pipeline is already open!")

        self.init_state()
        self.is_open = True

        self.get_rn_generator().set_base_seed(self.get("rng_base_seed", 0))
        pipeline_file_name = self.filesystem.pipeline_file_name

        if resume_after:
            # open existing pipeline
            logger.debug("open_pipeline - open existing pipeline")
            self.open_pipeline_store(pipeline_file_name, overwrite=False, mode=mode)
            try:
                self.load_checkpoint(resume_after)
            except KeyError as err:
                if "checkpoints" in err.args[0]:
                    # no checkpoints initialized, fall back to restart
                    self.last_checkpoint[CHECKPOINT_NAME] = INITIAL_CHECKPOINT_NAME
                    self.add_checkpoint(INITIAL_CHECKPOINT_NAME)
                else:
                    raise
        else:
            # open new, empty pipeline
            logger.debug("open_pipeline - new, empty pipeline")
            self.open_pipeline_store(pipeline_file_name, overwrite=True)
            # - not sure why I thought we needed this?
            # could have exogenous tables or prng instantiation under some circumstance??
            self.last_checkpoint[CHECKPOINT_NAME] = INITIAL_CHECKPOINT_NAME
            # empty table, in case they have turned off all checkpointing
            self.add_checkpoint(INITIAL_CHECKPOINT_NAME)

        logger.debug("open_pipeline complete")

    def last_checkpoint(self):
        """

        Returns
        -------
        last_checkpoint: str
            name of last checkpoint
        """

        assert self.is_open, f"Pipeline is not open."

        return self.last_checkpoint[CHECKPOINT_NAME]

    def close_pipeline(self):
        """
        Close any known open files
        """

        assert self.is_open, f"Pipeline is not open."

        self.close_open_files()

        if not isinstance(self.pipeline_store, Path):
            self.pipeline_store.close()

        self.init_state()

        logger.debug("close_pipeline")

    def intermediate_checkpoint(self, checkpoint_name=None):
        checkpoints = self.settings.checkpoints

        if checkpoints is True or checkpoints is False:
            return checkpoints

        assert isinstance(
            checkpoints, list
        ), f"setting 'checkpoints'' should be True or False or a list"

        return checkpoint_name in checkpoints

    def trace_memory_info(self, event):
        from activitysim.core.mem import trace_memory_info

        return trace_memory_info(event, whale=self)

    def run(self, models, resume_after=None, memory_sidecar_process=None):
        """
        run the specified list of models, optionally loading checkpoint and resuming after specified
        checkpoint.

        Since we use model_name as checkpoint name, the same model may not be run more than once.

        If resume_after checkpoint is specified and a model with that name appears in the models list,
        then we only run the models after that point in the list. This allows the user always to pass
        the same list of models, but specify a resume_after point if desired.

        Parameters
        ----------
        models : [str]
            list of model_names
        resume_after : str or None
            model_name of checkpoint to load checkpoint and AFTER WHICH to resume model run
        memory_sidecar_process : MemorySidecar, optional
            Subprocess that monitors memory usage

        returns:
            nothing, but with pipeline open
        """
        from activitysim.core.tracing import print_elapsed_time

        t0 = print_elapsed_time()

        self.open_pipeline(resume_after)
        t0 = print_elapsed_time("open_pipeline", t0)

        if resume_after == LAST_CHECKPOINT:
            resume_after = self.last_checkpoint[CHECKPOINT_NAME]

        if resume_after:
            logger.info("resume_after %s" % resume_after)
            if resume_after in models:
                models = models[models.index(resume_after) + 1 :]

        self.trace_memory_info("pipeline.run before preload_injectables")

        # preload any bulky injectables (e.g. skims) not in pipeline
        # if inject.get_injectable("preload_injectables", None):
        #     if memory_sidecar_process:
        #         memory_sidecar_process.set_event("preload_injectables")
        #     t0 = print_elapsed_time("preload_injectables", t0)

        self.trace_memory_info("pipeline.run after preload_injectables")

        t0 = print_elapsed_time()
        for model in models:
            if memory_sidecar_process:
                memory_sidecar_process.set_event(model)
            t1 = print_elapsed_time()
            self.run_model(model)
            self.trace_memory_info(f"pipeline.run after {model}")

            from activitysim.core.tracing import log_runtime

            log_runtime(self, model_name=model, start_time=t1)

        if memory_sidecar_process:
            memory_sidecar_process.set_event("finalizing")

        # add checkpoint with final tables even if not intermediate checkpointing
        if not self.intermediate_checkpoint():
            self.add_checkpoint(FINAL_CHECKPOINT_NAME)

        self.trace_memory_info("pipeline.run after run_models")

        t0 = print_elapsed_time("run_model (%s models)" % len(models), t0)

        # don't close the pipeline, as the user may want to read intermediate results from the store

    def get_table(self, table_name, checkpoint_name=None):
        """
        Return pandas dataframe corresponding to table_name

        if checkpoint_name is None, return the current (most recent) version of the table.
        The table can be a checkpointed table or any registered orca table (e.g. function table)

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

        assert self.is_open, f"Pipeline is not open."

        # orca table not in checkpoints (e.g. a merged table)
        if table_name not in self.last_checkpoint and self.is_table(table_name):
            if checkpoint_name is not None:
                raise RuntimeError(
                    "get_table: checkpoint_name ('%s') not supported"
                    "for non-checkpointed table '%s'" % (checkpoint_name, table_name)
                )

            return self.context.get(table_name)

        # if they want current version of table, no need to read from pipeline store
        if checkpoint_name is None:
            if table_name not in self.last_checkpoint:
                raise RuntimeError("table '%s' never checkpointed." % table_name)

            if not self.last_checkpoint[table_name]:
                raise RuntimeError("table '%s' was dropped." % table_name)

            # return orca.get_table(table_name).local
            return self.context.get(table_name)

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
            return self.context.get(table_name)

        return self.read_df(table_name, last_checkpoint_name)

    def get_checkpoints(self):
        """
        Get pandas dataframe of info about all checkpoints stored in pipeline

        pipeline doesn't have to be open

        Returns
        -------
        checkpoints_df : pandas.DataFrame

        """

        store = self.get_pipeline_store()

        if store is not None:
            if isinstance(store, Path):
                df = pd.read_parquet(
                    store.joinpath(CHECKPOINT_TABLE_NAME, "None.parquet")
                )
            else:
                df = store[CHECKPOINT_TABLE_NAME]
        else:
            pipeline_file_path = self.filesystem.get_pipeline_filepath()
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

    # def replace_table(self, table_name, df):
    #     """
    #     Add or replace a orca table, removing any existing added orca columns
    #
    #     The use case for this function is a method that calls to_frame on an orca table, modifies
    #     it and then saves the modified.
    #
    #     orca.to_frame returns a copy, so no changes are saved, and adding multiple column with
    #     add_column adds them in an indeterminate order.
    #
    #     Simply replacing an existing the table "behind the pipeline's back" by calling orca.add_table
    #     risks pipeline to failing to detect that it has changed, and thus not checkpoint the changes.
    #
    #     Parameters
    #     ----------
    #     table_name : str
    #         orca/pipeline table name
    #     df : pandas DataFrame
    #     """
    #
    #     assert self.is_open, f"Pipeline is not open."
    #
    #     if df.columns.duplicated().any():
    #         logger.error(
    #             "replace_table: dataframe '%s' has duplicate columns: %s"
    #             % (table_name, df.columns[df.columns.duplicated()])
    #         )
    #
    #         raise RuntimeError(
    #             "replace_table: dataframe '%s' has duplicate columns: %s"
    #             % (table_name, df.columns[df.columns.duplicated()])
    #         )
    #
    #     self.rewrap(table_name, df)
    #
    #     self.replaced_tables[table_name] = True

    def extend_table(self, table_name, df, axis=0):
        """
        add new table or extend (add rows) to an existing table

        Parameters
        ----------
        table_name : str
            orca/inject table name
        df : pandas DataFrame
        """
        assert axis in [0, 1]

        if self.is_table(table_name):
            table_df = self.get_dataframe(table_name)

            if axis == 0:
                # don't expect indexes to overlap
                assert len(table_df.index.intersection(df.index)) == 0
                missing_df_str_columns = [
                    c
                    for c in table_df.columns
                    if c not in df.columns and table_df[c].dtype == "O"
                ]
            else:
                # expect indexes be same
                assert table_df.index.equals(df.index)
                new_df_columns = [c for c in df.columns if c not in table_df.columns]
                df = df[new_df_columns]
                missing_df_str_columns = []

            # preserve existing column order
            df = pd.concat([table_df, df], sort=False, axis=axis)

            # backfill missing df columns that were str (object) type in table_df
            if axis == 0:
                for c in missing_df_str_columns:
                    df[c] = df[c].fillna("")

        self.add_table(table_name, df)

        return df

    def drop_table(self, table_name):

        if self.is_table(table_name):
            logger.debug("drop_table dropping orca table '%s'" % table_name)
            self.context.pop(table_name, None)
            self.existing_table_status.pop(table_name)

        if table_name in self.last_checkpoint:
            logger.debug(
                "drop_table removing table %s from last_checkpoint" % table_name
            )

            self.last_checkpoint[table_name] = ""

    def cleanup_pipeline(self):
        """
        Cleanup pipeline after successful run

        Open main pipeline if not already open (will be closed if multiprocess)
        Create a single-checkpoint pipeline file with latest version of all checkpointed tables,
        Delete main pipeline and any subprocess pipelines

        Called if cleanup_pipeline_after_run setting is True

        Returns
        -------
        nothing, but with changed state: pipeline file that was open on call is closed and deleted

        """
        # we don't expect to be called unless cleanup_pipeline_after_run setting is True
        assert self.settings.cleanup_pipeline_after_run

        if not self.is_open:
            self.open_pipeline("_")

        assert self.is_open, f"Pipeline is not open."

        FINAL_PIPELINE_FILE_NAME = f"final_{self.filesystem.pipeline_file_name}"
        FINAL_CHECKPOINT_NAME = "final"

        final_pipeline_file_path = self.filesystem.get_output_dir().joinpath(
            FINAL_PIPELINE_FILE_NAME
        )

        # keep only the last row of checkpoints and patch the last checkpoint name
        checkpoints_df = self.get_checkpoints().tail(1).copy()
        checkpoints_df["checkpoint_name"] = FINAL_CHECKPOINT_NAME

        if final_pipeline_file_path.suffix == ".h5":
            with pd.HDFStore(
                str(final_pipeline_file_path), mode="w"
            ) as final_pipeline_store:
                for table_name in self.checkpointed_tables():
                    # patch last checkpoint name for all tables
                    checkpoints_df[table_name] = FINAL_CHECKPOINT_NAME

                    table_df = self.get_table(table_name)
                    logger.debug(
                        f"cleanup_pipeline - adding table {table_name} {table_df.shape}"
                    )

                    final_pipeline_store[table_name] = table_df

                final_pipeline_store[CHECKPOINT_TABLE_NAME] = checkpoints_df
            self.close_pipeline()
        else:
            for table_name in self.checkpointed_tables():
                # patch last checkpoint name for all tables
                checkpoints_df[table_name] = FINAL_CHECKPOINT_NAME

                table_df = self.get_table(table_name)
                logger.debug(
                    f"cleanup_pipeline - adding table {table_name} {table_df.shape}"
                )
                table_df.to_parquet(
                    final_pipeline_file_path.joinpath(
                        table_name, f"{FINAL_CHECKPOINT_NAME}.parquet"
                    )
                )
            checkpoints_df.to_parquet(
                final_pipeline_file_path.joinpath(CHECKPOINT_TABLE_NAME, "None.parquet")
            )

        from activitysim.core.tracing import delete_output_files

        logger.debug(f"deleting all pipeline files except {final_pipeline_file_path}")
        delete_output_files(self, "h5", ignore=[final_pipeline_file_path])
        # TODO: delete nested directory structure.
        delete_output_files(self, "parquet", ignore=[final_pipeline_file_path])

    # @contextlib.contextmanager
    def chunk_log(self, *args, **kwargs):
        from activitysim.core.chunk import chunk_log

        return chunk_log(*args, **kwargs, settings=self.settings)

    def get_output_file_path(self, file_name: str) -> Path:
        prefix = self.get_injectable("output_file_prefix", None)
        if prefix:
            file_name = "%s-%s" % (prefix, file_name)
        return self.filesystem.get_output_dir().joinpath(file_name)
