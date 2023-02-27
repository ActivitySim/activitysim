from __future__ import annotations

import importlib
import io
import logging
import os
import sys
import warnings
from builtins import map, next
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pyarrow as pa
import xarray as xr
from pypyr.context import Context
from sharrow.dataset import construct as _dataset_construct

from activitysim.core.configuration import FileSystem, NetworkSettings, Settings
from activitysim.core.exceptions import StateAccessError
from activitysim.core.workflow.checkpoint import Checkpoints
from activitysim.core.workflow.extending import Extend
from activitysim.core.workflow.logging import Logging
from activitysim.core.workflow.report import Reporting
from activitysim.core.workflow.runner import Runner
from activitysim.core.workflow.steps import run_named_step
from activitysim.core.workflow.tracing import Tracing

from .checkpoint import CHECKPOINT_NAME, LAST_CHECKPOINT

# ActivitySim
# See full license in LICENSE.txt.


logger = logging.getLogger(__name__)

# name of the checkpoint dict keys
# (which are also columns in the checkpoints dataframe stored in hte pipeline store)

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


class StateAttr:
    def __init__(self, member_type, default_init=False):
        self.member_type = member_type
        self._default_init = default_init

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, objtype=None):
        try:
            return instance._context[self.name]
        except (KeyError, AttributeError):
            if self._default_init:
                instance._context[self.name] = self.member_type()
                return instance._context[self.name]
            raise StateAccessError(f"{self.name} not initialized for this state")

    def __set__(self, instance, value):
        if not isinstance(value, self.member_type):
            raise TypeError(f"{self.name} must be {self.member_type} not {type(value)}")
        instance._context[self.name] = value

    def __delete__(self, instance):
        self.__set__(instance, None)


class State:
    def __init__(self, context=None):
        self._pipeline_store: pd.HDFStore | Path | None = None
        """Location of checkpoint storage"""

        self.open_files: dict[str, io.TextIOBase] = {}
        """Files to close when state is destroyed or re-initialized."""

        if context is None:
            self._context = Context()
            self.init_state()
        elif isinstance(context, Context):
            self._context = context
        else:
            raise TypeError(f"cannot init {type(self)} with {type(context)}")

    def __del__(self):
        self.close_open_files()

    def __contains__(self, key):
        """
        Check if a key is already stored in this state's context.

        This does *not* check if the key is automatically loadable, it only
        checks if a cached value has already been stored.

        Parameters
        ----------
        key : str

        Returns
        -------
        bool
        """
        return self._context.__contains__(key)

    def copy(self):
        return self.__class__(context=Context(self._context.copy()))

    def init_state(self):
        self.checkpoint.initialize()

        self.close_open_files()

        from activitysim.core.random import Random  # TOP?

        self._context["prng"] = Random()
        self._initialize_prng()

        self.tracing.initialize()
        self._context["_salient_tables"] = {}

    def _initialize_prng(self, base_seed=None):
        from activitysim.core.random import Random

        self._context["prng"] = Random()
        if base_seed is None:
            try:
                self.settings
            except StateAccessError:
                base_seed = 0
            else:
                base_seed = self.settings.rng_base_seed
        self._context["prng"].set_base_seed(base_seed)

    def import_extensions(self, ext: str | Iterable[str] = None, append=True):
        if ext is None:
            return
        if isinstance(ext, str):
            ext = [ext]
        if append:
            extensions = self.get("imported_extensions", [])
        else:
            extensions = []
        if self.filesystem.working_dir:
            working_dir = self.filesystem.working_dir
        else:
            working_dir = Path.cwd()
        for e in ext:
            basepath, extpath = os.path.split(working_dir.joinpath(e))
            if not basepath:
                basepath = "."
            sys.path.insert(0, os.path.abspath(basepath))
            try:
                importlib.import_module(extpath)
            except ImportError as err:
                logger.exception("ImportError")
                raise
            except Exception as err:
                logger.exception(f"Error {err}")
                raise
            finally:
                del sys.path[0]
            extensions.append(e)
        self.set("imported_extensions", extensions)

    filesystem = StateAttr(FileSystem)
    settings = StateAttr(Settings)
    network_settings = StateAttr(NetworkSettings)
    predicates = StateAttr(dict, default_init=True)

    checkpoint = Checkpoints()
    logging = Logging()
    tracing = Tracing()
    extend = Extend()
    report = Reporting()

    @classmethod
    def make_default(
        cls, working_dir: Path = None, settings: dict[str, Any] = None, **kwargs
    ) -> "State":
        """
        Convenience constructor for mostly default States.

        Parameters
        ----------
        working_dir : Path-like
            If a directory, then this directory is the working directory.  Or,
            if the given path is actually a file, then the directory where the
            file lives is the working directory (typically as a convenience for
            using __file__ in testing).
        settings : Mapping[str, Any]
            Override settings values.
        **kwargs
            All other keyword arguments are forwarded to the
            initialize_filesystem method.

        Returns
        -------
        State
        """
        if working_dir:
            working_dir = Path(working_dir)
            if working_dir.is_file():
                working_dir = working_dir.parent
        self = cls().initialize_filesystem(working_dir, **kwargs)
        if self.filesystem.get_config_file_path(
            self.filesystem.settings_file_name
        ).exists():
            self.load_settings()
        else:
            self.default_settings()
        if settings:
            for k, v in settings.items():
                setattr(self.settings, k, v)
        return self

    @classmethod
    def make_temp(
        cls, source: Path = None, checkpoint_name: str = LAST_CHECKPOINT
    ) -> "State":
        """
        Initialize state with a temporary directory.

        Parameters
        ----------
        source : Path-like, optional
            Location of pipeline store to use to initialize this object.
        checkpoint_name : str, optional
            name of checkpoint to load from source store, defaults to
            the last checkpoint found

        Returns
        -------
        State
        """
        import tempfile

        temp_dir = tempfile.TemporaryDirectory()
        temp_dir_path = Path(temp_dir.name)
        temp_dir_path.joinpath("configs").mkdir()
        temp_dir_path.joinpath("data").mkdir()
        temp_dir_path.joinpath("configs/settings.yaml").write_text("# empty\n")
        state = cls.make_default(temp_dir_path)
        state._context["_TEMP_DIR_"] = temp_dir
        if source is not None:
            state.checkpoint.restore_from(source, checkpoint_name)
        return state

    def initialize_filesystem(
        self,
        working_dir=None,
        *,
        configs_dir=("configs",),
        data_dir=("data",),
        output_dir="output",
        profile_dir=None,
        cache_dir=None,
        settings_file_name="settings.yaml",
        pipeline_file_name="pipeline",
        **silently_ignored_kwargs,
    ) -> "State":
        if isinstance(configs_dir, (str, Path)):
            configs_dir = (configs_dir,)
        if isinstance(data_dir, (str, Path)):
            data_dir = (data_dir,)

        fs = dict(
            configs_dir=configs_dir,
            data_dir=data_dir,
            output_dir=output_dir,
            settings_file_name=settings_file_name,
            pipeline_file_name=pipeline_file_name,
        )
        if working_dir is not None:
            fs["working_dir"] = working_dir
        if profile_dir is not None:
            fs["profile_dir"] = profile_dir
        if cache_dir is not None:
            fs["cache_dir"] = cache_dir
        try:
            self.filesystem: FileSystem = FileSystem.parse_obj(fs)
        except Exception as err:
            print(err)
            raise
        return self

    def default_settings(self, force=False) -> "State":
        """
        Initialize with all default settings, rather than reading from a file.

        Parameters
        ----------
        force : bool, default False
            If settings are already loaded, this method does nothing unless
            this argument is true, in which case all existing settings are
            discarded in favor of the defaults.
        """
        try:
            _ = self.settings
            if force:
                raise StateAccessError
        except StateAccessError:
            self.settings = Settings()
        return self

    def load_settings(self) -> "State":
        # read settings file
        raw_settings = self.filesystem.read_settings_file(
            self.filesystem.settings_file_name,
            mandatory=True,
            include_stack=False,
        )

        # the settings can redefine the cache directories.
        cache_dir = raw_settings.pop("cache_dir", None)
        if cache_dir:
            if self.filesystem.cache_dir != cache_dir:
                logger.warning(f"settings file changes cache_dir to {cache_dir}")
                self.filesystem.cache_dir = cache_dir
        self.settings: Settings = Settings.parse_obj(raw_settings)

        extra_settings = set(self.settings.__dict__) - set(Settings.__fields__)

        if extra_settings:
            warnings.warn(
                "Writing arbitrary model values as top-level key in settings.yaml "
                "is deprecated, make them sub-keys of `other_settings` instead.",
                DeprecationWarning,
            )
            logger.warning(f"Found the following unexpected settings:")
            if self.settings.other_settings is None:
                self.settings.other_settings = {}
            for k in extra_settings:
                logger.warning(f" - {k}")
                self.settings.other_settings[k] = getattr(self.settings, k)
                delattr(self.settings, k)

        return self

    _RUNNABLE_STEPS = {}
    _LOADABLE_TABLES = {}
    _LOADABLE_OBJECTS = {}
    _PREDICATES = {}
    _TEMP_NAMES = set()  # never checkpointed

    @property
    def known_table_names(self):
        return self._LOADABLE_TABLES.keys() | self.existing_table_names

    @property
    def existing_table_names(self):
        return self.existing_table_status.keys()

    @property
    def existing_table_status(self) -> dict:
        return self._context["_salient_tables"]

    def uncheckpointed_table_names(self):
        uncheckpointed = []
        for tablename, table_status in self.existing_table_status.items():
            if table_status and tablename not in self._TEMP_NAMES:
                uncheckpointed.append(tablename)
        return uncheckpointed

    def _load_or_create_dataset(
        self, table_name, overwrite=False, swallow_errors=False
    ):
        """
        Load a table from disk or otherwise programmatically create it.

        Parameters
        ----------
        table_name : str
        overwrite : bool
        swallow_errors : bool

        Returns
        -------
        xarray.Dataset
        """
        if table_name in self.existing_table_names and not overwrite:
            if swallow_errors:
                return self.get_dataframe(table_name)
            raise ValueError(f"table {table_name} already loaded")
        if table_name not in self._LOADABLE_TABLES:
            if swallow_errors:
                return
            raise ValueError(f"table {table_name} has no loading function")
        logger.debug(f"loading table {table_name}")
        try:
            t = self._LOADABLE_TABLES[table_name](self._context)
        except StateAccessError:
            if not swallow_errors:
                raise
            else:
                t = None
        if t is not None:
            self.add_table(table_name, t)
        return t

    def get_dataset(
        self,
        table_name: str,
        column_names: Optional[list[str]] = None,
        as_copy: bool = False,
    ) -> xr.Dataset:
        """
        Get a workflow table or dataset as a xarray.Dataset.

        Parameters
        ----------
        table_name : str
            Name of table or dataset to get.
        column_names : list[str], optional
            Include only these columns or variables in the dataset.
        as_copy : bool, default False
            Return a copy of the dataset instead of the original.

        Returns
        -------
        xarray.Dataset
        """
        t = self._context.get(table_name, None)
        if t is None:
            t = self._load_or_create_dataset(table_name, swallow_errors=False)
        if t is None:
            raise KeyError(table_name)
        t = _dataset_construct(t)
        if isinstance(t, xr.Dataset):
            if column_names is not None:
                t = t[column_names]
            if as_copy:
                return t.copy()
            else:
                return t
        raise TypeError(f"cannot convert {table_name} to Dataset")

    def get_dataframe(
        self,
        tablename: str,
        columns: Optional[list[str]] = None,
        as_copy: bool = True,
    ) -> pd.DataFrame:
        """
        Get a workflow table as a pandas.DataFrame.

        Parameters
        ----------
        tablename : str
            Name of table to get.
        columns : list[str], optional
            Include only these columns in the dataframe.
        as_copy : bool, default True
            Return a copy of the dataframe instead of the original.

        Returns
        -------
        DataFrame
        """
        t = self._context.get(tablename, None)
        if t is None:
            t = self._load_or_create_dataset(tablename, swallow_errors=False)
        if t is None:
            raise KeyError(tablename)
        if isinstance(t, pd.DataFrame):
            if columns is not None:
                t = t[columns]
            if as_copy:
                return t.copy()
            else:
                return t
        elif isinstance(t, xr.Dataset):
            # this route through pyarrow is generally faster than xarray.to_pandas
            return t.single_dim.to_pyarrow().to_pandas()
        raise TypeError(f"cannot convert {tablename} to DataFrame")

    def get_dataarray(
        self,
        tablename: str,
        item: str,
        as_copy: bool = True,
    ) -> xr.DataArray:
        """
        Get a workflow table item as ax xarray.DataArray.

        Parameters
        ----------
        tablename : str
            Name of table to get.
        item : str
            Name of item within table.
        as_copy : bool, default True
            Return a copy of the data instead of the original.

        Returns
        -------
        DataArray
        """
        t = self._context.get(tablename, None)
        if t is None:
            t = self._load_or_create_dataset(tablename, swallow_errors=False)
        if t is None:
            raise KeyError(tablename)
        if isinstance(t, pd.DataFrame):
            if as_copy:
                return t[item].to_xarray().copy()
            else:
                return t[item].to_xarray()
        raise TypeError(f"cannot convert {tablename}.{item} to DataArray")

    def get_dataframe_index_name(self, tablename: str) -> str:
        """
        Get the index name for a workflow table.

        Parameters
        ----------
        tablename : str
            Name of table to get.

        Returns
        -------
        str
        """
        t = self._context.get(tablename, None)
        if t is None:
            t = self._load_or_create_dataset(tablename, swallow_errors=False)
        if t is None:
            raise KeyError(tablename)
        if isinstance(t, pd.DataFrame):
            return t.index.name
        raise TypeError(f"cannot get index name for {tablename}")

    def get_pyarrow(
        self, tablename: str, columns: Optional[list[str] | str] = None
    ) -> pa.Table:
        """
        Get a workflow table as a pyarrow.Table.

        Parameters
        ----------
        tablename : str
            Name of table to get.
        columns : list[str] or str, optional
            Include only these columns in the dataframe.

        Returns
        -------
        pyarrow.Table
        """
        if isinstance(columns, str):
            columns = [columns]
        t = self._context.get(tablename, None)
        if t is None:
            t = self._load_or_create_dataset(tablename, swallow_errors=False)
        if t is None:
            raise KeyError(tablename)
        if isinstance(t, pd.DataFrame):
            t = pa.Table.from_pandas(t, preserve_index=True, columns=columns)
        if isinstance(t, pa.Table):
            if columns is not None:
                t = t.select(columns)
            return t
        raise TypeError(f"cannot convert {tablename} to pyarrow.Table")

    def access(self, key, initializer):
        if key not in self._context:
            self.set(key, initializer)
        return self._context[key]

    def get(self, key, default: Any = NO_DEFAULT):
        if not isinstance(key, str):
            key_name = getattr(key, "__name__", None)
            if key_name in self._LOADABLE_TABLES or key_name in self._LOADABLE_OBJECTS:
                key = key_name
            if key_name in self._RUNNABLE_STEPS:
                raise ValueError(
                    f"cannot `get` {key_name}, it is a step, try State.run.{key_name}()"
                )
        result = self._context.get(key, None)
        if result is None:
            try:
                result = getattr(self.filesystem, key, None)
            except StateAccessError:
                result = None
        if result is None:
            if key in self._LOADABLE_TABLES:
                result = self._LOADABLE_TABLES[key](self._context)
            elif key in self._LOADABLE_OBJECTS:
                result = self._LOADABLE_OBJECTS[key](self._context)
        if result is None:
            if default != NO_DEFAULT:
                result = default
            else:
                self._context.assert_key_has_value(
                    key=key, caller=self.__class__.__name__
                )
                raise KeyError(key)
        if not isinstance(result, (xr.Dataset, xr.DataArray, pd.DataFrame, pd.Series)):
            result = self._context.get_formatted_value(result)
        return result

    def set(self, key, value):
        """
        Set a new value for a key in the context.

        Also removes from the context all other keys predicated on this key.
        They can be regenerated later (from fresh inputs) if needed.

        Parameters
        ----------
        key : str
        """
        self._context[key] = value
        for i in self._PREDICATES.get(key, []):
            if i in self._context:
                logger.debug(f"update of {key} clears cached {i}")
                self.drop(i)

    def drop(self, key):
        """
        Remove a key from the context.

        Also removes from the context all other keys predicated on this key.

        Parameters
        ----------
        key : str
        """
        del self._context[key]
        for i in self._PREDICATES.get(key, []):
            if i in self._context:
                logger.debug(f"dropping {key} clears cached {i}")
                self.drop(i)

    def extract(self, func):
        return func(self)

    get_injectable = get  # legacy function name
    add_injectable = set  # legacy function name

    def rng(self):
        return self._context["prng"]

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

    # def open_pipeline_store(self, pipeline_file_name:Optional[Path]=None, overwrite=False, mode="a"):
    #     """
    #     Open the pipeline checkpoint store.
    #
    #     If the pipeline_file_name setting ends in ".h5", then the pandas
    #     HDFStore file format is used, otherwise pipeline files are stored
    #     as parquet files organized in regular file system directories.
    #
    #     Parameters
    #     ----------
    #     pipeline_file_name : Path-like, optional
    #         An explicit pipeline file path.  If not given, the default pipeline
    #         file path is opened.
    #     overwrite : bool, default False
    #         delete file before opening (unless resuming)
    #     mode : {'a', 'w', 'r', 'r+'}, default 'a'
    #         ``'r'``
    #             Read-only; no data can be modified.
    #         ``'w'``
    #             Write; a new file is created (an existing file with the same
    #             name would be deleted).
    #         ``'a'``
    #             Append; an existing file is opened for reading and writing,
    #             and if the file does not exist it is created.
    #         ``'r+'``
    #             It is similar to ``'a'``, but the file must already exist.
    #     """
    #
    #     if self._pipeline_store is not None:
    #         raise RuntimeError("Pipeline store is already open!")
    #
    #     pipeline_file_path = pipeline_file_name or self.filesystem.get_pipeline_filepath()
    #
    #     if pipeline_file_path.suffix == ".h5":
    #         if overwrite:
    #             try:
    #                 if os.path.isfile(pipeline_file_path):
    #                     logger.debug("removing pipeline store: %s" % pipeline_file_path)
    #                     os.unlink(pipeline_file_path)
    #             except Exception as e:
    #                 print(e)
    #                 logger.warning("Error removing %s: %s" % (pipeline_file_path, e))
    #
    #         self._pipeline_store = pd.HDFStore(str(pipeline_file_path), mode=mode)
    #     else:
    #         self._pipeline_store = Path(pipeline_file_path)
    #
    #     logger.debug(f"opened pipeline_store {pipeline_file_path}")

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
        try:
            filesystem = self.filesystem
        except StateAccessError:
            return {}
        else:
            return filesystem.read_settings_file("constants.yaml", mandatory=False)

    def add_table(self, name, content, salient=None):
        if salient is None:
            salient = name not in self._TEMP_NAMES
        if salient:
            # mark this salient table as edited, so it can be checkpointed
            # at some later time if desired.
            self.existing_table_status[name] = True
        self.set(name, content)

    def is_table(self, name):
        return name in self.existing_table_status

    def registered_tables(self):
        """
        Return a list of the names of all currently registered dataframe tables
        """
        return [name for name in self.existing_table_status if name in self._context]

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

        # if not self.is_open:
        #     raise RuntimeError("Pipeline not initialized! Did you call open_pipeline?")

        # can't run same model more than once
        if model_name in [
            checkpoint[CHECKPOINT_NAME] for checkpoint in self.checkpoint.checkpoints
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
            checkpoint = self.should_save_checkpoint(model_name)

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
                self._context = run_named_step(step_name, self._context)
            out_file = self.filesystem.get_profiling_file_path(f"{step_name}.html")
            with open(out_file, "wt") as f:
                f.write(profiler.output_html())
        else:
            self._context = run_named_step(step_name, self._context)

        t0 = print_elapsed_time(
            "#run_model completed step '%s'" % model_name, t0, debug=True
        )
        self.trace_memory_info(f"pipeline.run_model {model_name} finished")

        self.add_injectable("step_args", None)

        self.rng().end_step(model_name)
        if checkpoint:
            self.checkpoint.add(model_name)
        else:
            logger.info("##### skipping %s checkpoint for %s" % (step_name, model_name))

    def close_pipeline(self):
        """
        Close any known open files
        """

        self.close_open_files()
        self.checkpoint.close_store()

        self.init_state()

        logger.debug("close_pipeline")

    def should_save_checkpoint(self, checkpoint_name=None) -> bool:
        checkpoints = self.settings.checkpoints

        if checkpoints is True or checkpoints is False:
            return checkpoints

        assert isinstance(
            checkpoints, list
        ), f"setting 'checkpoints'' should be True or False or a list"

        return checkpoint_name in checkpoints

    def trace_memory_info(self, event, trace_ticks=0):
        from activitysim.core.mem import trace_memory_info

        return trace_memory_info(event, state=self, trace_ticks=trace_ticks)

    run = Runner()

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

        if table_name not in self.checkpoint.last_checkpoint and self.is_table(
            table_name
        ):
            if checkpoint_name is not None:
                raise RuntimeError(
                    f"get_table: checkpoint_name ({checkpoint_name!r}) not "
                    f"supported for non-checkpointed table {table_name!r}"
                )

            return self._context.get(table_name)

        # if they want current version of table, no need to read from pipeline store
        if checkpoint_name is None:
            if table_name not in self.checkpoint.last_checkpoint:
                raise RuntimeError("table '%s' never checkpointed." % table_name)

            if not self.checkpoint.last_checkpoint[table_name]:
                raise RuntimeError("table '%s' was dropped." % table_name)

            return self._context.get(table_name)

        # find the requested checkpoint
        checkpoint = next(
            (
                x
                for x in self.checkpoint.checkpoints
                if x["checkpoint_name"] == checkpoint_name
            ),
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
        if (
            self.checkpoint.last_checkpoint.get(table_name, None)
            == last_checkpoint_name
        ):
            return self._context.get(table_name)

        return self.checkpoint._read_df(table_name, last_checkpoint_name)

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
            self._context.pop(table_name, None)
            self.existing_table_status.pop(table_name)

        if table_name in self.checkpoint.last_checkpoint:
            logger.debug(
                "drop_table removing table %s from last_checkpoint" % table_name
            )
            self.checkpoint.last_checkpoint[table_name] = ""

    def get_output_file_path(self, file_name: str, prefix: str | bool = None) -> Path:
        if prefix is None or prefix is True:
            prefix = self.get_injectable("output_file_prefix", None)
        if prefix:
            file_name = "%s-%s" % (prefix, file_name)
        return self.filesystem.get_output_dir().joinpath(file_name)

    def get_log_file_path(self, file_name: str, prefix=True) -> Path:
        prefix = self.get_injectable("output_file_prefix", None)
        if prefix:
            file_name = "%s-%s" % (prefix, file_name)
        return self.filesystem.get_log_file_path(file_name)

    def set_step_args(self, args=None):
        assert isinstance(args, dict) or args is None
        self.add_injectable("step_args", args)

    def get_step_arg(self, arg_name, default=NO_DEFAULT):
        args = self.get_injectable("step_args")

        assert isinstance(args, dict)
        if arg_name not in args and default == NO_DEFAULT:
            raise "step arg '%s' not found and no default" % arg_name

        return args.get(arg_name, default)
