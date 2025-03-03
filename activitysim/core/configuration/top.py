from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import model_validator, validator

from activitysim.core.configuration.base import PydanticBase, Union


class InputTable(PydanticBase):
    """
    The features that define an input table to be read by ActivitySim.
    """

    tablename: str
    """Name of the injected table"""

    filename: Path = None
    """
    Name of the CSV or HDF5 file to read.

    If not provided, defaults to `input_store`
    """

    index_col: Union[str, None] = "NOTSET"
    """table column to use for the index"""

    rename_columns: dict[str, str] = None
    """
    Dictionary of column name mappings.

    This allows for renaming data columns from the original names found in the
    header of the input file itself, into the names used internally by
    ActivitySim, in configuration and specification files.
    """

    recode_columns: dict[str, str] = None
    """
    Dictionary of column recoding instructions.

    Certain columns of data, notably TAZ and MAZ id's, are more efficiently
    stored as the index (offset) position of each value within a fixed array of
    values, instead of as the value itself.  To recode a column into this offset
    format, give the value "zero-based" for that column name.  This will replace
    the named column with a RangeIndex, starting from zero, and will create a
    lookup column of the original values, which is not used by ActivitySim other
    than to recode other related variables, or to reconstitute the original
    labels for a final output table.  This zero-based recoding is typically
    done for the `zone_id` field in the land_use table, but might also be done
    elsewhere.

    Alternatively, for columns that contain *references* to recoded data, give
    the recode instruction as "tablename.fieldname" (often, "land_use.zone_id").
    This will trigger a remapping of each value according to the stored lookup
    table for the original values, transforming the values in other columns to
    be consistent with the recoded zero-based values.  For example, if the
    `zone_id` field in the land_use table has been recoded to be zero-based,
    then the home_zone_id in the households table needs to be recoded to match.

    Note that recoding is done after renaming, so the key values in this mapping
    should correspond to the internally used names and not the original column
    names that appear in the input file (if they have been renamed).
    """

    keep_columns: list[str] = None
    """
    Columns to keep once read in to memory.

    Save only the columns needed for modeling or analysis to save on memory
    and file I/O.  If not given, all columns in the input file will be read
    and retained.
    """

    drop_columns: list[str] = None
    """
    Columns to drop once read in to memory.

    Save only the columns needed for modeling or analysis to save on memory
    and file I/O.  If not given, all columns in the input file will be read
    and retained.
    """

    h5_tablename: str = None
    """table name if reading from HDF5 and different from `tablename`"""

    dtypes: dict[str, str] = None
    """
    dtypes for loaded columns
    """


class OutputTable(PydanticBase):
    tablename: str
    """The name of the pipeline table to write out."""

    decode_columns: dict[str, str] = None
    """
    A mapping indicating columns to decode when writing out results.

    Column decoding is the inverse of column recoding, such that the original
    mapped values are restored.  For example, if TAZ ID's in the zone_id column
    of the land_use table have been recoded to zero-based, then any output
    column that gives a TAZ (i.e., household home zones, work or school
    locations, trip or tour origins and destinations) can and probably should be
    decoded from zero-based back into nominal TAZ ID's. If every value in the
    column is to be decoded, simply give the decode instruction in the same
    manner as the recode instruction, "tablename.fieldname" (often,
    "land_use.zone_id").

    For some columns, like work or school locations, only non-negative values
    should be decoded, as negative values indicate an absence of choice.  In
    these cases, the "tablename.fieldname" can be prefixed with a "nonnegative"
    filter, seperated by a pipe character (e.g. "nonnegative | land_use.zone_id").
    """


class OutputTables(PydanticBase):
    """Instructions on how to write out final pipeline tables."""

    h5_store: bool = False
    """Write tables into a single HDF5 store instead of individual CSVs."""

    file_type: Literal["csv", "parquet", "h5"] = "csv"
    """
    Specifies the file type for output tables. Options are limited to 'csv',
    'h5' or 'parquet'. Only applied if h5_store is set to False."""

    action: str
    """Whether to 'include' or 'skip' the enumerated tables in `tables`."""

    prefix: str = "final_"
    """This prefix is added to the filename of each output table."""

    sort: bool = False
    """Sort output in each table consistent with well defined column names."""

    tables: list[Union[str, OutputTable]] = None
    """
    A list of pipeline tables to include or to skip when writing outputs.

    If `action` is 'skip', the values in this list must all be simple
    strings giving the names of tables to skip.  Also, no decoding will be
    applied to any output tables in this case.

    If `action` is 'include', the values in this list can be either simple
    strings giving the names of tables to include, or :class:`OutputTable`
    definitions giving a name and decoding instructions.

    If omitted, the all tables are written out and no decoding will be
    applied to any output tables.
    """


class MultiprocessStepSlice(PydanticBase, extra="forbid"):
    """
    Instructions on how to slice tables for each subprocess.

    .. versionchanged:: 1.3

        In ActivitySim versions 1.2 and earlier, slicing instructions for
        multiprocess steps allowed for an "except" instruction, which has
        been renamed to be "exclude" to avoid problems from using a reserved
        Python keyword.
    """

    tables: list[str]
    """
    The names of tables that are to be sliced for multiprocessing.

    The index of the first table in the 'tables' list is the primary_slicer.
    Any other tables listed are dependent tables with either ref_cols to the
    primary_slicer or with the same index (i.e. having an index with the same
    name). This cascades, so any tables dependent on the primary_table can in
    turn have dependent tables that will be sliced by index or ref_col.

    For instance, if the primary_slicer is households, then persons can be
    sliced because it has a ref_col to (column with the same same name as) the
    household table index. And the tours table can be sliced since it has a
    ref_col to persons. Tables can also be sliced by index. For instance the
    person_windows table can be sliced because it has an index with the same
    names as the persons table.
    """

    exclude: Union[bool, str, list[str]] = None
    """
    Optional list of tables not to slice even if they have a sliceable index name.

    Or set to `True` or "*" to exclude all tables not explicitly listed in
    `tables`.

    Note in ActivitySim versions 1.2 and earlier, this option was named "except"
    instead of "exclude", but that is a reserved python keyword and cannot be
    used as a Pydantic field name.
    """


class MultiprocessStep(PydanticBase):
    """
    A contiguous group of model components that are multiprocessed together.
    """

    name: str
    """A descriptive name for this multiprocessing step."""

    begin: str
    """The first component that is part of this multiprocessing step."""

    num_processes: int = None
    """
    The number of processes to use in this multiprocessing step.

    If not provided, the default overall number of processes set in the main
    settings file is used.
    """

    slice: MultiprocessStepSlice = None
    """Instructions on how to slice tables for each subprocess."""

    chunk_size: int = None


class Settings(PydanticBase, extra="allow", validate_assignment=True):
    """
    The overall settings for the ActivitySim model system.

    The input for these settings is typically stored in one main YAML file,
    usually called ``settings.yaml``.

    Note that this implementation is presently used only for generating
    documentation, but future work may migrate the settings implementation to
    actually use this pydantic code to validate the settings before running
    the model.
    """

    models: list[str] = None
    """
    list of model steps to run - auto ownership, tour frequency, etc.

    See :ref:`model_steps` for more details about each step.
    """

    multiprocess: bool = False
    """Enable multiprocessing for this model."""

    num_processes: int = None
    """
    If running in multiprocessing mode, use this number of processes by default.

    If not given or set to 0, the number of processes to use is set to
    half the number of available CPU cores, plus 1.
    """

    multiprocess_steps: list[MultiprocessStep] = None
    """A list of multiprocess steps."""

    resume_after: str | None = None
    """to resume running the data pipeline after the last successful checkpoint"""

    input_table_list: list[InputTable] = None
    """list of table names, indices, and column re-maps for each table in `input_store`"""

    input_store: str = None
    """HDF5 inputs file"""

    create_input_store: bool = False
    """
    Write the inputs as read in back to an HDF5 store.

    If enabled, this writes the store to the outputs folder to use for subsequent
    model runs, as reading HDF5 can be faster than reading CSV files."""

    households_sample_size: int = None
    """
    Number of households to sample and simulate

    If omitted or set to 0, ActivitySim will simulate all households.
    """
    trace_hh_id: int | None = None
    """
    Trace this household id

    If omitted, no tracing is written out
    """

    trace_od: tuple[int, int] | None = None
    """
    Trace origin, destination pair in accessibility calculation

    If omitted, no tracing is written out.
    """

    chunk_training_mode: Literal[
        "disabled", "training", "production", "adaptive", "explicit"
    ] = "disabled"
    """
    The method to use for chunk training.

    * "disabled"
        All chunking is disabled. If you have enough RAM, this is the fastest
        mode, but it requires potentially a lot of RAM.
    * "training"
        The model is run in training mode, which tracks the amount of memory
        used by each table by submodel and writes the results to a cache file
        that is then re-used for production runs. This mode is significantly
        slower than production mode since it does significantly more memory
        inspection.
    * "production"
        The model is run in production mode, using the cache file created in
        training mode. If no such file is found, the model falls back to
        training mode. This mode is significantly faster than training mode, as
        it uses the cached memory inspection results to determine chunk sizes.
    * "adaptive"
        Like production mode, any existing cache file is used to determine the
        starting chunk settings, but the model also updates the cache settings
        based on additional memory inspection. This may additionally improve the
        cache settings to reduce runtimes when run in production mode, but at
        the cost of some slowdown during the run to accommodate extra memory
        inspection.
    * "explicit"
        The model is run without memory inspection, and the chunk cache file is
        not used, even if it exists. Instead, the chunk size settings are
        explicitly set in the settings file of each compatible model step.  Only
        those steps that have an "explicit_chunk" setting are chunkable with
        this mode, all other steps are run without chunking.

    See :ref:`chunk_size` for more details.
    """

    chunk_size: int = 0
    """
    Approximate amount of RAM to allocate to ActivitySim for batch processing.

    See :ref:`chunk_size` for more details.
    """

    chunk_method: Literal[
        "bytes",
        "uss",
        "hybrid_uss",
        "rss",
        "hybrid_rss",
    ] = "hybrid_uss"
    """
    Memory use measure to use for chunking.

    The following methods are supported to calculate memory overhead when chunking
    is enabled:

    * "bytes"
        expected rowsize based on actual size (as reported by numpy and
        pandas) of explicitly allocated data this can underestimate overhead due
        to transient data requirements of operations (e.g. merge, sort, transpose).
    * "uss"
        expected rowsize based on change in (unique set size) (uss) both as
        a result of explicit data allocation, and readings by MemMonitor sniffer
        thread that measures transient uss during time-consuming numpy and pandas
        operations.
    * "hybrid_uss"
        hybrid_uss avoids problems with pure uss, especially with
        small chunk sizes (e.g. initial training chunks) as numpy may recycle
        cached blocks and show no increase in uss even though data was allocated
        and logged.
    * "rss"
        like uss, but for resident set size (rss), which is the portion of
        memory occupied by a process that is held in RAM.
    * "hybrid_rss"
        like hybrid_uss, but for rss

    RSS is reported by :py:meth:`psutil.Process.memory_info` and USS is reported by
    :py:meth:`psutil.Process.memory_full_info`.  USS is the memory which is private to
    a process and which would be freed if the process were terminated.  This is
    the metric that most closely matches the rather vague notion of memory
    "in use" (the meaning of which is difficult to pin down in operating systems
    with virtual memory where memory can (but sometimes can't) be swapped or
    mapped to disk. Previous testing found `hybrid_uss` performs best and is most
    reliable and is therefore the default.

    For more, see :ref:`chunk_size`.
    """

    keep_chunk_logs: bool = True
    """
    Whether to keep chunk logs when deleting other files.
    """

    default_initial_rows_per_chunk: int = 100
    """
    Default number of rows to use in initial chunking.
    """

    min_available_chunk_ratio: float = 0.05
    """
    minimum fraction of total chunk_size to reserve for adaptive chunking
    """

    checkpoints: Union[bool, list] = True
    """
    When to write checkpoint (intermediate table states) to disk.

    If True, checkpoints are written at each step. If False, no intermediate
    checkpoints will be written before the end of run.  Or, provide an explicit
    list of models to checkpoint.
    """

    checkpoint_format: Literal["hdf", "parquet"] = "parquet"
    """
    Storage format to use when saving checkpoint files.
    """

    check_for_variability: bool = False
    """
    Debugging feature to find broken model specifications.

    Enabling this check does not alter valid results but slows down model runs.
    """

    log_alt_losers: bool = False
    """
    Write out expressions when all alternatives are unavailable.

    This can be useful for model development to catch errors in specifications.
    Enabling this check does not alter valid results but slows down model runs.
    """

    use_shadow_pricing: bool = False
    """turn shadow_pricing on and off for work and school location"""

    output_tables: OutputTables = None
    """list of output tables to write to CSV or HDF5"""

    want_dest_choice_sample_tables: bool = False
    """turn writing of sample_tables on and off for all models"""

    cleanup_pipeline_after_run: bool = False
    """
    Cleans up pipeline after successful run.

    This will clean up pipeline only after successful runs, by creating a
    single-checkpoint pipeline file, and deleting any subprocess pipelines.
    """

    cleanup_trace_files_on_resume: bool = False
    """Clean all trace files when restarting a model from a checkpoint."""

    sharrow: Union[bool, str] = False
    """
    Set the sharrow operating mode.

    .. versionadded:: 1.2

    * `false` - Do not use sharrow.  This is the default if no value is given.
    * `true` - Use sharrow optimizations when possible, but fall back to
      legacy `pandas.eval` systems when any error is encountered.  This is the
      preferred mode for running with sharrow if reliability is more important
      than performance.
    * `require` - Use sharrow optimizations, and raise an error if they fail
      unexpectedly.  This is the preferred mode for running with sharrow
      if performance is a concern.
    * `test` - Run every relevant calculation using both sharrow and legacy
      systems, and compare them to ensure the results match.  This is the slowest
      mode of operation, but useful for development and debugging.
    """

    disable_zarr: bool = False
    """
    Disable the use of zarr format skims.

    .. versionadded:: 1.2

    By default, if sharrow is enabled (any setting other than false), ActivitySim
    currently loads data from zarr format skims if a zarr location is provided,
    and data is found there.  If no data is found there, then original OMX skim
    data is loaded, any transformations or encodings are applied, and then this
    data is written out to a zarr file at that location.  Setting this option to
    True will disable the use of zarr.
    """

    store_skims_in_shm: bool = True
    """
    Store skim dataset in shared memory.

    .. versionadded:: 1.3

    By default, if sharrow is enabled (any setting other than false), ActivitySim
    stores the skim dataset in shared memory. This can be changed by setting this
    option to False, in which case skims are stores in "typical" process-local
    memory. Note that storing skims in shared memory is pretty much required for
    multiprocessing, unless you have a very small model or an absurdly large amount
    of RAM.
    """

    @model_validator(mode="after")
    def _check_store_skims_in_shm(self):
        if not self.store_skims_in_shm and self.multiprocess:
            raise ValueError("store_skims_in_shm requires multiprocess to be False")
        return self

    instrument: bool = False
    """
    Use `pyinstrument` to profile component performance.

    .. versionadded:: 1.2

    This is generally a developer-only feature and not needed for regular usage
    of ActivitySim.

    Use of this setting to enable statistical profiling of ActivitySim code,
    using the `pyinstrument` library (an optional dependency which must also be
    installed).  A separate profiling session is triggered for each model
    component. See the pyinstrument
    `documentation <https://pyinstrument.readthedocs.io/en/latest/how-it-works.html>`__
    for a description of how this tool works.

    When activated, a "profiling--\\*" directory is created in the output directory
    of the model, tagged with the date and time of the profiling run.  Profile
    output is always tagged like this and never overwrites previous profiling
    outputs, facilitating serial comparisons of runtimes in response to code or
    configuration changes.
    """

    memory_profile: bool = False
    """
    Generate a memory profile by sampling memory usage from a secondary process.

    .. versionadded:: 1.2

    This is generally a developer-only feature and not needed for regular usage
    of ActivitySim.

    Using this feature will open a secondary process, whose only job is to poll
    memory usage for the main ActivitySim process.  The usage is logged to a file
    with time stamps, so it can be cross-referenced against ActivitySim logs to
    identify what parts of the code are using RAM.  The profiling is done from
    a separate process to avoid the profiler itself from significantly slowing
    the main model core, or (more importantly) generating memory usage on its
    own that pollutes the collected data.
    """

    benchmarking: bool = False
    """
    Flag this model run as a benchmarking run.

    .. versionadded:: 1.1

    This is generally a developer-only feature and not needed for regular usage
    of ActivitySim.

    By flagging a model run as a benchmark, certain operations of the model are
    altered, to ensure valid benchmark readings.  For example, in regular
    operation, data such as skims are loaded on-demand within the first model
    component that needs them.  With benchmarking enabled, all data are always
    pre-loaded before any component is run, to ensure that recorded times are
    the runtime of the component itself, and not data I/O operations that are
    neither integral to that component nor necessarily stable over replication.
    """

    write_raw_tables: bool = False
    """
    Dump input tables back to disk immediately after loading them.

    This is generally a developer-only feature and not needed for regular usage
    of ActivitySim.

    The data tables are written out to `<output_dir>/raw_tables` before any
    annotation steps, but after initial processing (renaming, filtering columns,
    recoding).
    """

    disable_destination_sampling: bool = False

    want_dest_choice_presampling: bool = True

    testing_fail_trip_destination: bool = False

    fail_fast: bool = False

    rotate_logs: bool = False

    offset_preprocessing: bool = False
    """
    Flag to indicate whether offset preprocessing has already been done.

    .. versionadded:: 1.2

    This flag is generally set automatically within ActivitySim during a run,
    and not be a user ahead of time.  The ability to do so is provided as a
    developer-only feature for testing and development.
    """

    recode_pipeline_columns: bool = False
    """
    Apply recoding instructions on input and final output for pipeline tables.

    .. versionadded:: 1.2

    Recoding instructions can be provided in individual
    :py:attr:`InputTable.recode_columns` and :py:attr:`OutputTable.decode_columns`
    settings. This global setting permits disabling all recoding processes
    simultaneously.

    .. warning::

        Disabling recoding is fine in legacy mode but it is generally not
        compatible with using :py:attr:`Settings.sharrow`.
    """

    omx_ignore_patterns: list[str] = []
    """
    List of regex patterns to ignore when reading OMX files.

    This is useful if you have tables in your OMX file that you don't want to
    read in.  For example, if you have both time-of-day values and time-independent
    values (e.g., "BIKE_TIME" and "BIKE_TIME__AM"), you can ignore the time-of-day
    values by setting this to ["BIKE_TIME__.+"].

    .. versionadded:: 1.3
    """

    keep_mem_logs: bool = False

    pipeline_complib: str = "NOTSET"
    """
    Compression library to use when storing pipeline tables in an HDF5 file.

    .. versionadded:: 1.3
    """

    treat_warnings_as_errors: bool = False
    """
    Treat most warnings as errors.

    Use of this setting is not recommended outside of rigorous testing regimes.

    .. versionadded:: 1.3
    """

    log_settings: tuple[str] = (
        "households_sample_size",
        "chunk_size",
        "chunk_method",
        "chunk_training_mode",
        "multiprocess",
        "num_processes",
        "resume_after",
        "trace_hh_id",
        "memory_profile",
        "instrument",
        "sharrow",
    )
    """
    Setting to log on startup.
    """

    hh_ids: Path = None
    """
    Load only the household ids given in this file.

    The file need only contain the desired households ids, nothing else.
    If given as a relative path (or just a file name), both the data and
    config directories are searched, in that order, for the matching file.
    """

    source_file_paths: list[Path] = None
    """
    A list of source files from which these settings were loaded.

    This value should not be set by the user within the YAML settings files,
    instead it is populated as those files are loaded.  It is primarily
    provided for debugging purposes, and does not actually affect the operation
    of the model.
    """

    inherit_settings: Union[bool, Path] = None
    """
    Instruction on if and how to find other files that can provide settings.

    When this value is True, all config directories are searched in order for
    additional files with the same filename.  If other files are found they
    are also loaded, but only settings values that are not already explicitly
    set are applied.  Alternatively, set this to a different file name, in which
    case settings from that other file are loaded (again, backfilling unset
    values only).  Once the settings files are loaded, this value does not
    have any other effect on the operation of the model(s).
    """

    rng_base_seed: Union[int, None] = 0
    """Base seed for pseudo-random number generator."""

    duplicate_step_execution: Literal["error", "allow"] = "error"
    """
    How activitysim should handle attempts to re-run a step with the same name.

    .. versionadded:: 1.3

    * "error"
        Attempts to re-run a step that has already been run and
        checkpointed will raise a `RuntimeError`, halting model execution.
        This is the default if no value is given.
    * "allow"
        Attempts to re-run a step are allowed, potentially overwriting
        the results from the previous time that step was run.
    """

    downcast_int: bool = False
    """
    automatically downcasting integer variables.

    Use of this setting should be tested by the region to confirm result consistency.

    .. versionadded:: 1.3
    """

    downcast_float: bool = False
    """
    automatically downcasting float variables.

    Use of this setting should be tested by the region to confirm result consistency.

    .. versionadded:: 1.3
    """

    other_settings: dict[str, Any] = None

    def _get_attr(self, attr):
        try:
            return getattr(self, attr)
        except AttributeError:
            return self.other_settings.get(attr)
