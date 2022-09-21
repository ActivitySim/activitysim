from typing import Union

try:
    from pydantic import BaseModel as PydanticBase
except ModuleNotFoundError:

    class PydanticBase:
        pass


class InputTable(PydanticBase):
    """
    The features that define an input table to be read by ActivitySim.
    """

    tablename: str
    """Name of the injected table"""

    filename: str = None
    """
    Name of the CSV or HDF5 file to read.

    If not provided, defaults to `input_store`
    """

    index_col: str = None
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

    h5_tablename: str = None
    """table name if reading from HDF5 and different from `tablename`"""


class OutputTable(PydanticBase):
    tablename: str
    decode_columns: dict[str, str] = None


class OutputTables(PydanticBase):
    h5_store: bool = False
    action: str
    prefix: str
    tables: list[Union[str, OutputTable]]


class MultiprocessStepSlice(PydanticBase):
    """Instructions on how to slice tables for each subprocess."""

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

    exclude: Union[bool, str, list[str]]
    """
    Optional list of tables not to slice even if they have a sliceable index name.

    Or set to `True` or "*" to exclude all tables not explicitly listed in
    `tables`.
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


class Settings(PydanticBase):
    """
    The overall settings for the ActivitySim model system.

    The input for these settings is typically stored in one main YAML file,
    usually called ``settings.yaml``.

    Note that this implementation is presently used only for generating
    documentation, but future work may migrate the settings implementation to
    actually use this pydantic code to validate the settings before running
    the model.
    """

    models: list[str]
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

    multiprocess_steps: list[MultiprocessStep]
    """A list of multiprocess steps."""

    resume_after: str = None
    """to resume running the data pipeline after the last successful checkpoint"""

    input_table_list: list[InputTable]
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
    trace_hh_id: Union[int, list] = None
    """
    Trace household id(s)

    If omitted, no tracing is written out
    """

    trace_od: list[int] = None
    """
    Trace origin, destination pair in accessibility calculation

    If omitted, no tracing is written out.
    """

    chunk_training_mode: str = None
    """
    The method to use for chunk training.

    Valid values include {disabled, training, production, adaptive}.
    See :ref:`chunk_size` for more details.
    """

    chunk_size: int = None
    """
    Approximate amount of RAM to allocate to ActivitySim for batch processing.

    See :ref:`chunk_size` for more details.
    """

    chunk_method: str = None
    """
    Memory use measure to use for chunking.

    See :ref:`chunk_size`.
    """

    checkpoints: Union[bool, list] = True
    """
    When to write checkpoint (intermediate table states) to disk.

    If True, checkpoints are written at each step. If False, no intermediate
    checkpoints will be written before the end of run.  Or, provide an explicit
    list of models to checkpoint.
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

    output_tables: list[str] = None
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

    The data tables are written out before any annotation steps, but after
    initial processing (renaming, filtering columns, recoding).
    """

    disable_destination_sampling: bool = False

    want_dest_choice_presampling: bool = False

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

    recode_pipeline_columns: bool = True
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

    keep_mem_logs: bool = False


class ZarrDigitalEncoding(PydanticBase):
    """Digital encoding instructions for skim tables.

    .. versionadded:: 1.2
    """

    regex: str
    """A regular expression for matching skim matrix names.

    All skims with names that match under typical regular expression rules
    for Python will be processed together.
    """

    joint_dict: str
    """The name of the joint dictionary for this group.

    This must be a unique name for this set of skims, and a new array
    will be added to the Dataset with this name.  It will be an integer-
    type array indicating the position of each element in the jointly
    encoded dictionary."""


class TAZ_Settings(PydanticBase):
    """
    Complex settings for TAZ skims that are not just OMX file(s).

    .. versionadded:: 1.2
    """

    omx: str = None
    """The filename of the data stored in OMX format.

    This is treated as a fallback for the raw input data, if ZARR format data
    is not available.
    """

    zarr: str = None
    """The filename of the data stored in ZARR format.

    Reading ZARR data can be much faster than reading OMX format data, so if
    this filename is given, the ZARR file format is preferred if it exists. If
    it does not exist, then OMX data is read in and then ZARR data is written
    out for future usage.

    .. versionadded:: 1.2
    """

    zarr_digital_encoding: list[ZarrDigitalEncoding] = None
    """
    A list of encodings to apply before saving skims in ZARR format.

    .. versionadded:: 1.2

    Digital encodings transform how data is stored in memory and on disk,
    potentially reducing storage requirements without fundamentally changing
    the underlying data.
    """


class NetworkSettings(PydanticBase):
    """
    Network level of service and skims settings

    The input for these settings is typically stored in one YAML file,
    usually called ``network_los.yaml``.
    """

    zone_system: int
    """Which zone system type is used.

    * 1 - TAZ only.
    * 2 - MAZ and TAZ.
    * 3 - MAZ, TAZ, and TAP
    """

    taz_skims: Union[str, TAZ_Settings] = None
    """Instructions for how to load and pre-process skim matrices.

    If given as a string, it is interpreted as the location for OMX file(s),
    either as a single file or as a glob-matching pattern for multiple files.
    The time period for the matrix must be represented at the end of the matrix
    name and be seperated by a double_underscore (e.g. `BUS_IVT__AM` indicates base
    skim BUS_IVT with a time period of AM.

    Alternatively, this can be given as a nested dictionary defined via the
    TAZ_Settings class, which allows for ZARR transformation and pre-processing.
    """

    skim_time_periods: dict
    """time period upper bound values and labels

    * ``time_window`` - total duration (in minutes) of the modeled time span (Default: 1440 minutes (24 hours))
    * ``period_minutes`` - length of time (in minutes) each model time period represents. Must be whole factor of ``time_window``. (Default: 60 minutes)
    * ``periods`` - Breakpoints that define the aggregate periods for skims and assignment
    * ``labels`` - Labels to define names for aggregate periods for skims and assignment
    """

    read_skim_cache: bool = False
    """Read cached skims (using numpy memmap) from output directory.

    Reading from memmap is much faster than omx, but the memmap is a huge
    uncompressed file.
    """

    write_skim_cache: bool = False
    """Write memmapped cached skims to output directory.

    This is needed if you want to use the cached skims to speed up subsequent
    runs.
    """

    cache_dir: str = None
    """alternate dir to read/write cache files (defaults to output_dir)"""
