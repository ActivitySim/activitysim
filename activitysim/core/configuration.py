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
    """dictionary of column name mappings"""

    keep_columns: list[str] = None
    """
    columns to keep once read in to memory.

    Save only the columns needed for modeling or analysis to save on memory
    and file I/O
    """

    h5_tablename: str = None
    """table name if reading from HDF5 and different from `tablename`"""


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
