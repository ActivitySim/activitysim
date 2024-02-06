from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

from pydantic import PositiveInt, root_validator

from activitysim.core.configuration.base import (
    Any,
    PydanticBase,
    PydanticReadable,
    Union,
)


class DigitalEncoding(PydanticBase):
    """Digital encoding instructions for skim tables.

    These encoding instructions are used to digitally encode data prior
    to writing that data back to disk in the `zarr` file format.

    See :ref:`digital-encoding` documentation for details.

    .. versionadded:: 1.2
    """

    name: str = None
    """The name of an individual matrix skim to encode.

    Use this setting to encode specific individual skims.  To encode
    a group of related skims with the same encoding, or together with
    a joint encoding, use `regex` instead.  You cannot specify both
    `name` and `regex` at the same time.
    """

    regex: str = None
    """A regular expression for matching skim matrix names.

    All skims with names that match under typical regular expression rules
    for Python will be processed using the rules defined in this
    DigitalEncoding instruction.  To encode one specific skim,
    use `name` instead. You cannot specify both `name` and `regex` at
    the same time.
    """

    joint_dict: str = None
    """The name of the joint dictionary for this group.

    This must be a unique name for this set of skims, and a new array
    will be added to the Dataset with this name.  It will be an integer-
    type array indicating the position of each element in the jointly
    encoded dictionary.

    If the `joint_dict` name is given, then all other instructions in this
    DigitalEncoding are ignored, except `regex`.
    """

    missing_value: Any = None
    """
    Use this value to indicate "missing" values.

    For float variables, it is possible to use NaN to represent missing values,
    but other data types do not have a native missing value, so it will need
    to be given explicitly.
    """

    bitwidth: int = 16
    """Number of bits to use in encoded integers, either 8, 16 or 32.

    For basic fixed point encoding, it is usually sufficient to simply define
    the target bitwidth, and the `missing value` if applicable.  The other
    necessary parameters can then be inferred from the data.
    """

    min_value: Union[int, float] = None
    """
    Explicitly give the minimum value represented in the array.
    If not given, it is inferred from the data. It is useful to give
    these values if the array does not necessarily include all the values that
    might need to be inserted later.
    """

    max_value: Union[int, float] = None
    """
    Explicitly give the maximum value represented in the array.
    If not given, it is inferred from the data. It is useful to give
    these values if the array does not necessarily include all the values that
    might need to be inserted later.
    """

    scale: Union[int, float] = None
    """
    An explicitly defined scaling factor.

    The scaling factor can be inferred from the min and max values if not provided.
    """

    offset: Union[int, float] = None
    """
    An explicitly defined offset factor.

    The offset factor can be inferred from the min and max values if not provided.
    """

    by_dict: Union[int, bool] = None
    """
    Encode by dictionary, using a bitwidth from {8, 16, 32}, or `True`.

    If given, all arguments other settings for this data are ignored. If given
    as `True`, the bitwidth setting is not ignored, but everything else still
    is.
    """


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

    zarr_digital_encoding: list[DigitalEncoding] = None
    """
    A list of encodings to apply before saving skims in ZARR format.

    .. versionadded:: 1.2

    Digital encodings transform how data is stored in memory and on disk,
    potentially reducing storage requirements without fundamentally changing
    the underlying data.
    """


class MazToMazSettings(PydanticBase, extra="forbid"):
    tables: list[str] = []

    max_blend_distance: dict[str, float] = None

    blend_distance_skim_name: str | None = None
    """The name of the skim table used to blend distances for MAZs."""


class TimeSettings(PydanticReadable, extra="forbid"):
    """
    Settings to describe discrete time.
    """

    time_window: PositiveInt = 1440
    """total duration (in minutes) of the modeled time span."""

    period_minutes: PositiveInt = 60
    """length of time (in minutes) each model time period represents.

    Must be whole factor of ``time_window``."""

    periods: list[int]
    """Breakpoints that define the aggregate periods for skims and assignment.

    The first value should be zero and the last value should equal `time_window`
    divided by `period_minutes`.  The intervals between these various values
    represent the skimmed time periods, so this list should be one longer than
    that of `labels`.
    """

    labels: list[str]
    """Labels to define names for aggregate periods for skims and assignment"""

    @root_validator(pre=True)
    def hours_deprecated(cls, data):
        if "hours" in data:
            data["periods"] = data.pop("hours")
            warnings.warn(
                "support for `skim_time_periods` key `hours` will be removed in "
                "future verions. Use `periods` instead",
                FutureWarning,
                stacklevel=2,
            )
        return data


class NetworkSettings(PydanticReadable, extra="forbid"):
    """
    Network level of service and skims settings

    The input for these settings is typically stored in one YAML file,
    usually called ``network_los.yaml``.
    """

    name: str = None
    """Name of this network, not used for anything?"""

    zone_system: int
    """Which zone system type is used.

    * 1 - TAZ only.
    * 2 - MAZ and TAZ.
    * 3 - MAZ, TAZ, and TAP
    """

    taz_skims: Union[str, list[str], TAZ_Settings] = None
    """Instructions for how to load and pre-process skim matrices.

    If given as a string or a list of strings, it is interpreted as the location
    for OMX file(s), either as a single file or as a glob-matching pattern for
    multiple files. The time period for the matrix must be represented at the end
    of the matrix name and be seperated by a double_underscore (e.g. `BUS_IVT__AM`
    indicates base skim BUS_IVT with a time period of AM.

    Alternatively, this can be given as a nested dictionary defined via the
    TAZ_Settings class, which allows for ZARR transformation and pre-processing.
    """

    skim_time_periods: TimeSettings
    """How to discretize time in this model."""

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

    network_cache_dir: str = None
    """alternate dir to read/write cache files (defaults to output_dir)"""

    #### 2 ZONE ####

    maz: str = None
    """Filename for the MAZ data file.

    This file should contain the MAZ ID, TAZ, and land use and other MAZ attributes
    """

    maz_to_maz: MazToMazSettings | None = None
    """Settings to manage maz-to-maz level of service in 2- and 3-zone models."""

    #### 3 ZONE ####

    tap: str = None
    """Filename for the TAP data file.

    This file should contain the MAZ ID, TAZ, and land use and other MAZ attributes
    """

    maz_to_tap: dict[str, Any] = None
    """Settings to manage maz-to-tap level of service in 3-zone models."""

    demographic_segments: Any = None

    tap_skims: Union[str, list[str]] = None

    tap_lines: str = None
    """TAP lines filename."""

    TVPB_SETTINGS: Any = None

    rebuild_tvpb_cache: bool = True
    """
    rebuild and overwrite existing pre-computed TAP to TAP utilities cache
    """

    trace_tvpb_cache_as_csv: bool = False
    """Write a CSV version of TVPB cache for tracing

    Not currently implemented."""

    skim_dict_factory: Literal[
        "NumpyArraySkimFactory",
        "MemMapSkimFactory",
    ] = "NumpyArraySkimFactory"
    """The skim dict factory to use.

    The MemMapSkimFactory is strictly experimental.
    """

    source_file_paths: list[Path] = None
    """
    A list of source files from which these settings were loaded.

    This value should not be set by the user within the YAML settings files,
    instead it is populated as those files are loaded.  It is primarily
    provided for debugging purposes, and does not actually affect the operation
    of the model.
    """
