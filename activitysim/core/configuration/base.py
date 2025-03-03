from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal, TypeVar, Union  # noqa: F401

import pandas as pd
from pydantic import BaseModel as PydanticBase

from activitysim.core import configuration

PydanticReadableType = TypeVar("PydanticReadableType", bound="PydanticReadable")


class PydanticReadable(PydanticBase):
    """
    Base class for `pydantic.BaseModel`s readable from cascading config files.

    Although not formally defined as an abstract base class, there is generally
    no reason to instantiate a `PydanticReadable` object directly.
    """

    source_file_paths: list[Path] = None
    """
    A list of source files from which these settings were loaded.

    This value should not be set by the user within the YAML settings files,
    instead it is populated as those files are loaded.  It is primarily
    provided for debugging purposes, and does not actually affect the operation
    of any model.
    """

    @classmethod
    def read_settings_file(
        cls: type[PydanticReadableType],
        filesystem: configuration.FileSystem,
        file_name: str,
        mandatory: bool = True,
    ) -> PydanticReadableType:
        """
        Load settings from one or more yaml files.

        This method has been written to allow models to be configured with
        "settings file inheritance". This allows the user to avoid duplicating
        settings across multiple related model configurations.  Instead,
        settings can be written in a "cascading" manner: multiple files can be
        provided with settings values, and each particular key is set according
        to the first value found for that key.

        For example, suppose a user has a basic model setup with some settings, and
        they would like to do a model run with all the same settings except with the
        `foo` setting using a value of `'baz'` instead of the usual value of `'bar'`
        that is defined in the usual model setup.  They could accomplish this by
        placing a `file_name` file with *only*

        .. code-block:: yaml

            foo: baz
            inherit_settings: true

        in the first directory listed in `filesystem.configs_dir`. The
        `inherit_settings` flag tells the interpreter to search for other
        matching settings files in the chain of config directories, and to fill
        in other settings values that are not yet defined, but the `foo: baz` will
        preempt any other values for `foo` that may be set in those other files.
        If the `inherit_settings` flag is omitted or set to false, then the
        search process ends with this file, only the `foo` setting would be
        defined, and all other settings expected in this file would take on
        their default values.

        Alternatively, a settings file may include a `include_settings` key,

        .. code-block:: yaml

            include_settings: other-filename.yaml

        with an alternative file name as its value, in which case the method
        loads values from that other file instead. To avoid confusion, this
        directive must appear ALONE in the target file, without any additional
        settings or directives.

        Parameters
        ----------
        filesystem: configuration.FileSystem
            Provides the list of config directories to search.
        file_name : str
            The name of the YAML file to search for.
        mandatory : boolean, default True
            If true, raise SettingsFileNotFoundError if no matching settings file
            is found in any config directory, otherwise this method will return
            an empty dict or an all-default instance of the validator class.

        Returns
        -------
        PydanticReadable or derived class
        """
        # pass through to read_settings_file, requires validator_class and provides type hinting for IDE's
        return filesystem.read_settings_file(
            file_name,
            mandatory,
            validator_class=cls,
        )


class PreprocessorSettings(PydanticBase):
    """
    Preprocessor instructions.
    """

    SPEC: str
    """Specification to use for pre-processing.

    This is the name of the specification CSV file to be found in one of the
    configs directories.  The '.csv' extension may be omitted.
    """

    DF: str
    """Name of the primary table used for this preprocessor.

    The preprocessor will emit rows to a temporary table that match the rows
    in this table from the pipeline."""

    TABLES: list[str] | None = None
    """Names of the additional tables to be merged for the preprocessor.

    Data from these tables will be merged into the primary table, according
    to standard merge rules for these tables.  Care should be taken to limit the
    number of merged tables as the memory requirements for the preprocessor
    will increase with each table.
    """


class ComputeSettings(PydanticBase):
    """
    Sharrow settings for a component.
    """

    sharrow_skip: bool | dict[str, bool] = False
    """Skip sharrow when evaluating this component.

    This overrides the global sharrow setting, and is useful if you want to skip
    sharrow for particular components, either because their specifications are
    not compatible with sharrow or if the sharrow performance is known to be
    poor on this component.

    When a component has multiple subcomponents, the `sharrow_skip` setting can be
    a dictionary that maps the names of the subcomponents to boolean values.
    For example, to skip sharrow for an OUTBOUND and OUTBOUND_COND subcomponent
    but not the INBOUND subcomponent, use the following setting:

    ```yaml
    sharrow_skip:
        OUTBOUND: true
        INBOUND: false
        OUTBOUND_COND: true
    ```

    Alternatively, even for components with multiple subcomponents, the `sharrow_skip`
    value can be a single boolean true or false, which will be used for all
    subcomponents.

    """

    fastmath: bool = True
    """Use fastmath when evaluating this component with sharrow.

    The fastmath option can be used to speed up the evaluation of expressions in
    this component's spec files, but it does so by making some simplifying
    assumptions about the math, e.g. that neither inputs nor outputs of any
    computations are NaN or Inf.  This can lead to errors when the assumptions
    are violated.  If running in sharrow test mode generates errors, try turning
    this setting off.
    """

    use_bottleneck: bool | None = None
    """Use the bottleneck library with pandas.eval.

    Set to True or False to force the use of bottleneck or not. If set to None,
    the current pandas option setting of `compute.use_bottleneck` will be used.

    See https://pandas.pydata.org/docs/reference/api/pandas.set_option.html
    for more information."""

    use_numexpr: bool | None = None
    """Use the numexpr library with pandas.eval.

    Set to True or False to force the use of numexpr or not. If set to None,
    the current pandas option setting of `compute.use_numexpr` will be used.

    See https://pandas.pydata.org/docs/reference/api/pandas.set_option.html
    for more information.
    """

    use_numba: bool | None = None
    """Use the numba library with pandas.eval.

    Set to True or False to force the use of numba or not. If set to None,
    the current pandas option setting of `compute.use_numba` will be used.

    See https://pandas.pydata.org/docs/reference/api/pandas.set_option.html
    for more information.
    """

    drop_unused_columns: bool = True
    """Drop unused columns in the choosers df.

    Set to True or False to drop unused columns in data table for specific component.
    Default to True. If set to False, all columns in the data table will be kept.
    """

    protect_columns: list[str] = []
    """Protect these columns from being dropped from the chooser table."""

    def should_skip(self, subcomponent: str) -> bool:
        """Check if sharrow should be skipped for a particular subcomponent."""
        if isinstance(self.sharrow_skip, dict):
            return self.sharrow_skip.get(subcomponent, False)
        else:
            return bool(self.sharrow_skip)

    @contextmanager
    def pandas_option_context(self):
        """Context manager to set pandas options for compute settings."""
        args = ()
        if self.use_bottleneck is not None:
            args += ("compute.use_bottleneck", self.use_bottleneck)
        if self.use_numexpr is not None:
            args += ("compute.use_numexpr", self.use_numexpr)
        if self.use_numba is not None:
            args += ("compute.use_numba", self.use_numba)
        if args:
            with pd.option_context(*args):
                yield
        else:
            yield

    def subcomponent_settings(self, subcomponent: str) -> ComputeSettings:
        """Get the sharrow settings for a particular subcomponent."""
        return ComputeSettings(
            sharrow_skip=self.should_skip(subcomponent),
            fastmath=self.fastmath,
            use_bottleneck=self.use_bottleneck,
            use_numexpr=self.use_numexpr,
            use_numba=self.use_numba,
            drop_unused_columns=self.drop_unused_columns,
            protect_columns=self.protect_columns,
        )


class PydanticCompute(PydanticReadable):
    """Base class for component settings that include optional sharrow controls."""

    compute_settings: ComputeSettings = ComputeSettings()
    """Sharrow settings for this component."""
