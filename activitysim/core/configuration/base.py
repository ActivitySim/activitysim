from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypeVar, Union  # noqa: F401

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
    TABLES: list[str] | None


class MandatoryTourFrequencySettings(PydanticReadable):
    """
    Settings for the `mandatory_tour_frequency` component.
    """

    preprocessor: PreprocessorSettings | None = None
    """Setting for the preprocessor."""

    SPEC: str = "mandatory_tour_frequency.csv"
    """Filename for the accessibility specification (csv) file."""

    COEFFICIENTS: str = "mandatory_tour_frequency_coefficients.csv"
    """Filename for the mandatory tour frequency coefficients (csv) file."""

    annotate_persons: dict[str, Any] = {}
