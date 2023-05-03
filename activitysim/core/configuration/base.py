from __future__ import annotations

from typing import Any, Literal, TypeVar, Union  # noqa: F401

from pydantic import BaseModel as PydanticBase
from pydantic import validator

from activitysim.core import configuration

PydanticReadableType = TypeVar("PydanticReadableType", bound="PydanticReadable")


class PydanticReadable(PydanticBase):
    @classmethod
    def read_settings_file(
        cls: type[PydanticReadableType],
        filesystem: configuration.FileSystem,
        file_name: str,
        mandatory: bool = True,
    ) -> PydanticReadableType:
        """
        Load settings from one or more yaml files.

        This method will look for first occurrence of a yaml file named
        <file_name> in the directories in the `configs_dir` list of
        `filesystem`, and read settings from that yaml file.

        Settings file may contain directives that affect which file settings
        are returned:

        - inherit_settings (boolean)
            If found and set to true, this method will backfill settings
            in the current file with values from the next settings file
            in configs_dir list (if any)
        - include_settings: string <include_file_name>
            Read settings from specified include_file in place of the current
            file. To avoid confusion, this directive must appear ALONE in the
            target file, without any additional settings or directives.

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


class LogitNestSpec(PydanticBase):
    """
    Defines a nest in a nested logit model.
    """

    name: str
    """A descriptive name for this nest."""

    coefficient: str
    """The named parameter to be used as the logsum coefficient.

    This named parameter should appear in the logit models's `COEFFICIENTS`
    file.
    """

    alternatives: list[LogitNestSpec | str]
    """The alternatives within this nest.

    These can be either the names of elemental alternatives, or `LogitNestSpec`
    definitions for more nests.
    """


class LogitComponentSettings(PydanticBase):
    """
    Base configuration class for ActivitySim components that are logit models.
    """

    SPEC: str
    """Utility specification filename.

    This is sometimes alternatively called the utility expressions calculator
    (UEC). It is a CSV file giving all the functions for the terms of a
    linear-in-parameters utility expression.
    """

    COEFFICIENTS: str
    """Coefficients filename.

    This is a CSV file giving named parameters for use in the utility expression.
    """

    COEFFICIENT_TEMPLATE: str | None = None
    """Coefficients template filename.

    For a segmented model component, this maps the named parameters to
    segment-specific names.
    """

    LOGIT_TYPE: Literal["MNL", "NL"] = "MNL"
    """Logit model mathematical form.

    * "MNL"
        Multinomial logit model.
    * "NL"
        Nested multinomial logit model.
    """

    NESTS: LogitNestSpec | None = None

    @validator("NESTS")
    def nests_are_for_nl(cls, nests, values):
        """
        Checks that nests are provided if (and only if) `LOGIT_TYPE` is NL.
        """
        if "LOGIT_TYPE" in values and values["LOGIT_TYPE"] == "NL":
            if nests is None:
                raise ValueError("NESTS cannot be omitted for a NL model")
        if "LOGIT_TYPE" in values and values["LOGIT_TYPE"] == "MNL":
            if nests is not None:
                raise ValueError("NESTS cannot be provided for a MNL model")
        return nests

    CONSTANTS: dict[str, Any] | None = None
    """Named constants usable in the utility expressions."""
