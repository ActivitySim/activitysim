from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypeVar, Union  # noqa: F401

from pydantic import BaseModel as PydanticBase
from pydantic import validator

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
    definitions for more nests, or a mixture of these.
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
    """Nesting structure for a nested logit model.

    The nesting structure is specified heirarchically from the top, so the
    value of this field should be the "root" level nest of the nested logit
    tree, which should contain references to lower level nests and/or the
    actual alternatives.

    For example, this YAML defines a simple nesting structure for four
    alternatives (DRIVE, WALK, WALK_TO_TRANSIT, DRIVE_TO_TRANSIT) with the two
    transit alternatives grouped together in a nest:

    .. code-block:: yaml

        NESTS:
          name: root
          coefficient: coef_nest_root
          alternatives:
            - DRIVE
            - WALK
            - name: TRANSIT
              coefficient: coef_nest_transit
              alternatives:
              - WALK_TO_TRANSIT
              - DRIVE_TO_TRANSIT
    """

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
