from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel as PydanticBase
from pydantic import validator

from activitysim.core.configuration.base import PydanticReadable


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


class _BaseLogitComponentSettings(PydanticReadable):
    """
    Base configuration class for components that are logit models.

    These settings are common to all logit models. Component developers
    should generally prefer using a derived classes that defines a complete
    logit model such as `LogitComponentSettings`, or a compound component
    such as `LocationComponentSettings`, which melds together alternative
    sampling, logsums, and choice.
    """

    SPEC: Path
    """Utility specification filename.

    This is sometimes alternatively called the utility expressions calculator
    (UEC). It is a CSV file giving all the functions for the terms of a
    linear-in-parameters utility expression.
    """

    COEFFICIENTS: Path
    """Coefficients filename.

    This is a CSV file giving named parameters for use in the utility expression.
    """

    CONSTANTS: dict[str, Any] | None = None
    """Named constants usable in the utility expressions."""


class LogitComponentSettings(_BaseLogitComponentSettings):
    """
    Base configuration class for components that are individual logit models.
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


class TemplatedLogitComponentSettings(LogitComponentSettings):
    """
    Base configuration for segmented logit models with a coefficient template.
    """

    COEFFICIENT_TEMPLATE: str | None = None
    """Coefficients template filename.

    For a segmented model component, this maps the named parameters to
    segment-specific names.
    """


class LocationComponentSettings(_BaseLogitComponentSettings):
    """
    Base configuration class for components that are location choice models.
    """

    SAMPLE_SPEC: Path
    """The utility spec giving expressions to use in alternative sampling."""

    SAMPLE_SIZE: int
    """This many candidate alternatives will be sampled for each choice."""

    LOGSUM_SETTINGS: Path
    """Settings for the logsum computation."""


class TourLocationComponentSettings(LocationComponentSettings):

    # Logsum-related settings
    CHOOSER_ORIG_COL_NAME: str
    ALT_DEST_COL_NAME: str
    IN_PERIOD: int | dict[str, int]
    OUT_PERIOD: int | dict[str, int]
    LOGSUM_PREPROCESSOR: str = "preprocessor"
