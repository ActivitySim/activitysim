# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pydantic import model_validator

from activitysim.core import workflow
from activitysim.core.configuration import PydanticReadable
from activitysim.core.configuration.base import PydanticBase

CALIBRATION_SETTINGS_FILE_NAME = "calibration.yaml"


class CalibrationRunSettings(PydanticBase):
    """Run-control settings for calibration."""

    resume_after: Optional[str] = None
    calibrate_models: list[str]
    global_iterations: int = 1
    complete_steps: bool = False
    invalidate_tables: list[str] | None = None

    @model_validator(mode="after")
    def validate_run_settings(self):
        if not self.calibrate_models:
            raise ValueError(
                "calibration.run.calibrate_models must contain at least one model name"
            )
        return self

    """Tables to drop from state after each calibration restore so their
    ``@workflow.table`` factories regenerate from current data.

    Default (None): invalidates ``["vehicles"]``.  Set to ``[]`` to disable.

    A table should be listed here when ALL of the following are true:

    1. It is created by a ``@workflow.table`` factory from another table's
       values (not just from input data files).
    2. That source table is modified by a calibrated model or by a model
       whose outputs change when calibrated coefficients change.
    3. The factory uses source-table values to determine **row identity**
       (index values) or **row count**, not just column values.

    The canonical example is ``vehicles``: its factory repeats household
    rows by ``households["auto_ownership"]`` and derives ``vehicle_id``
    from ``household_id``.  When ``auto_ownership_simulate`` is calibrated,
    different coefficients produce different ownership counts, so the
    stale vehicles table loaded from a prior checkpoint would have the
    wrong number of rows and wrong vehicle IDs.  Dropping it forces the
    factory to regenerate vehicles consistent with the current households.

    Tables that only read *column values* from upstream tables (without
    affecting row identity) generally do NOT need invalidation — their
    content will be correct as long as the upstream table is correct at
    the restored checkpoint.
    """


class CalibrationReportsSettings(PydanticBase):
    """Reporting settings for a calibrated component."""

    generic: bool = True
    bespoke: str | None = None


class CalibrationComponentSettings(PydanticBase):
    """Settings for one calibratable model component."""

    calibration_spec: str
    helper_module: str | None = None
    submodel_max_iterations: int = 1
    reports: CalibrationReportsSettings = CalibrationReportsSettings()
    survey_file: Optional[str] = None


class CalibrationConfig(PydanticReadable):
    """Top-level calibration configuration."""

    enable: bool = False
    run: CalibrationRunSettings
    model_settings: dict[str, CalibrationComponentSettings] = {}

    @model_validator(mode="after")
    def validate_model_settings(self):
        """Validate that configured components are aligned with run settings."""
        for component in self.run.calibrate_models:
            if component not in self.model_settings:
                raise ValueError(
                    f"calibration model '{component}' is not in model_settings"
                )

        if self.run.global_iterations < 1:
            raise ValueError("max_iterations must be >= 1")

        return self


@dataclass
class CalibrationComponentResult:
    """Result details from calibrating one component."""

    component: str
    converged: bool
    component_iterations: int


@dataclass
class CalibrationRunResult:
    """Result details from a complete global calibration loop."""

    converged: bool
    completed_global_iterations: int


def read_calibration_settings(state: workflow.State) -> CalibrationConfig | None:
    """Read and validate calibration settings if calibration.yaml exists."""
    return CalibrationConfig.read_settings_file(
        state.filesystem,
        CALIBRATION_SETTINGS_FILE_NAME,
        mandatory=False,
    )


def calibration_enabled(state: workflow.State) -> bool:
    """Return True when calibration.yaml exists and is enabled."""
    settings = read_calibration_settings(state)
    return bool(settings and settings.enable)

