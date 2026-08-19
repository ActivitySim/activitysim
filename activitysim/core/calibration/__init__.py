# ActivitySim
# See full license in LICENSE.txt.
"""Automated calibration support for ActivitySim model runs.

The public API is intentionally small. Private attributes from the former
single-file module remain available through ``__getattr__`` for compatibility
while callers migrate to the package modules.
"""

from . import (
    coefficients,
    component,
    execution,
    expressions,
    multiprocess,
    recovery,
    reporting,
    settings,
)
from .orchestrator import run_calibration_loop
from .settings import (
    CalibrationComponentResult,
    CalibrationComponentSettings,
    CalibrationConfig,
    CalibrationReportsSettings,
    CalibrationRunResult,
    CalibrationRunSettings,
    calibration_enabled,
    read_calibration_settings,
)

__all__ = [
    "CalibrationComponentResult",
    "CalibrationComponentSettings",
    "CalibrationConfig",
    "CalibrationReportsSettings",
    "CalibrationRunResult",
    "CalibrationRunSettings",
    "calibration_enabled",
    "read_calibration_settings",
    "run_calibration_loop",
]

_COMPATIBILITY_MODULES = (
    settings,
    component,
    expressions,
    coefficients,
    reporting,
    recovery,
    execution,
    multiprocess,
)


def __getattr__(name):
    for module in _COMPATIBILITY_MODULES:
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    compatibility_names = {
        name
        for module in _COMPATIBILITY_MODULES
        for name in vars(module)
        if not name.startswith("__")
    }
    return sorted(set(globals()) | compatibility_names)
