from __future__ import annotations

import logging
import warnings
from typing import Any, TypeVar

from activitysim.core import workflow
from activitysim.core.configuration.base import PydanticBase
from activitysim.core.configuration.logit import LogitComponentSettings

# ActivitySim
# See full license in LICENSE.txt.


logger = logging.getLogger(__name__)


@workflow.cached_object
def locutor(state: workflow.State) -> bool:
    # when multiprocessing, sometimes you only want one process to write trace files
    # mp_tasks overrides this definition to designate a single sub-process as locutor
    return True


def future_model_settings(model_name, model_settings, future_settings):
    """
    Warn users of new required model settings, and substitute default values

    Parameters
    ----------
    model_name: str
        name of model
    model_settings: dict
        model_settings from settigns file
    future_settings: dict
        default values for new required settings

    Returns
    -------
    dict
        model_settings with any missing future_settings added
    """
    model_settings = model_settings.copy()
    for key, setting in future_settings.items():
        if key not in model_settings.keys():
            warnings.warn(
                f"Setting '{key}' not found in {model_name} model settings."
                f"Replacing with default value: {setting}."
                f"This setting will be required in future versions",
                FutureWarning,
            )
            model_settings[key] = setting

    return model_settings


T = TypeVar("T", bound=PydanticBase)


def future_component_settings(
    model_name: str, model_settings: T, future_settings: dict
) -> T:
    """
    Warn users of new required model settings, and substitute default values

    Parameters
    ----------
    model_name: str
        name of model
    model_settings: PydanticBase
        model_settings from settigns file
    future_settings: dict
        default values for new required settings
    """
    for key, setting in future_settings.items():
        if getattr(model_settings, key) is None:
            warnings.warn(
                f"Setting '{key}' not found in {model_name} model settings."
                f"Replacing with default value: {setting}."
                f"This setting will be required in future versions",
                FutureWarning,
                stacklevel=2,
            )
            setattr(model_settings, key, setting)
    return model_settings


def get_model_constants(model_settings):
    """
    Read constants from model settings file

    Returns
    -------
    constants : dict
        dictionary of constants to add to locals for use by expressions in model spec
    """
    if hasattr(model_settings, "CONSTANTS"):
        return model_settings.CONSTANTS
    return model_settings.get("CONSTANTS", {})


def get_logit_model_settings(
    model_settings: LogitComponentSettings | dict[str, Any] | None
):
    """
    Read nest spec (for nested logit) from model settings file

    Returns
    -------
    nests : dict
        dictionary specifying nesting structure and nesting coefficients
    """
    if isinstance(model_settings, LogitComponentSettings):
        # all the validation for well formatted settings is handled by pydantic,
        # so we just return the nests here.
        return model_settings.NESTS

    nests = None

    if model_settings is not None:
        # default to MNL
        logit_type = model_settings.get("LOGIT_TYPE", "MNL")

        if logit_type not in ["NL", "MNL"]:
            logger.error("Unrecognized logit type '%s'" % logit_type)
            raise RuntimeError("Unrecognized logit type '%s'" % logit_type)

        if logit_type == "NL":
            nests = model_settings.get("NESTS", None)
            if nests is None:
                logger.error("No NEST found in model spec for NL model type")
                raise RuntimeError("No NEST found in model spec for NL model type")

    return nests


def filter_warnings(state=None):
    """
    set warning filter to 'strict' if specified in settings
    """

    if state is None:
        strict = False
    else:
        strict = state.settings.treat_warnings_as_errors

    if strict:  # noqa: E402
        warnings.filterwarnings("error", category=Warning)
        warnings.filterwarnings(
            "default", category=PendingDeprecationWarning, module="future"
        )
        warnings.filterwarnings("default", category=FutureWarning, module="pandas")
        warnings.filterwarnings("default", category=RuntimeWarning, module="numpy")

    # pandas pytables.py __getitem__ (e.g. df = store['any_string'])
    # indirectly raises tables DeprecationWarning: tostring() is deprecated. Use tobytes() instead.
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning, module="tables", message="tostring"
    )

    #   File "tables/hdf5extension.pyx", line 1450, in tables.hdf5extension.Array._open_array
    # DeprecationWarning: `np.object` is a deprecated alias for the builtin `object`.
    # Deprecated in NumPy 1.20;
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module="tables",
        message="`np.object` is a deprecated alias",
    )

    # Numba triggers a DeprecationWarning from numpy about np.MachAr
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module="numba",
        message=".np.MachAr. is deprecated",
    )

    # beginning pandas version 1.3, various places emit a PerformanceWarning that is
    # caught in the "strict" filter above, but which are currently unavoidable for complex models.
    # Turning this filter back to "default" could be a good helper for finding places to
    # look for future performance enhancements.
    from pandas.errors import PerformanceWarning

    warnings.filterwarnings("ignore", category=PerformanceWarning)

    # pandas 1.5
    # beginning in pandas version 1.5, a new warning is emitted when a column is set via iloc
    # from an array of different dtype, the update will eventually be done in-place in future
    # versions. This is actually the preferred outcome for ActivitySim and no code changes are
    # needed.
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=(
            ".*will attempt to set the values inplace instead of always setting a new array. "
            "To retain the old behavior, use either.*"
        ),
    )
    # beginning in pandas version 1.5, a warning is emitted when using pandas.concat on dataframes
    # that contain object-dtype columns with all-bool values.  ActivitySim plans to address dtypes
    # and move away from object-dtype columns anyhow, so this is not a critical problem.
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=(
            ".*object-dtype columns with all-bool values will not be included in reductions.*"
        ),
    )
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=".*will attempt to set the values inplace instead of always setting a new array.*",
    )

    # beginning in sharrow version 2.5, a CacheMissWarning is emitted when a sharrow
    # flow cannot be loaded from cache and needs to be compiled.  These are performance
    # warnings for production runs and totally expected when running test or on new
    # machines
    try:
        from sharrow import CacheMissWarning
    except ImportError:
        pass
    else:
        warnings.filterwarnings("default", category=CacheMissWarning)

    # beginning from PR #660 (after 1.2.0), a FutureWarning is emitted when the trip
    # scheduling component lacks a logic_version setting
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message="The trip_scheduling component now has a logic_version setting.*",
    )
