import logging
from pandas import DataFrame
from pydantic import BaseModel as PydanticBase
from typing import Type

from activitysim.core.workflow import State
from activitysim.core.simulate import eval_coefficients

# import model settings
from activitysim.abm.models.accessibility import AccessibilitySettings
from activitysim.abm.models.atwork_subtour_frequency import (
    AtworkSubtourFrequencySettings,
)

# import logit model settings
from activitysim.core.configuration.logit import TourLocationComponentSettings

logger = logging.getLogger(__name__)
file_logger = logger.getChild("logfile")

COMPONENTS_TO_SETTINGS = {
    "compute_accessibility": {
        "settings_cls": AccessibilitySettings,
        "settings_file": "accessibility.yaml",
    },
    "atwork_subtour_destination": {
        "settings_cls": TourLocationComponentSettings,
        "settings_file": "atwork_subtour_destination.yaml",
    },
    "atwork_subtour_frequency": {
        "settings_cls": AtworkSubtourFrequencySettings,
        "settings_file": "atwork_subtour_frequency.yaml",
    },
}


def try_load_model_settings(
    model_name: str,
    model_settings_class: Type[PydanticBase],
    model_settings_file: str,
    state: State,
) -> PydanticBase:
    logger.info(
        f"Attempting to load model settings for {model_name} via {model_settings_class.__name__} and {model_settings_file}"
    )
    settings = model_settings_class.read_settings_file(
        state.filesystem, model_settings_file
    )
    logger.info(f"Successfully loaded model settings from {model_settings_file}")
    return settings


def try_load_spec(
    model_name: str, model_settings: PydanticBase, state: State
) -> DataFrame:
    logger.info(
        f"Attempting to load SPEC for {model_name} via {model_settings.__class__.__name__}"
    )
    spec_file = model_settings.model_dump().get("SPEC")
    if spec_file is None:
        logger.info(
            f"No SPEC file is associated with {model_settings.__class__.__name__}"
        )
    spec = state.filesystem.read_model_spec(spec_file)
    logger.info(f"Successfully loaded model SPEC from {spec_file}")
    return spec


def try_load_and_eval_coefs(
    model_name: str, model_settings: PydanticBase, spec: DataFrame, state: State
) -> tuple[DataFrame | None, DataFrame | None]:
    logger.info(
        f"Attempting to load coefficients for {model_name} via {model_settings.__class__.__name__}"
    )
    if hasattr(model_settings, "COEFFICIENTS"):
        coefs_file = model_settings.COEFFICIENTS
        coefs = state.filesystem.read_model_coefficients(model_settings)
        eval_coefs = eval_coefficients(state, spec, coefs, estimator=None)
        logger.info(f"Successfully read and evaluated coefficients from {coefs_file}")
        return coefs, eval_coefs
    else:
        logger.info(
            f"No coefficients file is associated with {model_settings.__class__.__name__}"
        )
    return None, None


def try_load_and_check(
    model_name: str,
    model_settings_class: Type[PydanticBase],
    model_settings_file: str,
    state: State,
) -> None:

    # first, attempt to load settings
    settings = try_load_model_settings(
        model_name=model_name,
        model_settings_class=model_settings_class,
        model_settings_file=model_settings_file,
        state=state,
    )

    # then, attempt to read SPEC file
    spec = try_load_spec(model_name=model_name, model_settings=settings, state=state)

    # then, attempt to read and evaluate coefficients
    coefs, eval_coefs = try_load_and_eval_coefs(
        model_name=model_name, model_settings=settings, spec=spec, state=state
    )

    # then, check preprocessors if any
    # for now, check is limited to check that the SPEC file is loadable
    if settings.model_dump().get("preprocessor"):
        preprocessor_settings = settings.preprocessor
        spec = try_load_spec(
            model_name=model_name + ": preprocessor",
            model_settings=preprocessor_settings,
            state=state,
        )
        breakpoint()


def check_model_settings(state: State) -> None:

    components = state.settings.models  # _RUNNABLE_STEPS.keys() may be better?

    for c in components:

        # TODO: this check allows incremental development, but should be deleted.
        if not c in COMPONENTS_TO_SETTINGS:
            logger.info(
                f"Cannot pre-check settings for model component {c}: mapping to a Pydantic data model is undefined in the checker."
            )
            continue

        # first, attempt to load the model settings
        settings_cls = COMPONENTS_TO_SETTINGS[c]["settings_cls"]
        settings_file = COMPONENTS_TO_SETTINGS[c]["settings_file"]
        try_load_and_check(
            model_name=c,
            model_settings_class=settings_cls,
            model_settings_file=settings_file,
            state=state,
        )
