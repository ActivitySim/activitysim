import logging

from activitysim.core.workflow import State
from activitysim.core.simulate import eval_coefficients

# import model settings
from activitysim.abm.models.accessibility import AccessibilitySettings

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
}


def load_settings_and_eval_spec(state: State) -> None:

    filesystem = state.filesystem
    components = state.settings.models  # _RUNNABLE_STEPS.keys() may be better?

    for c in components:

        # TODO: this check allows incremental development, but should be deleted.
        if not c in COMPONENTS_TO_SETTINGS:
            continue

        # first, attempt to load the model settings
        settings_cls = COMPONENTS_TO_SETTINGS[c]["settings_cls"]
        settings_file = COMPONENTS_TO_SETTINGS[c]["settings_file"]
        logger.info(
            f"Attempting to load model settings for {c} via {settings_cls.__name__} and {settings_file}"
        )
        settings = settings_cls.read_settings_file(filesystem, settings_file)
        logger.info(f"Successfully loaded model settings from {settings_file}")

        # then, attempt to read SPEC file
        logger.info(f"Attempting to load SPEC for {c} via {settings_cls.__name__}")
        spec_file = settings.model_dump().get("SPEC")
        if spec_file is None:
            logger.info(f"No SPEC file is associated with {settings_cls.__name__}")
        spec = filesystem.read_model_spec(spec_file)
        logger.info(f"Successfully loaded model SPEC from {spec_file}")

        # finally, attempt to read and evaluate coefficients
        logger.info(
            f"Attempting to load coefficients for {c} via {settings_cls.__name__}"
        )
        if hasattr(settings, "COEFFICIENTS"):
            coefs_file = settings.COEFFICIENTS
            coefs = filesystem.read_model_coefficients(settings)
            eval_spec = eval_coefficients(state, spec, coefs, estimator=None)
            logger.info(
                f"Successfully read and evaluated coefficients from {coefs_file}"
            )
        else:
            logger.info(
                f"No coefficients file is associated with {settings_cls.__name__}"
            )
        breakpoint()
