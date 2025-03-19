import logging

from activitysim.core.workflow import State
from activitysim.core.simulate import eval_coefficients

# import model settings
from activitysim.abm.models.accessibility import AccessibilitySettings

logger = logging.getLogger(__name__)
file_logger = logger.getChild("logfile")

COMPONENTS_TO_SETTINGS = {
    'compute_accessibility': {'settings_cls': AccessibilitySettings, 'settings_file': 'accessibility.yaml'}
}

def load_settings_and_eval_spec(state: State) -> None:

    filesystem = state.filesystem
    components = state.settings.models # _RUNNABLE_STEPS.keys() may be better?

    for c in components:
        
        # TODO: this check allows incremental development, but should be deleted.
        if not c in COMPONENTS_TO_SETTINGS:
            continue
        
        settings_cls = COMPONENTS_TO_SETTINGS[c]['settings_cls']
        settings_file = COMPONENTS_TO_SETTINGS[c]['settings_file']
        logger.info(f"Attempting to load {c} settings via {settings_cls.__name__} and {settings_file}")
        settings = settings_cls.read_settings_file(filesystem, settings_file)

