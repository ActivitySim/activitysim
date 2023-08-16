# ActivitySim
# See full license in LICENSE.txt.
import os

# The following import statement has the side-effect of registering injectables:
import activitysim.abm  # noqa: F401
from activitysim.core import configuration, workflow


def test_misc():
    configs_dir = os.path.join(os.path.dirname(__file__), "configs_test_misc")
    data_dir = os.path.join(os.path.dirname(__file__), "data")

    state = workflow.State().initialize_filesystem(
        configs_dir=configs_dir,
        data_dir=data_dir,
    )

    state.load_settings()
    assert isinstance(state.settings, configuration.Settings)

    # default values if not specified in settings
    assert state.settings.chunk_size == 0
