# ActivitySim
# See full license in LICENSE.txt.
import os

import pytest

from activitysim.core import inject

# The following import statement has the side-effect of registering injectables:
from .. import __init__


def test_misc():

    inject.clear_cache()

    with pytest.raises(RuntimeError) as excinfo:
        inject.get_injectable("configs_dir")
    assert "directory does not exist" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        inject.get_injectable("data_dir")
    assert "directory does not exist" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        inject.get_injectable("output_dir")
    assert "directory does not exist" in str(excinfo.value)

    configs_dir = os.path.join(os.path.dirname(__file__), "configs_test_misc")
    inject.add_injectable("configs_dir", configs_dir)

    settings = inject.get_injectable("settings")
    assert isinstance(settings, dict)

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    inject.add_injectable("data_dir", data_dir)

    # default values if not specified in settings
    assert inject.get_injectable("chunk_size") == 0
