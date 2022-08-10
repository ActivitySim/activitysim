# ActivitySim
# See full license in LICENSE.txt.
import os

import pytest

# Note that the following import statement has the side-effect of registering injectables:
from .. import config, inject


def teardown_function(func):
    inject.clear_cache()
    inject.reinject_decorated_tables()


def test_defaults():

    inject.clear_cache()

    with pytest.raises(RuntimeError) as excinfo:
        inject.get_injectable("configs_dir")
    assert "directory does not exist" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        inject.get_injectable("data_dir")
    assert "directory does not exist" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        output_dir = inject.get_injectable("output_dir")
        print("output_dir", output_dir)
    assert "directory does not exist" in str(excinfo.value)

    configs_dir = os.path.join(os.path.dirname(__file__), "configs_test_defaults")
    inject.add_injectable("configs_dir", configs_dir)

    settings = inject.get_injectable("settings")
    assert isinstance(settings, dict)

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    inject.add_injectable("data_dir", data_dir)
