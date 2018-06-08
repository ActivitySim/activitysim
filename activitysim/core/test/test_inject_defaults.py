# ActivitySim
# See full license in LICENSE.txt.

import os
import tempfile

import numpy as np
import orca
import pytest
import yaml

from .. import inject

# Also note that the following import statement has the side-effect of registering injectables:
from .. import inject_defaults


def teardown_function(func):
    orca.clear_cache()
    inject.reinject_decorated_tables()


def test_defaults():

    orca.clear_cache()

    with pytest.raises(RuntimeError) as excinfo:
        orca.get_injectable("configs_dir")
    assert "directory does not exist" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        orca.get_injectable("data_dir")
    assert "directory does not exist" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        output_dir = orca.get_injectable("output_dir")
        print "output_dir", output_dir
    assert "directory does not exist" in str(excinfo.value)

    configs_dir = os.path.join(os.path.dirname(__file__), 'configs_test_defaults')
    orca.add_injectable("configs_dir", configs_dir)

    settings = orca.get_injectable("settings")
    assert isinstance(settings, dict)

    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    orca.add_injectable("data_dir", data_dir)

    # default values if not specified in settings
    assert orca.get_injectable("chunk_size") == 0
