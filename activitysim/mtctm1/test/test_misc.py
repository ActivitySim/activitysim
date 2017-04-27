# ActivitySim
# See full license in LICENSE.txt.

import os
import tempfile

import numpy as np
import orca
import pytest
import yaml

# orca injectables complicate matters because the decorators are executed at module load time
# and since py.test collects modules and loads them at the start of a run
# if a test method does something that has a lasting side-effect, then that side effect
# will carry over not just to subsequent test functions, but to subsequently called modules
# for instance, columns added with add_column will remain attached to orca tables
# pytest-xdist allows us to run py.test with the --boxed option which runs every function
# with a brand new python interpreter

# Also note that the following import statement has the side-effect of registering injectables:
from .. import __init__


def test_misc():

    orca.clear_cache()

    with pytest.raises(RuntimeError) as excinfo:
        orca.get_injectable("configs_dir")
    assert "directory does not exist" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        orca.get_injectable("data_dir")
    assert "directory does not exist" in str(excinfo.value)

    with pytest.raises(RuntimeError) as excinfo:
        orca.get_injectable("output_dir")
    assert "directory does not exist" in str(excinfo.value)

    configs_dir = os.path.join(os.path.dirname(__file__), 'configs_test_misc')
    orca.add_injectable("configs_dir", configs_dir)

    settings = orca.get_injectable("settings")
    assert isinstance(settings, dict)

    assert orca.get_injectable("trace_person_ids") == []

    assert orca.get_injectable("trace_tour_ids") == []

    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    orca.add_injectable("data_dir", data_dir)

    with pytest.raises(RuntimeError) as excinfo:
        orca.get_injectable("store")
    assert "store file name not specified in settings" in str(excinfo.value)

    settings = {'store': 'bogus.h5'}
    orca.add_injectable("settings", settings)
    with pytest.raises(RuntimeError) as excinfo:
        orca.get_injectable("store")
    assert "store file not found" in str(excinfo.value)

    # these should be None until overridden
    assert orca.get_injectable("hh_index_name") is None
    assert orca.get_injectable("persons_index_name") is None

    # default values if not specified in settings
    assert orca.get_injectable("hh_chunk_size") == 0
    assert orca.get_injectable("chunk_size") == 0
