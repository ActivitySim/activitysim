# ActivitySim
# See full license in LICENSE.txt.
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

# Note that the following import statement has the side-effect of registering injectables:
from activitysim.core import workflow
from activitysim.core.configuration import Settings
from activitysim.core.exceptions import WhaleAccessError


def test_defaults():

    whale = workflow.Whale()
    with pytest.raises(ValidationError):
        whale.initialize_filesystem(working_dir=Path(__file__).parents[1])

    work_dir = Path(__file__).parents[0]
    whale.initialize_filesystem(working_dir=work_dir)

    assert whale.filesystem.get_configs_dir() == (work_dir.joinpath("configs"),)
    assert whale.filesystem.get_data_dir() == (work_dir.joinpath("data"),)
    assert whale.filesystem.get_output_dir() == work_dir.joinpath("output")

    configs_dir = os.path.join(os.path.dirname(__file__), "configs_test_defaults")
    with pytest.raises(ValidationError):
        # can't write one path to configs_dir, must be a tuple
        whale.filesystem.configs_dir = Path(configs_dir)
    whale.filesystem.configs_dir = (Path(configs_dir),)

    with pytest.raises(WhaleAccessError):
        settings = whale.settings

    whale.load_settings()
    assert isinstance(whale.settings, Settings)
