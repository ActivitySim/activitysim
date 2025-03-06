# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

# Note that the following import statement has the side-effect of registering injectables:
from activitysim.core import workflow
from activitysim.core.configuration import Settings
from activitysim.core.exceptions import StateAccessError


def test_defaults():
    state = workflow.State()
    with pytest.raises(ValidationError):
        state.initialize_filesystem(working_dir=Path(__file__).parents[1])

    work_dir = Path(__file__).parents[0]
    state.initialize_filesystem(working_dir=work_dir)

    assert state.filesystem.get_configs_dir() == (work_dir.joinpath("configs"),)
    assert state.filesystem.get_data_dir() == (work_dir.joinpath("data"),)
    assert state.filesystem.get_output_dir() == work_dir.joinpath("output")

    configs_dir = os.path.join(os.path.dirname(__file__), "configs_test_defaults")
    with pytest.raises(ValidationError):
        # can't write one path to configs_dir, must be a tuple
        state.filesystem.configs_dir = Path(configs_dir)
    state.filesystem.configs_dir = (Path(configs_dir),)

    with pytest.raises(StateAccessError):
        settings = state.settings

    state.load_settings()
    assert isinstance(state.settings, Settings)
