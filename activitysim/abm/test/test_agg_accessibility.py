from __future__ import annotations

import logging

import pytest

from activitysim.abm import models  # noqa: F401
from activitysim.abm.models.accessibility import (
    AccessibilitySettings,
    compute_accessibility,
)
from activitysim.core import workflow

logger = logging.getLogger(__name__)


@pytest.fixture
def state() -> workflow.State:
    state = workflow.create_example("prototype_mtc", temp=True)

    state.settings.models = [
        "initialize_landuse",
        "initialize_households",
        "compute_accessibility",
    ]
    state.settings.chunk_size = 0
    state.settings.sharrow = False

    state.run.by_name("initialize_landuse")
    state.run.by_name("initialize_households")
    return state


def test_simple_agg_accessibility(state, dataframe_regression):
    state.run.by_name("compute_accessibility")
    df = state.get_dataframe("accessibility")
    dataframe_regression.check(df, basename="simple_agg_accessibility")


def test_agg_accessibility_explicit_chunking(state, dataframe_regression):
    # set top level settings
    state.settings.chunk_size = 0
    state.settings.sharrow = False
    state.settings.chunk_training_mode = "explicit"

    # read the accessibility settings and override the explicit chunk size to 5
    model_settings = AccessibilitySettings.read_settings_file(
        state.filesystem, "accessibility.yaml"
    )
    model_settings.explicit_chunk = 5

    compute_accessibility(
        state,
        state.get_dataframe("land_use"),
        state.get_dataframe("accessibility"),
        state.get("network_los"),
        model_settings,
        model_settings_file_name="accessibility.yaml",
        trace_label="compute_accessibility",
        output_table_name="accessibility",
    )
    df = state.get_dataframe("accessibility")
    dataframe_regression.check(df, basename="simple_agg_accessibility")
