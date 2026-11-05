from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
import pytest

# import models is necessary to initalize the model steps
from activitysim.abm import models  # noqa: F401
from activitysim.core import los, workflow


@pytest.fixture(scope="module")
def initialize_pipeline(
    module: str,
    tables: dict[str, str],
    initialize_network_los: bool,
    base_dir: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> workflow.State:
    if base_dir is None:
        base_dir = Path("test").joinpath(module)
    configs_dir = base_dir.joinpath("configs")
    data_dir = base_dir.joinpath("data")
    output_dir = tmp_path_factory.mktemp("summarize-output")

    state = (
        workflow.State()
        .initialize_filesystem(
            configs_dir=configs_dir,
            data_dir=data_dir,
            output_dir=output_dir,
        )
        .load_settings()
    )

    # Read in the input test dataframes
    for dataframe_name, idx_name in tables.items():
        df = pd.read_csv(
            data_dir.joinpath(f"{dataframe_name}.csv"),
            index_col=idx_name,
        )
        state.add_table(dataframe_name, df)

    if initialize_network_los:
        net_los = los.Network_LOS(state)
        net_los.load_data()
        state.add_injectable("network_los", net_los)

    # Add the dataframes to the pipeline
    state.checkpoint.restore()
    state.checkpoint.add("init")
    state.checkpoint.close_store()

    # By convention, this method needs to yield something
    yield state

    # pytest teardown code
    state.checkpoint.close_store()
    pipeline_file_path = os.path.join(output_dir, "pipeline.h5")
    if os.path.exists(pipeline_file_path):
        os.unlink(pipeline_file_path)


@pytest.fixture(scope="module")
def base_dir() -> Path:
    """
    A pytest fixture that returns the data folder location.
    :return: folder location for any necessary data to initialize the tests
    """
    return Path(__file__).parent


# Used by conftest.py initialize_pipeline method
@pytest.fixture(scope="module")
def module() -> str:
    """
    A pytest fixture that returns the data folder location.
    :return: folder location for any necessary data to initialize the tests
    """
    return "summarize"


# Used by conftest.py initialize_pipeline method
@pytest.fixture(scope="module")
def tables() -> dict[str, str]:
    """
    A pytest fixture that returns the "mock" tables to build pipeline dataframes. The
    key-value pair is the name of the table and the index column.
    :return: dict
    """
    return {
        "land_use": "zone_id",
        "tours": "tour_id",
        "trips": "trip_id",
        "persons": "person_id",
        "households": "household_id",
    }


# Used by conftest.py initialize_pipeline method
# Set to true if you need to read skims into the pipeline
@pytest.fixture(scope="module")
def initialize_network_los() -> bool:
    """
    A pytest boolean fixture indicating whether network skims should be read from the
    fixtures test data folder.
    :return: bool
    """
    return True


def test_summarize(initialize_pipeline: workflow.State, caplog):
    state = initialize_pipeline
    # Run summarize model
    caplog.set_level(logging.DEBUG)
    state.run(models=["summarize"])

    # Retrieve output tables to check contents
    model_settings = state.filesystem.read_model_settings("summarize.yaml")
    output_location = (
        model_settings["OUTPUT"] if "OUTPUT" in model_settings else "summaries"
    )
    # Check that households are counted correctly
    households_count = pd.read_csv(
        state.get_output_file_path(
            os.path.join(output_location, f"households_count.csv")
        )
    )
    households = pd.read_csv(state.filesystem.get_data_file_path("households.csv"))
    assert int(households_count.iloc[0]) == len(households)

    # Check that bike trips are counted correctly
    trips_by_mode_count = pd.read_csv(
        state.get_output_file_path(
            os.path.join(output_location, f"trips_by_mode_count.csv")
        )
    )
    trips = pd.read_csv(state.filesystem.get_data_file_path("trips.csv"))
    assert int(trips_by_mode_count.BIKE.iloc[0]) == len(
        trips[trips.trip_mode == "BIKE"]
    )

    # Check that _del removes temporary dataframes from the expression namespace
    temporary_dataframe_deleted = pd.read_csv(
        state.get_output_file_path(
            os.path.join(output_location, "temporary_dataframe_deleted.csv")
        )
    )
    assert temporary_dataframe_deleted["deleted"].tolist() == [True]


def test_summarize_drop_unused_columns(
    initialize_pipeline: workflow.State, monkeypatch
):
    from activitysim.abm.models import summarize as summarize_module
    from activitysim.abm.models.summarize import SummarizeSettings

    state = initialize_pipeline
    state.settings.duplicate_step_execution = "allow"
    read_settings_file = SummarizeSettings.read_settings_file.__func__

    column_counts = []
    trim_unused_columns = summarize_module.trim_unused_columns

    def recording_trim(state, model_settings, spec, *tables):
        trimmed = trim_unused_columns(state, model_settings, spec, *tables)
        column_counts.append(
            [(len(a.columns), len(b.columns)) for a, b in zip(tables, trimmed)]
        )
        return trimmed

    monkeypatch.setattr(summarize_module, "trim_unused_columns", recording_trim)

    output_dirs = {}
    for drop_unused in (False, True):

        def patched_read_settings_file(cls, *args, drop_unused=drop_unused, **kwargs):
            model_settings = read_settings_file(cls, *args, **kwargs)
            model_settings.EXPORT_PIPELINE_TABLES = False
            model_settings.DROP_UNUSED_COLUMNS = drop_unused
            model_settings.OUTPUT = f"summarize_drop_{drop_unused}"
            return model_settings

        monkeypatch.setattr(
            SummarizeSettings,
            "read_settings_file",
            classmethod(patched_read_settings_file),
        )
        state.run.summarize()
        output_dirs[drop_unused] = Path(
            state.get_output_file_path(f"summarize_drop_{drop_unused}")
        )

    # trimming is only invoked when enabled, and must actually remove columns
    assert len(column_counts) == 1
    assert all(after < before for before, after in column_counts[0])

    full_files = sorted(p.name for p in output_dirs[False].glob("*.csv"))
    assert full_files
    assert full_files == sorted(p.name for p in output_dirs[True].glob("*.csv"))
    for name in full_files:
        pd.testing.assert_frame_equal(
            pd.read_csv(output_dirs[False] / name),
            pd.read_csv(output_dirs[True] / name),
            obj=name,
        )
