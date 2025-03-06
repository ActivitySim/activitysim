# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import os

import pandas as pd
import pytest
import yaml

# Note that the following import statement has the side-effect of registering injectables:
from activitysim.core import configuration, input, workflow


@pytest.fixture(scope="module")
def seed_households():
    return pd.DataFrame(
        {
            "HHID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "home_zone_id": [8, 8, 8, 8, 12, 12, 15, 16, 16, 18],
        }
    )


@pytest.fixture(scope="module")
def state():
    configs_dir = os.path.join(os.path.dirname(__file__), "configs")

    output_dir = os.path.join(os.path.dirname(__file__), "output")

    data_dir = os.path.join(os.path.dirname(__file__), "temp_data")

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    state = workflow.State().initialize_filesystem(
        configs_dir=(configs_dir,),
        output_dir=output_dir,
        data_dir=(data_dir,),
    )

    yield state

    for file in os.listdir(data_dir):
        os.remove(os.path.join(data_dir, file))

    os.rmdir(data_dir)


def test_missing_table_list(state):
    state.load_settings()
    assert isinstance(state.settings, configuration.Settings)

    with pytest.raises(AssertionError) as excinfo:
        input.read_input_table(state, "households")
    assert "no input_table_list found" in str(excinfo.value)


def test_csv_reader(seed_households, state):
    settings_yaml = """
        input_table_list:
          - tablename: households
            filename: households.csv
            index_col: household_id
            rename_columns:
              HHID: household_id
    """

    settings = yaml.load(settings_yaml, Loader=yaml.SafeLoader)
    settings = configuration.Settings.model_validate(settings)
    state.settings = settings

    hh_file = state.filesystem.get_data_dir()[0].joinpath("households.csv")
    seed_households.to_csv(hh_file, index=False)

    assert os.path.isfile(hh_file)

    df = input.read_input_table(state, "households")

    assert df.index.name == "household_id"


def test_hdf_reader1(seed_households, state):
    settings_yaml = """
        input_table_list:
          - tablename: households
            filename: households.h5
            index_col: household_id
            rename_columns:
              HHID: household_id
    """

    settings = yaml.load(settings_yaml, Loader=yaml.SafeLoader)
    settings = configuration.Settings.model_validate(settings)
    state.settings = settings

    hh_file = state.filesystem.get_data_dir()[0].joinpath("households.h5")
    seed_households.to_hdf(hh_file, key="households", mode="w")

    assert os.path.isfile(hh_file)

    df = input.read_input_table(state, "households")

    assert df.index.name == "household_id"


def test_hdf_reader2(seed_households, state):
    settings_yaml = """
        input_table_list:
          - tablename: households
            h5_tablename: seed_households
            filename: households.h5
            index_col: household_id
            rename_columns:
              HHID: household_id
    """

    settings = yaml.load(settings_yaml, Loader=yaml.SafeLoader)
    settings = configuration.Settings.model_validate(settings)
    state.settings = settings

    hh_file = state.filesystem.get_data_dir()[0].joinpath("households.h5")
    seed_households.to_hdf(hh_file, key="seed_households", mode="w")

    assert os.path.isfile(hh_file)

    df = input.read_input_table(state, "households")

    assert df.index.name == "household_id"


def test_hdf_reader3(seed_households, state):
    settings_yaml = """
        input_store: input_data.h5
        input_table_list:
          - tablename: households
            index_col: household_id
            rename_columns:
              HHID: household_id
    """

    settings = yaml.load(settings_yaml, Loader=yaml.SafeLoader)
    settings = configuration.Settings.model_validate(settings)
    state.settings = settings

    hh_file = state.filesystem.get_data_dir()[0].joinpath("input_data.h5")
    seed_households.to_hdf(hh_file, key="households", mode="w")

    assert os.path.isfile(hh_file)

    df = input.read_input_table(state, "households")

    assert df.index.name == "household_id"


def test_missing_filename(seed_households, state):
    settings_yaml = """
        input_table_list:
          - tablename: households
            index_col: household_id
            rename_columns:
              HHID: household_id
    """

    settings = yaml.load(settings_yaml, Loader=yaml.SafeLoader)
    settings = configuration.Settings.model_validate(settings)
    state.settings = settings

    with pytest.raises(AssertionError) as excinfo:
        input.read_input_table(state, "households")
    assert "no input file provided" in str(excinfo.value)


def test_create_input_store(seed_households, state):
    settings_yaml = """
        create_input_store: True
        input_table_list:
          - tablename: households
            h5_tablename: seed_households
            filename: households.csv
            index_col: household_id
            rename_columns:
              HHID: household_id
    """

    settings = yaml.load(settings_yaml, Loader=yaml.SafeLoader)
    settings = configuration.Settings.model_validate(settings)
    state.settings = settings

    hh_file = state.filesystem.get_data_dir()[0].joinpath("households.csv")
    seed_households.to_csv(hh_file, index=False)

    assert os.path.isfile(hh_file)

    with pytest.raises(NotImplementedError):
        df = input.read_input_table(state, "households")

    # TODO if create_input_store is ever implemented
    #
    # assert df.index.name == "household_id"
    #
    # output_store = os.path.join(inject.get_injectable("output_dir"), "input_data.h5")
    # assert os.path.exists(output_store)
    #
    # store_df = pd.read_hdf(output_store, "seed_households")
    # assert store_df.equals(seed_households)
