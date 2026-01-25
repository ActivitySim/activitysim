from __future__ import annotations

# ActivitySim
# See full license in LICENSE.txt.
import importlib.resources
import os
from shutil import copytree

import pandas as pd
import pytest
import yaml

from activitysim.core import workflow


def example_path(dirname):
    resource = os.path.join("examples", "prototype_mtc", dirname)
    return str(importlib.resources.files("activitysim").joinpath(resource))


def dir_test_path(dirname):
    return os.path.join(os.path.dirname(__file__), dirname)


data_dir = example_path("data")
new_configs_dir = dir_test_path("configs")
new_settings_file = os.path.join(new_configs_dir, "settings.yaml")
# copy example configs to test/skip_failed_choices/configs if not already there
if not os.path.exists(new_configs_dir):
    copytree(example_path("configs"), new_configs_dir)


def update_settings(settings_file, key, value):
    with open(settings_file, "r") as f:
        settings = yaml.safe_load(f)

    settings[key] = value

    with open(settings_file, "w") as f:
        yaml.safe_dump(settings, f)


def update_uec_csv(uec_file, expression, coef_value):
    # read in the uec file
    df = pd.read_csv(uec_file)
    # append a new row, set expression and coef_value
    df.loc[len(df), "Expression"] = expression
    # from the 4th column onward are coefficients
    for col in df.columns[3:]:
        df.loc[len(df) - 1, col] = coef_value
    df.to_csv(uec_file, index=False)


@pytest.fixture
def state():
    configs_dir = new_configs_dir
    output_dir = dir_test_path("output")
    data_dir = example_path("data")

    # turn the global setting on to skip failed choices
    update_settings(new_settings_file, "skip_failed_choices", True)

    # make some choices fail by setting extreme coefficients in the uec
    # auto ownership
    auto_ownership_uec_file = os.path.join(new_configs_dir, "auto_ownership.csv")
    # forcing households in home zone 8 (recoded 7) to fail auto ownership choice
    update_uec_csv(auto_ownership_uec_file, "@df.home_zone_id==7", -999.0)

    # work location choice
    work_location_choice_uec_file = os.path.join(
        new_configs_dir, "workplace_location.csv"
    )
    # forcing workers from home zone 18 to fail work location choice
    # as if there is a network connection problem for zone 18
    update_uec_csv(work_location_choice_uec_file, "@df.home_zone_id==18", -999.0)

    # trip mode choice
    trip_mode_choice_uec_file = os.path.join(new_configs_dir, "trip_mode_choice.csv")
    # forcing trips on drivealone tours to fail trip mode choice
    update_uec_csv(trip_mode_choice_uec_file, "@df.tour_mode=='DRIVEALONEFREE'", -999.0)

    state = workflow.State.make_default(
        configs_dir=configs_dir,
        output_dir=output_dir,
        data_dir=data_dir,
    )

    from activitysim.abm.tables.skims import network_los_preload

    state.get(network_los_preload)

    state.logging.config_logger()
    return state


def test_skip_failed_choices_low_threshold_prototype_mtc(state):

    # check that the setting is indeed set to True
    assert state.settings.skip_failed_choices is True

    # set the threshold to 1% of households are allowed to be skipped
    state.settings.fraction_of_failed_choices_allowed = 0.01

    # the run will fail because more than 1% households will be skipped
    with pytest.raises(RuntimeError) as excinfo:
        state.run(models=state.settings.models, resume_after=None)
    assert "exceeds the allowed threshold of" in str(excinfo.value)


def test_skip_failed_choices_high_threshold_prototype_mtc(state):

    # check that the setting is indeed set to True
    assert state.settings.skip_failed_choices is True

    # set the threshold to 100% of households are allowed to be skipped
    state.settings.fraction_of_failed_choices_allowed = 1.0

    # state.run(models=state.settings.models, resume_after=None)

    for model in state.settings.models:
        state.run.by_name(model)
        skipped_household_ids_dict = state.get("skipped_household_ids", dict())
        # get the keys of the dict
        skipped_household_ids_keys = skipped_household_ids_dict.keys()
        if model == "initialize_households":
            # check households with home zone if 7 and 18 are in the simulated households
            assert 7 in state.get("households").loc[:, "home_zone_id"].values
            assert 18 in state.get("households").loc[:, "home_zone_id"].values
        if model == "auto_ownership_simulate":
            # assert one of the keys contains "auto_ownership_simulate"
            assert any(
                "auto_ownership_simulate" in key for key in skipped_household_ids_keys
            )
            # assert that the number of skipped households for auto_ownership_simulate is 598
            assert state.get("num_skipped_households", 0) == (69 + 598)
            # check no households in home zone 8 (recoded 7) in the remaining households
            assert all(state.get("households").loc[:, "home_zone_id"] != 7)
            assert all(state.get("persons").loc[:, "home_zone_id"] != 7)
        elif model == "workplace_location":
            # assert one of the keys contains "workplace_location"
            assert any(
                "workplace_location" in key for key in skipped_household_ids_keys
            )
            # assert that the number of skipped households for workplace_location is 69
            assert state.get("num_skipped_households", 0) == 69
            # assert no workers from home zone 18 in the remaining workers
            persons_df = state.get_dataframe("persons")
            assert not any(
                (persons_df["home_zone_id"] == 18) & (persons_df["is_worker"] == True)
            )
        elif model == "trip_mode_choice":
            # assert one of the keys contains "trip_mode_choice"
            assert any("trip_mode_choice" in key for key in skipped_household_ids_keys)
            # assert that the number of skipped households for trip_mode_choice is 276
            assert state.get("num_skipped_households", 0) == (69 + 598 + 276)
            # assert no DRIVEALONEFREE tours in the remaining tours
            assert all(state.get("tours").loc[:, "tour_mode"] != "DRIVEALONEFREE")

    # check that the number of skipped households is recorded in state
    assert state.get("num_skipped_households", 0) == 943
