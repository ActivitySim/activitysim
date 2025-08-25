from __future__ import annotations

# ActivitySim
# See full license in LICENSE.txt.
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pkg_resources
import pytest

from activitysim.core import test, workflow


def example_path(dirname):
    resource = os.path.join("examples", "placeholder_multiple_zone", dirname)
    return pkg_resources.resource_filename("activitysim", resource)


def mtc_example_path(dirname):
    resource = os.path.join("examples", "prototype_mtc", dirname)
    return pkg_resources.resource_filename("activitysim", resource)


def build_data():
    if os.environ.get("TRAVIS") != "true":
        if os.environ.get("GITHUB_ACTIONS") == "true":
            go = ["coverage", "run"]
        else:
            go = [sys.executable]
        subprocess.check_call(go + [example_path("scripts/two_zone_example_data.py")])
        subprocess.check_call(go + [example_path("scripts/three_zone_example_data.py")])


@pytest.fixture(scope="module")
def data():
    build_data()


def run_test(zone, multiprocess=False):
    def test_path(dirname):
        return os.path.join(os.path.dirname(__file__), dirname)

    def regress(zone):
        # regress tours
        regress_tours_df = pd.read_csv(
            test_path(f"regress/final_tours_{zone}_zone.csv")
        )
        tours_df = pd.read_csv(test_path("output/final_tours.csv"))
        tours_df.to_csv(
            test_path(f"regress/final_tours_{zone}_zone_last_run.csv"), index=False
        )
        print("regress tours")
        test.assert_frame_substantively_equal(
            tours_df, regress_tours_df, rtol=1e-03, check_dtype=False
        )

        # regress trips
        regress_trips_df = pd.read_csv(
            test_path(f"regress/final_trips_{zone}_zone.csv")
        )
        trips_df = pd.read_csv(test_path("output/final_trips.csv"))
        trips_df.to_csv(
            test_path(f"regress/final_trips_{zone}_zone_last_run.csv"), index=False
        )
        print("regress trips")
        test.assert_frame_substantively_equal(
            trips_df, regress_trips_df, rtol=1e-03, check_dtype=False
        )

    file_path = os.path.join(os.path.dirname(__file__), "simulation.py")

    run_args = [
        "-c",
        test_path(f"configs_{zone}_zone"),
        "-c",
        example_path(f"configs_{zone}_zone"),
        "-c",
        mtc_example_path("configs"),
        "-d",
        example_path(f"data_{zone}"),
        "-o",
        test_path("output"),
    ]

    if multiprocess:
        run_args = run_args + ["-s", "settings_mp"]
    elif zone == "3":
        run_args = run_args + ["-s", "settings_static"]

    if os.environ.get("GITHUB_ACTIONS") == "true":
        subprocess.run(["coverage", "run", "-a", file_path] + run_args, check=True)
    else:
        subprocess.run([sys.executable, file_path] + run_args, check=True)

    regress(zone)


def test_2_zone(data):
    run_test(zone="2", multiprocess=False)


def test_2_zone_mp(data):
    run_test(zone="2", multiprocess=True)


def test_3_zone(data):
    # python simulation.py -c configs_3_zone -c ../configs_3_zone -c \
    # ../../prototype_mtc/configs -d ../data_3 -o output -s settings_mp
    run_test(zone="3", multiprocess=False)


def test_3_zone_mp(data):
    run_test(zone="3", multiprocess=True)


EXPECTED_MODELS = [
    "initialize_landuse",
    "initialize_households",
    "compute_accessibility",
    "school_location",
    "workplace_location",
    "auto_ownership_simulate",
    "free_parking",
    "cdap_simulate",
    "mandatory_tour_frequency",
    "mandatory_tour_scheduling",
    "joint_tour_frequency",
    "joint_tour_composition",
    "joint_tour_participation",
    "joint_tour_destination",
    "joint_tour_scheduling",
    "non_mandatory_tour_frequency",
    "non_mandatory_tour_destination",
    "non_mandatory_tour_scheduling",
    "tour_mode_choice_simulate",
    "atwork_subtour_frequency",
    "atwork_subtour_destination",
    "atwork_subtour_scheduling",
    "atwork_subtour_mode_choice",
    "stop_frequency",
    "trip_purpose",
    "trip_destination",
    "trip_purpose_and_destination",
    "trip_scheduling",
    "trip_mode_choice",
    "write_data_dictionary",
    "track_skim_usage",
    "write_trip_matrices",
    "write_tables",
    "summarize",
]


@test.run_if_exists("reference_pipeline_2_zone.zip")
def test_multizone_progressive(zone="2"):
    zone = str(zone)

    import activitysim.abm  # register components

    def test_path(dirname):
        return os.path.join(os.path.dirname(__file__), dirname)

    if zone == "3":
        settings_file_name = "settings_static.yaml"
    else:
        settings_file_name = "settings.yaml"

    state = workflow.State.make_default(
        configs_dir=(
            test_path(f"configs_{zone}_zone"),
            example_path(f"configs_{zone}_zone"),
            mtc_example_path("configs"),
        ),
        data_dir=(example_path(f"data_{zone}"),),
        output_dir=test_path("output"),
        settings_file_name=settings_file_name,
    )

    assert state.settings.models == EXPECTED_MODELS
    assert state.settings.chunk_size == 0
    assert state.settings.sharrow == False

    state.settings.trace_hh_id = 1099626
    state.tracing.validation_directory = (
        Path(__file__).parent / "reference_trace_2_zone"
    )

    for step_name in EXPECTED_MODELS:
        state.run.by_name(step_name)
        try:
            state.checkpoint.check_against(
                Path(__file__).parent.joinpath("reference_pipeline_2_zone.zip"),
                checkpoint_name=step_name,
            )
        except Exception:
            print(f"> {zone} zone {step_name}: ERROR")
            raise
        else:
            print(f"> {zone} zone {step_name}: ok")


if __name__ == "__main__":
    build_data()
    run_test(zone="2", multiprocess=False)
    run_test(zone="2", multiprocess=True)

    run_test(zone="3", multiprocess=False)
    run_test(zone="3", multiprocess=True)
