from __future__ import annotations

# ActivitySim
# See full license in LICENSE.txt.
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pkg_resources

from activitysim.core import test, workflow


def run_test_mtc(multiprocess=False, chunkless=False, recode=False, sharrow=False):
    def example_path(dirname):
        resource = os.path.join("examples", "prototype_mtc", dirname)
        return pkg_resources.resource_filename("activitysim", resource)

    def test_path(dirname):
        return os.path.join(os.path.dirname(__file__), dirname)

    def regress():
        regress_trips_df = pd.read_csv(test_path("regress/final_trips.csv"))
        final_trips_df = pd.read_csv(test_path("output/final_trips.csv"))
        test.assert_frame_substantively_equal(final_trips_df, regress_trips_df)

    file_path = os.path.join(os.path.dirname(__file__), "simulation.py")

    if multiprocess:
        run_args = [
            "-c",
            test_path("configs_mp"),
            "-c",
            example_path("configs_mp"),
            "-c",
            example_path("configs"),
            "-d",
            example_path("data"),
            "-o",
            test_path("output"),
        ]
    elif chunkless:
        run_args = [
            "-c",
            test_path("configs_chunkless"),
            "-c",
            example_path("configs"),
            "-d",
            example_path("data"),
            "-o",
            test_path("output"),
        ]
    elif recode:
        run_args = [
            "-c",
            test_path("configs_recode"),
            "-c",
            example_path("configs"),
            "-d",
            example_path("data"),
            "-o",
            test_path("output"),
        ]
    elif sharrow:
        run_args = [
            "-c",
            test_path("configs_sharrow"),
            "-c",
            example_path("configs"),
            "-d",
            example_path("data"),
            "-o",
            test_path("output"),
        ]
    else:
        run_args = [
            "-c",
            test_path("configs"),
            "-c",
            example_path("configs"),
            "-d",
            example_path("data"),
            "-o",
            test_path("output"),
        ]

    if os.environ.get("GITHUB_ACTIONS") == "true":
        subprocess.run(["coverage", "run", "-a", file_path] + run_args, check=True)
    else:
        subprocess.run([sys.executable, file_path] + run_args, check=True)

    regress()


def test_mtc():
    run_test_mtc(multiprocess=False)


def test_mtc_chunkless():
    run_test_mtc(multiprocess=False, chunkless=True)


def test_mtc_mp():
    run_test_mtc(multiprocess=True)


def test_mtc_recode():
    run_test_mtc(recode=True)


def test_mtc_sharrow():
    run_test_mtc(sharrow=True)


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


@test.run_if_exists("prototype_mtc_reference_pipeline.zip")
def test_mtc_progressive():
    import activitysim.abm  # register components

    state = workflow.create_example("prototype_mtc", temp=True)

    assert state.settings.models == EXPECTED_MODELS
    assert state.settings.chunk_size == 0
    assert state.settings.sharrow == False

    for step_name in EXPECTED_MODELS:
        state.run.by_name(step_name)
        try:
            state.checkpoint.check_against(
                Path(__file__).parent.joinpath("prototype_mtc_reference_pipeline.zip"),
                checkpoint_name=step_name,
            )
        except Exception:
            print(f"> prototype_mtc {step_name}: ERROR")
            raise
        else:
            print(f"> prototype_mtc {step_name}: ok")


if __name__ == "__main__":
    run_test_mtc(multiprocess=False)
    run_test_mtc(multiprocess=True)
    run_test_mtc(multiprocess=False, chunkless=True)
    run_test_mtc(recode=True)
    run_test_mtc(sharrow=True)
