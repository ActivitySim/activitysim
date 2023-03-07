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

from activitysim.core import configuration, test, workflow


def _test_prototype_mtc_extended(
    multiprocess=False, sharrow=False, shadow_pricing=True
):
    def example_path(dirname):
        resource = os.path.join("examples", "prototype_mtc_extended", dirname)
        return pkg_resources.resource_filename("activitysim", resource)

    def example_mtc_path(dirname):
        resource = os.path.join("examples", "prototype_mtc", dirname)
        return pkg_resources.resource_filename("activitysim", resource)

    def test_path(dirname):
        return os.path.join(os.path.dirname(__file__), dirname)

    def regress():
        regress_suffix = ""
        if shadow_pricing:
            regress_suffix += "-shadowpriced"
        if sharrow:
            regress_suffix += "-sharrow"

        output_dir = "output"
        regress_trips_df = pd.read_csv(
            test_path(f"regress/final_trips{regress_suffix}.csv")
        )
        final_trips_df = pd.read_csv(test_path(f"{output_dir}/final_trips.csv"))

        regress_vehicles_df = pd.read_csv(test_path("regress/final_vehicles.csv"))
        final_vehicles_df = pd.read_csv(test_path(f"{output_dir}/final_vehicles.csv"))

        regress_accessibility_df = pd.read_csv(
            test_path(
                f"regress/final_proto_disaggregate_accessibility{regress_suffix}.csv"
            )
        )
        final_accessibiliy_df = pd.read_csv(
            test_path(f"{output_dir}/final_proto_disaggregate_accessibility.csv")
        )
        # new transforms may add columns to final_accessibiliy_df, but that is
        # not a test breakage if the existing columns still match.
        final_accessibiliy_df = final_accessibiliy_df.drop(
            columns=[
                i
                for i in final_accessibiliy_df.columns
                if i not in regress_accessibility_df.columns
            ]
        )
        test.assert_frame_substantively_equal(
            final_accessibiliy_df,
            regress_accessibility_df,
            rtol=1.0e-4,
            check_dtype=False,
        )

        test.assert_frame_substantively_equal(
            final_trips_df, regress_trips_df, rtol=1.0e-4
        )
        test.assert_frame_substantively_equal(
            final_vehicles_df, regress_vehicles_df, rtol=1.0e-4
        )

    file_path = os.path.join(os.path.dirname(__file__), "simulation.py")
    shadowprice_configs = (
        [] if shadow_pricing else ["-c", test_path("no-shadow-pricing")]
    )
    if sharrow:
        sh_configs = ["-c", example_path("configs_sharrow")]
    else:
        sh_configs = []
    if multiprocess:
        mp_configs = [
            "-c",
            test_path("configs_mp"),
            "-c",
            example_path("configs_mp"),
        ]
    elif sharrow:
        mp_configs = [
            "-c",
            test_path("configs"),
        ]
    else:
        mp_configs = [
            "-c",
            test_path("configs"),
        ]
    run_args = (
        shadowprice_configs
        + sh_configs
        + mp_configs
        + [
            "-c",
            example_path("configs"),
            "-c",
            example_mtc_path("configs"),
            "-d",
            example_mtc_path("data"),
            "-o",
            test_path("output"),
        ]
    )
    if os.environ.get("GITHUB_ACTIONS") == "true":
        subprocess.run(["coverage", "run", "-a", file_path] + run_args, check=True)
    else:
        subprocess.run(
            [sys.executable, "-m", "activitysim", "run"] + run_args, check=True
        )

    regress()


def test_prototype_mtc_extended():
    _test_prototype_mtc_extended(
        multiprocess=False, sharrow=False, shadow_pricing=False
    )


def test_prototype_mtc_extended_sharrow():
    _test_prototype_mtc_extended(multiprocess=False, sharrow=True, shadow_pricing=False)


def test_prototype_mtc_extended_mp():
    _test_prototype_mtc_extended(multiprocess=True, sharrow=False, shadow_pricing=False)


def test_prototype_mtc_extended_shadow_pricing():
    _test_prototype_mtc_extended(multiprocess=False, sharrow=False, shadow_pricing=True)


def test_prototype_mtc_extended_sharrow_shadow_pricing():
    _test_prototype_mtc_extended(multiprocess=False, sharrow=True, shadow_pricing=True)


def test_prototype_mtc_extended_mp_shadow_pricing():
    _test_prototype_mtc_extended(multiprocess=True, sharrow=False, shadow_pricing=True)


EXPECTED_MODELS = [
    "initialize_proto_population",
    "compute_disaggregate_accessibility",
    "initialize_landuse",
    "initialize_households",
    "compute_accessibility",
    "school_location",
    "workplace_location",
    "auto_ownership_simulate",
    "vehicle_type_choice",
    "free_parking",
    "cdap_simulate",
    "mandatory_tour_frequency",
    "mandatory_tour_scheduling",
    "school_escorting",
    "joint_tour_frequency",
    "joint_tour_composition",
    "joint_tour_participation",
    "joint_tour_destination",
    "joint_tour_scheduling",
    "non_mandatory_tour_frequency",
    "non_mandatory_tour_destination",
    "non_mandatory_tour_scheduling",
    "vehicle_allocation",
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
]


@test.run_if_exists("prototype_mtc_extended_reference_pipeline.zip")
def test_prototype_mtc_extended_progressive():
    import activitysim.abm  # register components

    state = workflow.create_example("prototype_mtc_extended", temp=True)

    state.settings.households_sample_size = 10
    state.settings.use_shadow_pricing = False
    state.settings.want_dest_choice_sample_tables = False
    state.settings.want_dest_choice_presampling = True
    state.settings.recode_pipeline_columns = False
    state.settings.output_tables = configuration.OutputTables(
        h5_store=False,
        action="include",
        prefix="final_",
        sort=True,
        tables=[
            configuration.OutputTable(
                tablename="trips",
                decode_columns=dict(
                    origin="land_use.zone_id", destination="land_use.zone_id"
                ),
            ),
            "vehicles",
            "proto_disaggregate_accessibility",
        ],
    )

    assert state.settings.models == EXPECTED_MODELS
    assert state.settings.chunk_size == 0
    assert state.settings.sharrow == False

    for step_name in EXPECTED_MODELS:
        state.run.by_name(step_name)
        try:
            state.checkpoint.check_against(
                Path(__file__).parent.joinpath(
                    "prototype_mtc_extended_reference_pipeline.zip"
                ),
                checkpoint_name=step_name,
            )
        except Exception:
            print(f"> prototype_mtc_extended {step_name}: ERROR")
            raise
        else:
            print(f"> prototype_mtc_extended {step_name}: ok")


if __name__ == "__main__":

    test_prototype_mtc_extended()
    test_prototype_mtc_extended_sharrow()
    test_prototype_mtc_extended_mp()
