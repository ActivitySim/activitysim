# ActivitySim
# See full license in LICENSE.txt.
import os
import subprocess

import pandas as pd
import pandas.testing as pdt
import pkg_resources
import pytest

from activitysim.core import inject


def teardown_function(func):
    inject.clear_cache()
    inject.reinject_decorated_tables()


def example_path(dirname):
    resource = os.path.join("examples", "placeholder_multiple_zone", dirname)
    return pkg_resources.resource_filename("activitysim", resource)


def mtc_example_path(dirname):
    resource = os.path.join("examples", "prototype_mtc", dirname)
    return pkg_resources.resource_filename("activitysim", resource)


def build_data():
    if os.environ.get("TRAVIS") != "true":
        subprocess.check_call(
            ["coverage", "run", example_path("scripts/two_zone_example_data.py")]
        )
        subprocess.check_call(
            ["coverage", "run", example_path("scripts/three_zone_example_data.py")]
        )


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
        pdt.assert_frame_equal(tours_df, regress_tours_df, rtol=1e-03)

        # regress trips
        regress_trips_df = pd.read_csv(
            test_path(f"regress/final_trips_{zone}_zone.csv")
        )
        trips_df = pd.read_csv(test_path("output/final_trips.csv"))
        trips_df.to_csv(
            test_path(f"regress/final_trips_{zone}_zone_last_run.csv"), index=False
        )
        print("regress trips")
        pdt.assert_frame_equal(trips_df, regress_trips_df, rtol=1e-03)

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

    subprocess.run(["coverage", "run", "-a", file_path] + run_args, check=True)

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


if __name__ == "__main__":

    build_data()
    run_test(zone="2", multiprocess=False)
    run_test(zone="2", multiprocess=True)

    run_test(zone="3", multiprocess=False)
    run_test(zone="3", multiprocess=True)
