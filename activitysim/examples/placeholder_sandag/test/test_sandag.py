# ActivitySim
# See full license in LICENSE.txt.
import os
import shutil
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
    resource = os.path.join("examples", "placeholder_sandag", dirname)
    return pkg_resources.resource_filename("activitysim", resource)


def mtc_example_path(dirname):
    resource = os.path.join("examples", "prototype_mtc", dirname)
    return pkg_resources.resource_filename("activitysim", resource)


def psrc_example_path(dirname):
    resource = os.path.join("examples", "placeholder_psrc", dirname)
    return pkg_resources.resource_filename("activitysim", resource)


def build_data():
    shutil.copy(
        example_path(os.path.join("data_3", "maz_to_maz_bike.csv")),
        example_path(os.path.join("data_2", "maz_to_maz_bike.csv")),
    )


@pytest.fixture(scope="module")
def data():
    build_data()


def run_test(zone, multiprocess=False):
    def test_path(dirname):
        return os.path.join(os.path.dirname(__file__), dirname)

    def regress(zone):

        # ## regress tours
        regress_tours_df = pd.read_csv(
            test_path(f"regress/final_{zone}_zone_tours.csv")
        )
        tours_df = pd.read_csv(test_path(f"output/final_{zone}_zone_tours.csv"))
        tours_df.to_csv(
            test_path(f"regress/final_{zone}_zone_tours_last_run.csv"), index=False
        )
        print(f"regress tours")
        pdt.assert_frame_equal(tours_df, regress_tours_df, rtol=1e-03)

        # ## regress trips
        regress_trips_df = pd.read_csv(
            test_path(f"regress/final_{zone}_zone_trips.csv")
        )
        trips_df = pd.read_csv(test_path(f"output/final_{zone}_zone_trips.csv"))
        trips_df.to_csv(
            test_path(f"regress/final_{zone}_zone_trips_last_run.csv"), index=False
        )
        print(f"regress trips")
        pdt.assert_frame_equal(trips_df, regress_trips_df, rtol=1e-03)

    # run test
    file_path = os.path.join(os.path.dirname(__file__), "simulation.py")

    if zone == "2":
        base_configs = psrc_example_path(f"configs")
    else:
        base_configs = mtc_example_path(f"configs")

    run_args = [
        "-c",
        test_path(f"configs_{zone}_zone"),
        "-c",
        example_path(f"configs_{zone}_zone"),
        "-c",
        base_configs,
        "-d",
        example_path(f"data_{zone}"),
        "-o",
        test_path("output"),
    ]

    if multiprocess:
        run_args = run_args + ["-s", "settings_mp.yaml"]

    subprocess.run(["coverage", "run", "-a", file_path] + run_args, check=True)

    regress(zone)


def test_1_zone(data):
    run_test(zone="1", multiprocess=False)


def test_1_zone_mp(data):
    run_test(zone="1", multiprocess=True)


def test_2_zone(data):
    run_test(zone="2", multiprocess=False)


def test_2_zone_mp(data):
    run_test(zone="2", multiprocess=True)


def test_3_zone(data):
    run_test(zone="3", multiprocess=False)


def test_3_zone_mp(data):
    run_test(zone="3", multiprocess=True)


if __name__ == "__main__":

    # call each test explicitly so we get a pass/fail for each
    build_data()
    run_test(zone="1", multiprocess=False)
    run_test(zone="1", multiprocess=True)

    run_test(zone="2", multiprocess=False)
    run_test(zone="2", multiprocess=True)

    run_test(zone="3", multiprocess=False)
    run_test(zone="3", multiprocess=True)
