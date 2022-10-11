# ActivitySim
# See full license in LICENSE.txt.
import os
import subprocess

import pandas as pd
import pandas.testing as pdt
import pkg_resources

from activitysim.core import inject


def teardown_function(func):
    inject.clear_cache()
    inject.reinject_decorated_tables()


def run_test_mtc_extended(multiprocess=False):
    def example_path(dirname):
        resource = os.path.join("examples", "prototype_mtc_extended", dirname)
        return pkg_resources.resource_filename("activitysim", resource)

    def example_mtc_path(dirname):
        resource = os.path.join("examples", "prototype_mtc", dirname)
        return pkg_resources.resource_filename("activitysim", resource)

    def test_path(dirname):
        return os.path.join(os.path.dirname(__file__), dirname)

    def regress():
        regress_trips_df = pd.read_csv(test_path("regress/final_trips.csv"))
        final_trips_df = pd.read_csv(test_path("output/final_trips.csv"))

        regress_vehicles_df = pd.read_csv(test_path("regress/final_vehicles.csv"))
        final_vehicles_df = pd.read_csv(test_path("output/final_vehicles.csv"))

        regress_accessibility_df = pd.read_csv(
            test_path("regress/final_proto_disaggregate_accessibility.csv")
        )
        final_accessibiliy_df = pd.read_csv(
            test_path("output/final_proto_disaggregate_accessibility.csv")
        )

        pdt.assert_frame_equal(final_trips_df, regress_trips_df)
        pdt.assert_frame_equal(final_vehicles_df, regress_vehicles_df)
        pdt.assert_frame_equal(final_accessibiliy_df, regress_accessibility_df)

    file_path = os.path.join(os.path.dirname(__file__), "simulation.py")

    if multiprocess:
        run_args = [
            "-c",
            test_path("configs_mp"),
            "-c",
            example_path("configs_mp"),
            "-c",
            test_path("configs"),
            "-c",
            example_path("configs"),
            "-c",
            example_mtc_path("configs"),
            "-d",
            example_mtc_path("data"),
            "-o",
            test_path("output"),
        ]
    else:
        run_args = [
            "-c",
            test_path("configs"),
            "-c",
            example_path("configs"),
            "-c",
            example_mtc_path("configs"),
            "-d",
            example_mtc_path("data"),
            "-o",
            test_path("output"),
        ]

    subprocess.run(["coverage", "run", "-a", file_path] + run_args, check=True)

    regress()


def test_mtc_extended():
    run_test_mtc_extended(multiprocess=False)


def test_mtc_extended_mp():
    run_test_mtc_extended(multiprocess=True)
