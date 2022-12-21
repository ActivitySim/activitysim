# ActivitySim
# See full license in LICENSE.txt.
import os
import subprocess
import sys

import pandas as pd
import pandas.testing as pdt
import pkg_resources

from activitysim.core import inject


def teardown_function(func):
    inject.clear_cache()
    inject.reinject_decorated_tables()


def _test_prototype_mtc_extended(multiprocess=False, sharrow=False):
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

        pdt.assert_frame_equal(final_trips_df, regress_trips_df, rtol=1.0e-4)
        pdt.assert_frame_equal(final_vehicles_df, regress_vehicles_df, rtol=1.0e-4)

    file_path = os.path.join(os.path.dirname(__file__), "simulation.py")
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
    else:
        mp_configs = [
            "-c",
            test_path("configs"),
        ]
    run_args = sh_configs + mp_configs + [
        "-c",
        example_path("configs"),
        "-c",
        example_mtc_path("configs"),
        "-d",
        example_mtc_path("data"),
        "-o",
        test_path("output"),
    ]
    if os.environ.get("GITHUB_ACTIONS") == "true":
        subprocess.run(["coverage", "run", "-a", file_path] + run_args, check=True)
    else:
        subprocess.run(
            [sys.executable, "-m", "activitysim", "run"] + run_args, check=True
        )

    regress()


def test_prototype_mtc_extended():
    _test_prototype_mtc_extended(multiprocess=False, sharrow=False)


def test_prototype_mtc_extended_sharrow():
    _test_prototype_mtc_extended(multiprocess=False, sharrow=True)


def test_prototype_mtc_extended_mp():
    _test_prototype_mtc_extended(multiprocess=True, sharrow=False)


if __name__ == "__main__":

    test_prototype_mtc_extended()
    test_prototype_mtc_extended_sharrow()
    test_prototype_mtc_extended_mp()
