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


def _test_sandag_xborder(sharrow=False, mp=True):
    def example_path(dirname):
        resource = os.path.join("examples", "prototype_sandag_xborder", dirname)
        return pkg_resources.resource_filename("activitysim", resource)

    def test_path(dirname):
        return os.path.join(os.path.dirname(__file__), dirname)

    def regress():
        if mp:
            regress_trips_df = pd.read_csv(test_path("regress/final_trips.csv"))
        else:
            regress_trips_df = pd.read_csv(
                test_path("regress/final_trips_1_process.csv")
            )
        final_trips_df = pd.read_csv(test_path("output/final_trips.csv"))
        pdt.assert_frame_equal(final_trips_df, regress_trips_df)

    file_path = os.path.join(os.path.dirname(__file__), "../simulation.py")

    run_args = [file_path]
    if sharrow:
        run_args += ["-c", test_path("configs_sharrow")]
    if not mp:
        run_args += ["-c", test_path("configs_single_process")]
    run_args += [
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
        subprocess.run(["coverage", "run", "-a"] + run_args, check=True)
    else:
        subprocess.run([sys.executable] + run_args, check=True)

    regress()


def test_sandag_xborder():
    _test_sandag_xborder(mp=False)


def test_sandag_xborder_mp():
    _test_sandag_xborder(mp=True)


def test_sandag_xborder_sharrow():
    _test_sandag_xborder(sharrow=True, mp=False)


if __name__ == "__main__":

    test_sandag_xborder()
