from __future__ import annotations

# ActivitySim
# See full license in LICENSE.txt.
import os
import subprocess
from shutil import copytree

import pandas as pd
import pandas.testing as pdt
import pkg_resources
import yaml


def update_settings(settings_file, key, value):
    with open(settings_file, "r") as f:
        settings = yaml.safe_load(f)
        f.close()

    settings[key] = value

    with open(settings_file, "w") as f:
        yaml.safe_dump(settings, f)
        f.close()


def run_test_random_seed():
    steps_to_run = [
        "initialize_landuse",
        "initialize_households",
        "compute_accessibility",
        "workplace_location",
        "write_tables",
    ]

    def example_path(dirname):
        resource = os.path.join("examples", "prototype_mtc", dirname)
        return pkg_resources.resource_filename("activitysim", resource)

    def test_path(dirname):
        return os.path.join(os.path.dirname(__file__), dirname)

    def create_rng_configs(rng_base_seed=None):
        new_configs_dir = test_path("configs_random_seed__{}".format(rng_base_seed))
        new_settings_file = os.path.join(new_configs_dir, "settings.yaml")
        copytree(example_path("configs"), new_configs_dir)

        update_settings(new_settings_file, "models", steps_to_run)
        if rng_base_seed != "":  # Undefined
            update_settings(new_settings_file, "rng_base_seed", rng_base_seed)

    # (run name, rng_base_seed value)
    runs = [
        ("0-a", 0),
        ("0-b", 0),
        ("1-a", 1),
        ("1-b", 1),
        ("None-a", None),
        ("None-b", None),
        ("Undefined", ""),
    ]

    seeds = list(set([run[1] for run in runs]))
    for seed in seeds:
        create_rng_configs(seed)

    outputs = {}

    def check_outputs(df1, df2, should_be_equal=True):
        """
        Compares df1 and df2 and raises an AssertionError if they are unequal when `should_be_equal` is True and equal when `should_be_equal` is False
        """
        if should_be_equal:
            pdt.assert_frame_equal(outputs[df1], outputs[df2])
        else:
            try:
                pdt.assert_frame_equal(outputs[df1], outputs[df2])
            except AssertionError:
                pass
            else:
                raise AssertionError("outputs did not change when they should have")

    file_path = os.path.join(os.path.dirname(__file__), "simulation.py")

    # running prototype mtc model through workplace_location with 3 random seed settings, 0, 1, None, and undefined.
    # final_persons.csv is compared to ensure the same setting returns the same variables and a different setting returns something different.
    for name, seed in runs:
        run_args = [
            "-c",
            test_path("configs_random_seed__{}".format(seed)),
            "-d",
            example_path("data"),
            "-o",
            test_path("output"),
        ]

        try:
            os.mkdir(test_path("output"))
        except FileExistsError:
            pass

        subprocess.run(["coverage", "run", "-a", file_path] + run_args, check=True)

        # Read in output to memory to compare later
        outputs[name] = pd.read_csv(
            os.path.join(test_path("output"), "final_persons.csv")
        )

    check_outputs("0-a", "0-b", True)
    check_outputs("0-a", "Undefined", True)
    check_outputs("1-a", "1-b", True)
    check_outputs("None-a", "None-b", False)
    check_outputs("0-a", "1-a", False)
    check_outputs("None-a", "0-a", False)
    check_outputs("None-a", "1-a", False)
    check_outputs("None-b", "0-a", False)
    check_outputs("None-b", "1-a", False)


def test_random_seed():
    run_test_random_seed()


if __name__ == "__main__":
    run_test_random_seed()
