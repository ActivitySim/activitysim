from __future__ import annotations

# ActivitySim
# See full license in LICENSE.txt.
import importlib.resources
import os
import subprocess
from shutil import copytree

import pandas as pd
import pandas.testing as pdt
import yaml


def update_settings(settings_file, key, value):
    with open(settings_file, "r") as f:
        settings = yaml.safe_load(f)
        f.close()

    settings[key] = value

    with open(settings_file, "w") as f:
        yaml.safe_dump(settings, f)
        f.close()


def test_trace_ids_have_same_hash():
    def example_path(dirname):
        resource = os.path.join("examples", "prototype_mtc", dirname)
        return str(importlib.resources.files("activitysim").joinpath(resource))

    def test_path(dirname):
        return os.path.join(os.path.dirname(__file__), dirname)

    new_configs_dir = test_path("configs")
    new_mp_configs_dir = test_path("configs_mp")
    new_settings_file = os.path.join(new_configs_dir, "settings.yaml")
    copytree(example_path("configs"), new_configs_dir)
    copytree(example_path("configs_mp"), new_mp_configs_dir)

    update_settings(
        new_settings_file, "trace_hh_id", 1932009
    )  # Household in the prototype_mtc example with 11 people

    def check_csv_suffix(directory):
        suffix = None
        mismatched_files = []
        for root, dirs, files in os.walk(directory):
            for filename in files:
                if filename.lower().endswith(".csv"):
                    file_suffix = filename[-10:]
                    if suffix is None:
                        suffix = file_suffix
                    elif file_suffix != suffix:
                        mismatched_files.append(os.path.join(root, filename))
        if mismatched_files:
            raise AssertionError(
                f"CSV files with mismatched suffixes: {mismatched_files}"
            )

    file_path = os.path.join(os.path.dirname(__file__), "simulation.py")

    run_args = [
        "-c",
        test_path("configs_mp"),
        "-c",
        test_path("configs"),
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

    check_csv_suffix(os.path.join(test_path("output"), "trace"))


if __name__ == "__main__":
    test_trace_ids_have_same_hash()
