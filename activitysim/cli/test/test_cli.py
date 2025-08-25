from __future__ import annotations

# ActivitySim
# See full license in LICENSE.txt.
import os
import shutil
import subprocess
import sys

import pytest

if sys.version_info < (3, 7):
    pytest.skip("capture_output introduced in Python 3.7", allow_module_level=True)


def test_help():
    # cp = completed process
    cp = subprocess.run(["activitysim", "-h"], capture_output=True)

    assert "usage: activitysim [-h] [--version]" in str(cp.stdout)


def test_create_help():
    cp = subprocess.run(["activitysim", "create", "-h"], capture_output=True)

    assert "usage: activitysim create [-h] (-l | -e PATH) [-d PATH]" in str(cp.stdout)


def test_create_list():
    cp = subprocess.run(["activitysim", "create", "--list"], capture_output=True)

    assert "Available examples" in str(cp.stdout)
    assert "name: prototype_mtc" in str(cp.stdout)


def test_create_copy():
    target = os.path.join(os.path.dirname(__file__), "test_example")
    cp = subprocess.run(
        [
            "activitysim",
            "create",
            "--example",
            "prototype_mtc",
            "--destination",
            target,
        ],
        capture_output=True,
    )

    assert "copying data ..." in str(cp.stdout)
    assert "copying configs ..." in str(cp.stdout)
    assert "copying configs_mp ..." in str(cp.stdout)
    assert "copying output ..." in str(cp.stdout)

    # replace slashes on windows
    assert str(target).replace("\\\\", "\\") in str(cp.stdout).replace("\\\\", "\\")

    assert os.path.exists(target)
    for folder in ["configs", "configs_mp", "data", "output"]:
        assert os.path.isdir(os.path.join(target, "prototype_mtc", folder))

    # clean up
    shutil.rmtree(target)
    assert not os.path.exists(target)


def test_run():
    cp = subprocess.run(["activitysim", "run"], capture_output=True)

    # expect error
    assert "missing" in str(cp.stderr)


if __name__ == "__main__":
    test_help()
    test_create_help()
    test_create_list()
    test_create_copy()
    test_run()
