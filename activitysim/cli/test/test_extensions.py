"""No-data, end-to-end regressions for CLI and Python API extension loading."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

EXTENSION = """\
import multiprocessing
from pathlib import Path
from activitysim.core import workflow
from .value import VALUE

@workflow.step(cache=True, kind="cached_object", overloading=True)
def network_los_preload(state: workflow.State):
    return None

@workflow.step(cache=True, kind="cached_object", overloading=True)
def shadow_pricing_info(state: workflow.State):
    return None

@workflow.step(cache=True, kind="cached_object", overloading=True)
def shadow_pricing_choice_info(state: workflow.State):
    return None

@workflow.step
def extension_hello(state: workflow.State):
    process = multiprocessing.current_process().name
    Path(state.get_output_file_path("hello.txt")).write_text(f"{VALUE}:{process}")
"""

SETTINGS = """\
models: [extension_hello]
num_processes: 1
multiprocess_steps:
  - name: mp_hello
    begin: extension_hello
    num_processes: 1
check_model_settings: true
memory_profile: false
sharrow: false
use_shadow_pricing: false
"""

API_RUNNER = """\
import importlib
import multiprocessing
import os
import sys
from pathlib import Path
from activitysim import abm
from activitysim.core import workflow

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    model, output, extension, multiprocess, elsewhere = sys.argv[1:]
    state = workflow.State.make_default(Path(model), output_dir=Path(output))
    state.import_extensions(extension)
    # Existing example repositories use this public list as importable names.
    for name in state.get_injectable("imported_extensions"):
        checker = importlib.import_module(name + ".settings_checker")
        assert checker.EXTENSION_CHECKER_SETTINGS == {}
    state.settings.multiprocess = multiprocess == "yes"
    os.chdir(elsewhere)
    state.run.all()
"""


@pytest.fixture
def model(tmp_path):
    root = tmp_path / "model space"
    for directory in ("configs", "data", "extensions"):
        (root / directory).mkdir(parents=True)
    (root / "configs" / "settings.yaml").write_text(SETTINGS)
    (root / "extensions" / "__init__.py").write_text(EXTENSION)
    (root / "extensions" / "value.py").write_text("VALUE = 42\n")
    (root / "extensions" / "settings_checker.py").write_text(
        'print("EXTENSION_SETTINGS_CHECKER_IMPORTED", flush=True)\n'
        "EXTENSION_CHECKER_SETTINGS = {}\n"
    )
    return root


def run_and_check(command, cwd, output, multiprocess):
    env = os.environ.copy()
    for variable in (
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[variable] = "1"
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout
    assert (output / "hello.txt").read_text() == (
        "42:mp_hello" if multiprocess else "42:MainProcess"
    )
    return result.stdout


@pytest.mark.parametrize("multiprocess", [False, True], ids=["single", "multiprocess"])
@pytest.mark.parametrize(
    "form", ["relative", "absolute", "bare", "dot", "trailing", "working-dir"]
)
def test_cli_extensions(model, tmp_path, multiprocess, form):
    output = tmp_path / "output"
    cwd = tmp_path
    extra = []
    if form == "relative":
        extension = os.path.join(model.name, "extensions")
    elif form == "absolute":
        extension = str(model / "extensions")
    elif form == "working-dir":
        # A relative -w must not be applied twice after chdir.
        extension = "extensions"
        extra = ["-w", model.name]
    else:
        cwd = model
        extension = {
            "bare": "extensions",
            "dot": "./extensions",
            "trailing": "extensions" + os.sep,
        }[form]
    command = [
        sys.executable,
        "-m",
        "activitysim",
        "run",
        "-c",
        str(model / "configs"),
        "-d",
        str(model / "data"),
        "-o",
        str(output),
        "--ext",
        extension,
        *extra,
    ]
    if multiprocess:
        command += ["-m"]
    stdout = run_and_check(command, cwd, output, multiprocess)
    assert "EXTENSION_SETTINGS_CHECKER_IMPORTED" in stdout


@pytest.mark.parametrize("multiprocess", [False, True], ids=["single", "spawn"])
@pytest.mark.parametrize("form", ["relative", "absolute", "dot", "trailing"])
def test_api_extensions(model, tmp_path, multiprocess, form):
    runner = tmp_path / "api_runner.py"
    runner.write_text(API_RUNNER)
    output = tmp_path / "output"
    elsewhere = tmp_path / "unrelated cwd"
    elsewhere.mkdir()
    extension = {
        "relative": "extensions",
        "absolute": str(model / "extensions"),
        "dot": "./extensions",
        "trailing": "extensions" + os.sep,
    }[form]
    command = [
        sys.executable,
        str(runner),
        str(model),
        str(output),
        extension,
        "yes" if multiprocess else "no",
        str(elsewhere),
    ]
    run_and_check(command, tmp_path, output, multiprocess)
