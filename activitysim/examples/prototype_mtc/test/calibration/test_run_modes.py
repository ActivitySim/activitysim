from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest
import yaml


HERE = Path(__file__).resolve().parent
EXAMPLE_ROOT = HERE.parents[1]
TEST_ROOT = EXAMPLE_ROOT / "test"
BASE_CONFIGS = EXAMPLE_ROOT / "configs"
BASE_MP_CONFIGS = EXAMPLE_ROOT / "configs_mp"
DATA = EXAMPLE_ROOT / "data"
SIMULATION = TEST_ROOT / "simulation.py"

COEFFICIENT_FILES = (
    "workplace_location_coefficients.csv",
    "auto_ownership_coefficients.csv",
    "tour_mode_choice_coefficients.csv",
)

RESULT_TABLES = (
    "final_households.csv",
    "final_persons.csv",
    "final_tours.csv",
)


def _prepare_run(tmp_path: Path, name: str, multiprocess: bool, failing: bool):
    run_dir = tmp_path / name
    configs_dir = run_dir / "configs"
    output_dir = run_dir / "output"
    shutil.copytree(HERE / "configs", configs_dir)
    output_dir.mkdir(parents=True)

    settings_source = (
        HERE / "configs_mp" / "settings.yaml"
        if multiprocess
        else HERE / "configs" / "settings.yaml"
    )
    shutil.copyfile(settings_source, configs_dir / "settings.yaml")

    for file_name in COEFFICIENT_FILES:
        shutil.copyfile(BASE_CONFIGS / file_name, configs_dir / file_name)

    if failing:
        shutil.copyfile(
            configs_dir / "tour_mode_choice_calibration_failing.csv",
            configs_dir / "tour_mode_choice_calibration.csv",
        )

    return configs_dir, output_dir


def _run(configs_dir: Path, output_dir: Path, multiprocess: bool):
    args = [
        sys.executable,
        str(SIMULATION),
        "-c",
        str(configs_dir),
    ]
    if multiprocess:
        args.extend(["-c", str(BASE_MP_CONFIGS)])
    args.extend(
        [
            "-c",
            str(BASE_CONFIGS),
            "-d",
            str(DATA),
            "-o",
            str(output_dir),
        ]
    )
    return subprocess.run(args, check=False, capture_output=True, text=True)


def _resume(configs_dir: Path):
    shutil.copyfile(
        HERE / "configs" / "tour_mode_choice_calibration.csv",
        configs_dir / "tour_mode_choice_calibration.csv",
    )
    settings_path = configs_dir / "settings.yaml"
    with open(settings_path, encoding="utf-8") as stream:
        settings = yaml.safe_load(stream)
    settings["resume_after"] = "non_mandatory_tour_scheduling"
    with open(settings_path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(settings, stream, sort_keys=False)


def _assert_success(result: subprocess.CompletedProcess):
    assert result.returncode == 0, result.stdout + result.stderr


def _assert_equivalent(left_output: Path, right_output: Path):
    for file_name in RESULT_TABLES:
        left = pd.read_csv(left_output / file_name).sort_index(axis=1)
        right = pd.read_csv(right_output / file_name).sort_index(axis=1)
        pdt.assert_frame_equal(left, right, check_dtype=False)

    left_coefficients = pd.read_csv(
        left_output / "calibration" / "final_calibrated_coefficients.csv"
    ).sort_values(["component", "coefficient_name"])
    right_coefficients = pd.read_csv(
        right_output / "calibration" / "final_calibrated_coefficients.csv"
    ).sort_values(["component", "coefficient_name"])
    pdt.assert_frame_equal(
        left_coefficients.reset_index(drop=True),
        right_coefficients.reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def _run_reference(root: Path, name: str, multiprocess: bool) -> Path:
    configs, output = _prepare_run(root, name, multiprocess, failing=False)
    _assert_success(_run(configs, output, multiprocess))
    return output


def _run_resumed(root: Path, name: str, multiprocess: bool) -> Path:
    configs, output = _prepare_run(root, name, multiprocess, failing=True)
    failed = _run(configs, output, multiprocess)
    assert failed.returncode != 0

    progress_path = output / "calibration" / "calibration_progress.json"
    with open(progress_path, encoding="utf-8") as stream:
        progress = json.load(stream)
    assert progress["in_progress_iteration"] == 1

    upstream_coefficients = {
        file_name: (configs / file_name).read_bytes()
        for file_name in COEFFICIENT_FILES[:2]
    }

    _resume(configs)
    _assert_success(_run(configs, output, multiprocess))

    for file_name, contents in upstream_coefficients.items():
        assert (configs / file_name).read_bytes() == contents
    return output


@pytest.fixture(scope="module")
def run_root(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("calibration_run_modes")


@pytest.fixture(scope="module")
def single_output(run_root: Path) -> Path:
    return _run_reference(run_root, "single", multiprocess=False)


@pytest.fixture(scope="module")
def multiprocess_output(run_root: Path) -> Path:
    return _run_reference(run_root, "multiprocess", multiprocess=True)


@pytest.fixture(scope="module")
def single_resumed_output(run_root: Path) -> Path:
    return _run_resumed(run_root, "single_resumed", multiprocess=False)


@pytest.fixture(scope="module")
def multiprocess_resumed_output(run_root: Path) -> Path:
    return _run_resumed(run_root, "multiprocess_resumed", multiprocess=True)


def test_single_and_multiprocess_are_equivalent(
    single_output: Path,
    multiprocess_output: Path,
):
    _assert_equivalent(single_output, multiprocess_output)


def test_single_resume_matches_uninterrupted(
    single_output: Path,
    single_resumed_output: Path,
):
    _assert_equivalent(single_output, single_resumed_output)


def test_multiprocess_resume_matches_uninterrupted(
    multiprocess_output: Path,
    multiprocess_resumed_output: Path,
):
    _assert_equivalent(multiprocess_output, multiprocess_resumed_output)


def test_resumed_single_and_multiprocess_are_equivalent(
    single_resumed_output: Path,
    multiprocess_resumed_output: Path,
):
    _assert_equivalent(single_resumed_output, multiprocess_resumed_output)
