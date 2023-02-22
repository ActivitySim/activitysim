import logging
import os
import tarfile
import zipfile
from pathlib import Path

import appdirs
import yaml

from activitysim.cli.create import download_asset

logger = logging.getLogger(__name__)


def registered_external_example(name, working_dir):
    with open(
        Path(__file__).parent.joinpath("external_example_manifest.yaml"), "r"
    ) as eem:
        registered_examples = yaml.load(eem, yaml.SafeLoader)
    if name not in registered_examples:
        raise KeyError(f"{name!r} is not a registered external example")
    if "name" not in registered_examples[name]:
        registered_examples[name]["name"] = name
    return download_external_example(
        working_dir,
        **registered_examples[name],
    )


def list_registered_examples():
    with open(
        Path(__file__).parent.joinpath("external_example_manifest.yaml"), "r"
    ) as eem:
        registered_examples = yaml.load(eem, yaml.SafeLoader)
    return list(registered_examples.keys())


def exercise_external_example(
    name, working_dir, maxfail: int = None, verbose=2, durations=0
):
    try:
        directory = registered_external_example(name, working_dir)
    except Exception as err:
        logger.exception(err)
        raise
    import pytest

    args = []
    if verbose:
        args.append("-" + "v" * verbose)
    if maxfail:
        args.append(f"--maxfail={int(maxfail)}")
    if durations is not None:
        args.append(f"--durations={int(durations)}")
    args.append(os.path.relpath(os.path.normpath(os.path.realpath(directory))))
    return pytest.main(args)


def _run_tests_on_example(name):
    import tempfile

    tempdir = tempfile.TemporaryDirectory()
    resultcode = exercise_external_example(name, tempdir.name)
    return resultcode


def default_cache_dir() -> Path:
    return Path(appdirs.user_cache_dir(appname="ActivitySim")).joinpath(
        "External-Examples"
    )


def download_external_example(
    working_dir,
    url,
    cache_dir=None,
    cache_file_name=None,
    sha256=None,
    name=None,
    assets: dict = None,
    link_assets=True,
):
    # set up cache dir
    if cache_dir is None:
        cache_dir = default_cache_dir()
    else:
        cache_dir = Path(cache_dir)
    if name:
        cache_dir = cache_dir.joinpath(name)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # check if target file exists in cache dir
    if cache_file_name is None:
        cache_file_name = url
        if cache_file_name.startswith("https://github.com/"):
            cache_file_name = cache_file_name.replace("https://github.com/", "")
        if "//" in cache_file_name:
            cache_file_name = cache_file_name.split("//", 1)[1]
        cache_file_name = cache_file_name.replace("/", "_").replace("\\", "_")

    target_path = cache_dir.joinpath(cache_file_name)

    download_asset(url, target_path, sha256, link=False)

    working_dir = Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    # decompress cache file into working directory
    if target_path.suffixes[-2:] == [".tar", ".gz"]:
        with tarfile.open(target_path) as tfile:
            common_prefix = os.path.commonprefix(tfile.getnames())
            if name is not None and common_prefix in {"", ".", "./", None}:
                common_prefix = name
                working_dir = working_dir.joinpath(name)
                working_dir.mkdir(parents=True, exist_ok=True)
            tfile.extractall(working_dir)
    elif target_path.suffix == ".zip":
        with zipfile.ZipFile(target_path, "r") as zf:
            common_prefix = os.path.commonprefix(zf.namelist())
            if name is not None and common_prefix in {"", ".", "./", None}:
                common_prefix = name
                working_dir = working_dir.joinpath(name)
                working_dir.mkdir(parents=True, exist_ok=True)
            zf.extractall(working_dir)
    else:
        raise ValueError(f"unknown archive file type {''.join(target_path.suffixes)}")

    # download assets if any:
    if assets:
        for asset_name, asset_info in assets.items():
            if link_assets:
                asset_target_path = working_dir.joinpath(common_prefix, asset_name)
                download_asset(
                    asset_info.get("url"),
                    asset_target_path,
                    sha256=asset_info.get("sha256", "deadbeef"),
                    link=cache_dir,
                    base_path=working_dir.joinpath(common_prefix),
                )
            else:
                # TODO should cache and copy, this just downloads to new locations
                download_asset(
                    asset_info.get("url"),
                    working_dir.joinpath(common_prefix, asset_name),
                    sha256=asset_info.get("sha256", "deadbeef"),
                )

    return working_dir.joinpath(common_prefix)
