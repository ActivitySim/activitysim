from __future__ import annotations

import logging
import os
import tarfile
import zipfile
from pathlib import Path

import platformdirs
import yaml

from activitysim.cli.create import download_asset

logger = logging.getLogger(__name__)


def registered_external_example(name, working_dir):
    """
    Download a registered external example and copy into a working directory.

    Parameters
    ----------
    name : str
        The unique name for the registered external example.  See
        `activitysim/examples/external_example_manifest.yaml` or run
        `list_registered_examples()` for the names of the registered examples.
    working_dir : path-like
        The location to install the external example.

    Returns
    -------
    Path
        The location where the example was installed, generally a subdirectory
        of `working_dir`.
    """
    with open(Path(__file__).parent.joinpath("external_example_manifest.yaml")) as eem:
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
    """
    Read a list of registered example names.

    Returns
    -------
    list[str]
    """
    with open(Path(__file__).parent.joinpath("external_example_manifest.yaml")) as eem:
        registered_examples = yaml.load(eem, yaml.SafeLoader)
    return list(registered_examples.keys())


def exercise_external_example(
    name, working_dir, maxfail: int = None, verbose=2, durations=0
):
    """
    Use pytest to ensure that an external example is functioning correctly.

    Parameters
    ----------
    name : str
        The unique name for the registered external example.  See
        `activitysim/examples/external_example_manifest.yaml` or run
        `list_registered_examples()` for the names of the registered examples.
    working_dir : path-like
        The location to install a copy of the external example for testing.
    maxfail : int, optional
        Stop testing after this many failures have been detected.
    verbose : int, default 2
        Verbosity level given to pytest.
    durations : int, default 0
        Report the durations of this many of the slowest tests conducted.
        Leave as 0 to report all durations, or set to None to report no
        durations.

    Returns
    -------
    int
        The result code returned by pytest.
    """
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
    return Path(platformdirs.user_cache_dir(appname="ActivitySim")).joinpath(
        "External-Examples"
    )


def download_external_example(
    working_dir: Path,
    url: str | None = None,
    cache_dir: Path | None = None,
    cache_file_name: str | None = None,
    sha256: str | None = None,
    name=None,
    assets: dict = None,
    link_assets: bool = True,
):
    """
    Download an external example.

    Parameters
    ----------
    working_dir : Path
        The working directory where the external example files will be installed.
        The `name` subdirectory of this directory will be created if it does not
        exist, and downloaded files will be installed there.
    url : str, optional
        The main url for the example to download.  This should point to an
        archive file (e.g. *.tar.gz) that will be unpacked into the target
        working subdirectory.
    cache_dir : Path, optional
        The compressed archive(s) will be downloaded and cached in this
        directory.  If not provided, a suitable cache location is chosen based
        on the suggested user cache locations from platformdirs library.
    cache_file_name : str, optional
        The archive at the primary url will be cached with this filename.  It
        is typically not necessary to provide this file name explicitly, as a
        file name will be generated automatically from the url if not given.
    sha256 : str, optional
        This checksum is used to validate the download and/or the cached
        archive file. If the cached file exists but the checksum does not match,
        the file will be re-downloaded.
    name : str, optional
        The name of the external example.  This will become the working
        subdirectory name where files are installed, unless the main archive
        has an embedded name (as a common prefix) in which case the name is
        ignored.
    assets : dict, optional
        Instructions for additional files to be downloaded to support this
        external example (e.g. large data files not in the main archive).
    link_assets : bool, default True
        If set to True, this function will attempt to symlink assets from the
        cache into the target directory instead of copying them.  This can
        save disk space when the same external example is installed multiple
        times.

    Returns
    -------
    Path
        The working subdirectory name where files are installed.
    """
    # set up cache dir
    if cache_dir is None:
        cache_dir = default_cache_dir()
    else:
        cache_dir = Path(cache_dir)
    if name:
        cache_dir = cache_dir.joinpath(name)
    cache_dir.mkdir(parents=True, exist_ok=True)

    working_dir = Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)
    common_prefix = "."

    if url:
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

        # decompress cache file into working directory
        if target_path.suffixes[-2:] == [".tar", ".gz"]:
            with tarfile.open(target_path) as tfile:
                common_prefix = os.path.commonprefix(tfile.getnames())
                if name is not None and common_prefix in {"", ".", "./", None}:
                    common_prefix = name
                    working_dir = working_dir.joinpath(name)
                    working_dir.mkdir(parents=True, exist_ok=True)
                    working_subdir = working_dir
                else:
                    working_subdir = working_dir.joinpath(common_prefix)
                tfile.extractall(working_dir)
        elif target_path.suffixes[-2:] == [".tar", ".zst"]:
            working_dir = working_dir.joinpath(name)
            working_dir.mkdir(parents=True, exist_ok=True)
            working_subdir = working_dir
            from sharrow.utils.tar_zst import extract_zst

            extract_zst(target_path, working_dir)
        elif target_path.suffix == ".zip":
            with zipfile.ZipFile(target_path, "r") as zf:
                common_prefix = os.path.commonprefix(zf.namelist())
                if name is not None and common_prefix in {"", ".", "./", None}:
                    common_prefix = name
                    working_dir = working_dir.joinpath(name)
                    working_dir.mkdir(parents=True, exist_ok=True)
                    working_subdir = working_dir
                else:
                    working_subdir = working_dir.joinpath(common_prefix)
                zf.extractall(working_dir)
        else:
            raise ValueError(
                f"unknown archive file type {''.join(target_path.suffixes)}"
            )

    # download assets if any:
    if assets:
        for asset_name, asset_info in assets.items():
            if link_assets:
                asset_target_path = working_subdir.joinpath(asset_name)
                download_asset(
                    asset_info.get("url"),
                    asset_target_path,
                    sha256=asset_info.get("sha256", "deadbeef"),
                    link=cache_dir,
                    base_path=working_subdir,
                )
            else:
                # TODO should cache and copy, this just downloads to new locations
                download_asset(
                    asset_info.get("url"),
                    working_subdir.joinpath(asset_name),
                    sha256=asset_info.get("sha256", "deadbeef"),
                )

    return working_subdir
