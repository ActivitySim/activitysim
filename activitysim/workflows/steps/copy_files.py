from __future__ import annotations

import glob
import os
import shutil

from activitysim.workflows.steps.wrapping import workstep


@workstep
def copy_files(
    source_glob,
    dest_dir,
) -> None:
    """
    Copy files.

    Parameters
    ----------
    source_glob : str or Sequence[str]
        One or more file glob patterns to copy
    dest_dir : path-like
        Files that match the source_glob(s) will be copied here.
    """
    os.makedirs(dest_dir, exist_ok=True)
    if isinstance(source_glob, str):
        source_glob = [source_glob]
    for pattern in source_glob:
        for filename in glob.glob(pattern):
            shutil.copy2(filename, dest_dir)
