from pypyr.errors import KeyNotInContextError
from pypyr.steps.py import run_step as _run_step
from .progression import progress, progress_overall, progress_step
from pypyr.steps.fetchyaml import run_step as _fetch
from pypyr.steps.filewriteyaml import run_step as _write
import glob
import os
import shutil

def run_step(context):
    """
    Copy files.

    Args:
        context: pypyr.context.Context. Mandatory.
                 The following context keys expected:
                - copyFiles
                    - source. mandatory. glob-like. Copy these files.
                    - dest. mandatory. path=like. Copy them here.

    Returns:
        None.

    """
    context.assert_key_has_value('copyFiles', __name__)
    copyFiles = context.get_formatted('copyFiles')
    dest = copyFiles['dest']
    os.makedirs(dest, exist_ok=True)
    for filename in glob.glob(copyFiles['source']):
        shutil.copy2(filename, dest)
