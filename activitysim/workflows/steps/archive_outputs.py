import os
from pathlib import Path
import glob
import shutil

import logging

from pypyr.context import Context
from pypyr.errors import KeyNotInContextError

from .progression import progress_step, progress

logger = logging.getLogger(__name__)


def run_step(context: Context) -> None:

    logger.debug("started")

    context.assert_key_has_value(key='workspace', caller=__name__)
    context.assert_key_has_value(key='example_name', caller=__name__)
    context.assert_key_has_value(key='source', caller=__name__)
    context.assert_key_has_value(key='destination', caller=__name__)

    workspace = context.get_formatted('workspace')
    example_name = context.get_formatted('example_name')
    source = context.get_formatted('source')
    destination = context.get_formatted('destination')
    try:
        patterns = context.get_formatted('patterns')
    except KeyNotInContextError:
        patterns = None
    if patterns is None:
        patterns = ['log/*.*', "*.csv", "*.h5", "*.omx"]

    source_dir = Path(f"{workspace}/{example_name}/{source}")
    dest_dir = Path(f"{workspace}/{example_name}/{destination}")

    progress.reset(progress_step, description=f"[bold black]Archiving to {dest_dir}")

    for pattern in patterns:
        pattern_dir, pattern_base = os.path.split(pattern)
        if pattern_dir:
            into = dest_dir / pattern_dir
            os.makedirs(into, exist_ok=True)
        else:
            into = dest_dir
        for file in glob.glob(os.path.join(source_dir, pattern)):
            shutil.move(file, into)

