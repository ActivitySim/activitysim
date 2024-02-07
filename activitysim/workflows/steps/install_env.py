from __future__ import annotations

import logging
import os.path
import subprocess

from .cmd.dsl import stream_process
from .progression import reset_progress_step
from .wrapping import workstep

logger = logging.getLogger(__name__)


@workstep(returns_names="install_env_returncode")
def install_env(
    env_prefix,
    asim_version="1.0.4",
    cwd=None,
    label=None,
    python_version="3.9",
):
    if os.path.exists(env_prefix):
        return 0
    reset_progress_step(description=f"Creating activitysim v{asim_version} environment")

    os.makedirs(os.path.dirname(env_prefix), exist_ok=True)
    command = [
        "mamba",
        "create",
        "--prefix",
        env_prefix,
        f"python={python_version}",
        f"activitysim={asim_version}",
        "-c",
        "conda-forge",
        "--override-channels",
        "--yes",
    ]

    if label is None:
        label = f"Creating {asim_version} Environment"

    logger.info(f"running command:\n{' '.join(command)}")
    process = subprocess.Popen(
        " ".join(command),
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
    )
    stream_process(process, label)

    # don't swallow the error, because it's the Step swallow decorator
    # responsibility to decide to ignore or not.
    if process.returncode:
        raise subprocess.CalledProcessError(
            process.returncode,
            process.args,
            process.stdout,
            process.stderr,
        )

    return process.returncode
