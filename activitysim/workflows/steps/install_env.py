import os.path

command = (
    "conda create -n asim python=3.9 activitysim -c conda-forge --override-channels"
)

import subprocess

from .wrapping import workstep
from .cmd.dsl import stream_process


@workstep(returns_names="install_env_returncode")
def install_env(
    env_prefix, asim_version="1.0.4", cwd=None, label=None,
):
    if os.path.exists(env_prefix):
        return 0

    command = [
        "mamba",
        "create",
        "--prefix",
        env_prefix,
        f"python=3.9",
        f"activitysim={asim_version}",
        "-c",
        "conda-forge",
        "--override-channels",
        "--yes",
    ]

    if label is None:
        label = f"Creating {asim_version} Environment"

    process = subprocess.Popen(
        command,
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
    )
    stream_process(process, label)

    # don't swallow the error, because it's the Step swallow decorator
    # responsibility to decide to ignore or not.
    if process.returncode:
        raise subprocess.CalledProcessError(
            process.returncode, process.args, process.stdout, process.stderr,
        )

    return process.returncode
