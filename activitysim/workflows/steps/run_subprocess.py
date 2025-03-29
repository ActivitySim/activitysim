from __future__ import annotations

import logging
import os
import subprocess
from tempfile import TemporaryFile
from time import sleep

from pypyr.errors import KeyNotInContextError

from .progression import reset_progress_step
from .wrapping import workstep


def _get_formatted(context, key, default):
    try:
        out = context.get_formatted(key)
    except KeyNotInContextError:
        out = None
    if out is None:
        out = default
    return out


def _stream_subprocess(args, cwd, env):
    with TemporaryFile() as outputstream:
        process = subprocess.Popen(
            args=args,
            shell=True,
            stdout=outputstream,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            env=env,
        )
        while process.poll() is None:
            where = outputstream.tell()
            lines = outputstream.read()
            if not lines:
                # Adjust the sleep interval to your needs
                sleep(0.25)
                # make sure pointing to the last place we read
                outputstream.seek(where)
            else:
                # Windows adds an extra carriage return and then chokes on
                # it when displaying (or, as it were, not displaying) the
                # output.  So we give Windows a little helping hand.
                print(lines.decode().replace("\r\n", "\n"), end="")
    return process


@workstep
def run_activitysim_as_subprocess(
    label=None,
    cwd=None,
    pre_config_dirs=(),
    config_dirs=("configs",),
    data_dir="data",
    output_dir="output",
    ext_dirs=None,
    settings_file=None,
    resume_after=None,
    fast=True,
    conda_prefix=None,
    single_thread=True,
    multi_thread=None,
    persist_sharrow_cache=False,
) -> None:
    if isinstance(pre_config_dirs, str):
        pre_config_dirs = [pre_config_dirs]
    else:
        pre_config_dirs = list(pre_config_dirs)
    if isinstance(config_dirs, str):
        config_dirs = [config_dirs]
    else:
        config_dirs = list(config_dirs)
    if isinstance(ext_dirs, str):
        ext_dirs = [ext_dirs]
    elif ext_dirs is None:
        ext_dirs = []
    else:
        ext_dirs = list(ext_dirs)
    flags = []
    if resume_after:
        flags.append(f" -r {resume_after}")
    if fast:
        flags.append("--fast")
    if persist_sharrow_cache:
        flags.append("--persist-sharrow-cache")
    if settings_file:
        flags.append(f"-s {settings_file}")
    flags = " ".join(flags)
    cfgs = " ".join(f"-c {c}" for c in pre_config_dirs + config_dirs)
    exts = "".join(f" -e {e}" for e in ext_dirs)
    args = f"activitysim run {cfgs}{exts} -d {data_dir} -o {output_dir} {flags}"
    if label is None:
        label = f"{args}"
    else:
        logging.getLogger(__name__).critical(f"\n=======\nSUBPROC {args}\n=======")

    reset_progress_step(description=f"{label}", prefix="[bold green]")

    # args = shlex.split(args)

    env = os.environ.copy()
    _pythonpath = env.pop("PYTHONPATH", None)

    if single_thread:
        env["MKL_NUM_THREADS"] = "1"
        env["OMP_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["NUMBA_NUM_THREADS"] = "1"
        env["VECLIB_MAXIMUM_THREADS"] = "1"
        env["NUMEXPR_NUM_THREADS"] = "1"

    if multi_thread:
        env["MKL_NUM_THREADS"] = str(multi_thread.get("MKL", 1))
        env["OMP_NUM_THREADS"] = str(multi_thread.get("OMP", 1))
        env["OPENBLAS_NUM_THREADS"] = str(multi_thread.get("OPENBLAS", 1))
        env["NUMBA_NUM_THREADS"] = str(multi_thread.get("NUMBA", 1))
        env["VECLIB_MAXIMUM_THREADS"] = str(multi_thread.get("VECLIB", 1))
        env["NUMEXPR_NUM_THREADS"] = str(multi_thread.get("NUMEXPR", 1))

    if conda_prefix:
        conda_prefix_1 = os.environ.get("CONDA_PREFIX_1", None)
        if conda_prefix_1 is None:
            conda_prefix_1 = os.environ.get("CONDA_PREFIX", None)
        if os.name == "nt":
            conda_prefix_1 = conda_prefix_1.replace("\\", "/")
            conda_prefix = conda_prefix.replace("\\", "/")
            joiner = "; "
        else:
            joiner = " && "
        script = [
            f"source {conda_prefix_1}/etc/profile.d/conda.sh",
            f'conda activate "{conda_prefix}"',
            args,
        ]
        args = "bash -c '" + joiner.join(script) + "'"

    process = _stream_subprocess(
        args=args,
        cwd=cwd,
        env=env,
    )

    # don't swallow any error, because it's the Step swallow decorator
    # responsibility to decide to ignore or not.
    if process.returncode:
        raise subprocess.CalledProcessError(
            process.returncode,
            process.args,
            process.stdout,
            process.stderr,
        )
