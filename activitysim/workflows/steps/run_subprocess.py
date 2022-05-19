import os
import shlex
import subprocess

from pypyr.errors import KeyNotInContextError

from ...standalone.utils import chdir
from .cmd import run_step as _run_cmd
from .cmd.dsl import stream_process
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


@workstep
def run_activitysim_as_subprocess(
    label=None,
    cwd=None,
    pre_config_dirs=(),
    config_dirs=("configs",),
    data_dir="data",
    output_dir="output",
    resume_after=None,
    fast=True,
    conda_prefix=None,
    single_thread=True,
) -> None:
    if isinstance(pre_config_dirs, str):
        pre_config_dirs = [pre_config_dirs]
    else:
        pre_config_dirs = list(pre_config_dirs)
    if isinstance(config_dirs, str):
        config_dirs = [config_dirs]
    else:
        config_dirs = list(config_dirs)
    flags = []
    if resume_after:
        flags.append(f" -r {resume_after}")
    if fast:
        flags.append("--fast")
    flags = " ".join(flags)
    cfgs = " ".join(f"-c {c}" for c in pre_config_dirs + config_dirs)
    args = f"activitysim run {cfgs} -d {data_dir} -o {output_dir} {flags}"
    if label is None:
        label = f"{args}"

    reset_progress_step(description=f"{label}", prefix="[bold green]")

    # args = shlex.split(args)

    env = os.environ.copy()
    pythonpath = env.pop("PYTHONPATH", None)

    if single_thread:
        env["MKL_NUM_THREADS"] = "1"
        env["OMP_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["NUMBA_NUM_THREADS"] = "1"
        env["VECLIB_MAXIMUM_THREADS"] = "1"
        env["NUMEXPR_NUM_THREADS"] = "1"

    # if pythonpath:
    #     print(f"removed PYTHONPATH from ENV: {pythonpath}")
    # else:
    #     print(f"no removed PYTHONPATH from ENV!")
    #
    # for k, v in env.items():
    #     print(f"  - {k}: {v}")

    # if conda_prefix is not None:
    # args = ["conda", "init", "bash", "&&", 'conda', 'activate', conda_prefix, '&&'] + list(args)
    # args = ['conda', 'run', '-p', conda_prefix] + list(args)

    if conda_prefix:
        conda_prefix_1 = os.environ.get("CONDA_PREFIX_1", None)
        if conda_prefix_1 is None:
            conda_prefix_1 = os.environ.get("CONDA_PREFIX", None)
        if os.name == 'nt':
            conda_prefix_1 = conda_prefix_1.replace("\\","/")
            conda_prefix = conda_prefix.replace("\\","/")
            joiner = "; "
        else:
            joiner = " && "
        script = [
            f"source {conda_prefix_1}/etc/profile.d/conda.sh",
            f'conda activate "{conda_prefix}"',
            args,
        ]
        process = subprocess.Popen(
            args="bash -c '" + joiner.join(script) + "'",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=env,
        )
    else:
        process = subprocess.Popen(
            args=args,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=env,
        )

    stream_process(process, label)

    # don't swallow the error, because it's the Step swallow decorator
    # responsibility to decide to ignore or not.
    if process.returncode:
        raise subprocess.CalledProcessError(
            process.returncode, process.args, process.stdout, process.stderr,
        )

    # # Clear all saved state from ORCA
    # import orca
    # orca.clear_cache()
    # orca.clear_all()
    #
    # # Re-inject everything from ActivitySim
    # from ...core.inject import reinject_decorated_tables
    # reinject_decorated_tables(steps=True)
    #
    # # Call the run program inside this process
    # from activitysim.cli.main import prog
    # with chdir(cwd):
    #     namespace = prog().parser.parse_args(shlex.split(args))
    #     namespace.afunc(namespace)
