from __future__ import annotations

import shlex

from pypyr.errors import KeyNotInContextError

from activitysim.workflows.steps.progression import reset_progress_step
from activitysim.workflows.steps.wrapping import workstep
from activitysim.workflows.utils import chdir


def _get_formatted(context, key, default):
    try:
        out = context.get_formatted(key)
    except KeyNotInContextError:
        out = None
    if out is None:
        out = default
    return out


@workstep
def run_activitysim(
    label=None,
    cwd=None,
    pre_config_dirs=(),
    config_dirs=("configs",),
    data_dir="data",
    output_dir="output",
    ext_dirs=None,
    resume_after=None,
    fast=True,
    settings_file=None,
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
    if settings_file:
        flags.append(f" -s {settings_file}")
    flags = " ".join(flags)
    cfgs = " ".join(f"-c {c}" for c in pre_config_dirs + config_dirs)
    exts = "".join(f" -e {e}" for e in ext_dirs)
    args = f"run {cfgs}{exts} -d {data_dir} -o {output_dir} {flags}"
    if label is None:
        label = f"activitysim {args}"

    reset_progress_step(description=f"{label}", prefix="[bold green]")

    # Call the run program inside this process
    import activitysim.abm  # noqa: F401
    from activitysim.cli.main import prog

    with chdir(cwd):
        namespace = prog().parser.parse_args(shlex.split(args))
        namespace.afunc(namespace)
