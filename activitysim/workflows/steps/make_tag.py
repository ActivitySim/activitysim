import time

from .progression import reset_progress_step
from .wrapping import workstep


@workstep(updates_context=True)
def make_tag(
    example_name,
    tag=None,
    compile=True,
    sharrow=True,
    legacy=True,
    resume_after=True,
    fast=True,
    mp=False,
    reference=None,
    reference_asim_version="0.0.0",
    multiprocess=0,
):
    reset_progress_step(description="Initialize Tag")
    if tag is None:
        tag = time.strftime("%Y-%m-%d-%H%M%S")
    contrast = sharrow and legacy

    flags = []
    from activitysim.cli.create import EXAMPLES

    run_flags = EXAMPLES.get(example_name, {}).get("run_flags", {})
    if isinstance(run_flags, str):
        flags.append(run_flags)
    elif run_flags:
        flags.append(run_flags.get("multi" if mp else "single"))

    if resume_after:
        flags.append(f" -r {resume_after}")
    if fast:
        flags.append("--fast")

    out = dict(tag=tag, contrast=contrast, flags=" ".join(flags))
    if isinstance(reference, str) and "." in reference:
        out["reference_asim_version"] = reference
        out["reference"] = True
    out["relabel_tablesets"] = {"reference": f"v{reference_asim_version}"}
    out["is_multiprocess"] = multiprocess > 1
    out["num_processes"] = int(multiprocess)
    return out
