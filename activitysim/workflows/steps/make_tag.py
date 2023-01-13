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
    chunk_training_mode=None,
    main_n_households=None,
):
    reset_progress_step(description="Initialize Tag")
    if tag is None:
        tag = time.strftime("%Y-%m-%d-%H%M%S")
    contrast = sharrow and legacy

    flags = []
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

    if chunk_training_mode and chunk_training_mode != "disabled":
        out["chunk_application_mode"] = "production"
    else:
        out["chunk_application_mode"] = chunk_training_mode

    return out
