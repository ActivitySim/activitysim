from __future__ import annotations

import multiprocessing
import time

from ..progression import reset_progress_step
from ..wrapping import workstep


@workstep(updates_context=True)
def contrast_setup(
    example_name,
    tag=None,
    compile=True,
    sharrow=True,
    legacy=True,
    resume_after=True,
    fast=True,
    reference=None,
    reference_asim_version="0.0.0",
    multiprocess=0,
    chunk_training_mode=None,
    main_n_households=None,
    persist_sharrow_cache=False,
):
    reset_progress_step(description="Constrast Setup")
    if tag is None:
        tag = time.strftime("%Y-%m-%d-%H%M%S")
    contrast = sharrow and legacy

    flags = []
    if resume_after:
        flags.append(f" -r {resume_after}")
    if fast:
        flags.append("--fast")
    if persist_sharrow_cache:
        flags.append("--persist-sharrow-cache")

    out = dict(tag=tag, contrast=contrast, flags=" ".join(flags))
    if isinstance(reference, str) and "." in reference:
        out["reference_asim_version"] = reference
        out["reference"] = True
    out["relabel_tablesets"] = {"reference": f"v{reference_asim_version}"}
    multiprocess = int(multiprocess)
    out["is_multiprocess"] = (multiprocess > 1) or (multiprocess < 0)
    if multiprocess >= 0:
        out["num_processes"] = multiprocess
    else:
        # when negative, count the number of cpu cores, and run on half the
        # cores except the absolute value of `multiprocess`, so e.g.
        # if set to -3 and you have 48 cores, then run on (48/2)-3 = 21
        # processes (but at least 2).
        out["num_processes"] = int(multiprocessing.cpu_count() / 2) + multiprocess
        if out["num_processes"] < 2:
            out["num_processes"] = 2

    if chunk_training_mode and chunk_training_mode != "disabled":
        out["chunk_application_mode"] = "production"
    else:
        out["chunk_application_mode"] = chunk_training_mode

    return out
