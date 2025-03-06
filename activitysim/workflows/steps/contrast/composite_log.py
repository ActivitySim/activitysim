from __future__ import annotations

import os

import pandas as pd

from ..progression import reset_progress_step
from ..wrapping import workstep


def to_csv_safe(obj, filename, *args, **kwargs):
    """
    Write a csv as normal, changing the filename if a permission error occurs.
    """
    try:
        obj.to_csv(filename, *args, **kwargs)
    except PermissionError:
        n = 1
        f1, f2 = os.path.splitext(filename)
        while os.path.exists(f"{f1} ({n}){f2}"):
            n += 1
        obj.to_csv(f"{f1} ({n}){f2}", *args, **kwargs)


@workstep(updates_context=True)
def composite_log(
    tag,
    archive_dir,
    compares=("compile", "sharrow", "legacy", "reference"),
) -> dict:
    reset_progress_step(description="composite timing and memory logs")

    timings = {}
    compares = list(compares)
    for t in compares:
        filename = f"{archive_dir}/output-{t}/log/timing_log.csv"
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            df = df.set_index("model_name")["seconds"]
            timings[t] = df.loc[~df.index.duplicated()]
    if timings:
        composite_timing = pd.concat(timings, axis=1)
        to_csv_safe(composite_timing, f"{archive_dir}/combined_timing_log-{tag}.csv")
    mems = {}
    for t in compares:
        filename = f"{archive_dir}/output-{t}/log/mem.csv"
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            df = df.set_index("event")[["rss", "full_rss", "uss"]]
            mems[t] = df.loc[~df.index.duplicated()]
    if mems:
        composite_mem = pd.concat(mems, axis=1)
        to_csv_safe(composite_mem, f"{archive_dir}/combined_mem_log-{tag}.csv")
    peaks = {}
    for t in compares:
        filename = f"{archive_dir}/output-{t}/log/memory_profile.csv"
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            df["time"] = pd.to_datetime(df["time"])
            peak_by_event = (
                df.groupby("event")[["rss", "full_rss", "uss", "time"]]
                .max()
                .sort_values("time")
            )
            peaks[t] = peak_by_event[["rss", "full_rss", "uss"]]
    if peaks:
        composite_peaks = pd.concat(peaks, axis=1)
        to_csv_safe(composite_peaks, f"{archive_dir}/combined_mem_peak-{tag}.csv")
    return dict(
        combined_timing_log=f"{archive_dir}/combined_timing_log-{tag}.csv",
        combined_mem_log=f"{archive_dir}/combined_mem_log-{tag}.csv",
        combined_peakmem_log=f"{archive_dir}/combined_mem_peak-{tag}.csv",
    )
