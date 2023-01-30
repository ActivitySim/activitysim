import logging

import altair as alt
import numpy as np
import pandas as pd

from ..progression import reset_progress_step
from ..wrapping import workstep

logger = logging.getLogger(__name__)


@workstep
def memory_use_peak(
    combined_peakmem_log,
    include_runs=("legacy", "sharrow", "reference"),
    relabel_tablesets=None,
    memory_measure="uss",
):
    reset_progress_step(description="report component peak memory use")

    if relabel_tablesets is None:
        relabel_tablesets = {}

    include_runs = list(include_runs)

    logger.info(f"building peak memory use report from {combined_peakmem_log}")

    try:
        df = pd.read_csv(
            combined_peakmem_log,
            header=[0, 1],
            skipinitialspace=True,
            index_col=0,
        )
    except FileNotFoundError:
        return f"File Not Found: {combined_peakmem_log}"
    df = df.fillna(0).astype(np.float64)
    df.columns = df.columns.set_names(["source", "mem"])
    mem = df.stack().query(f"mem == '{memory_measure}'")
    include_runs = [i for i in include_runs if i in mem.columns]
    mem = (
        mem[include_runs]
        .reset_index()
        .reset_index()
        .set_index(["event", "index"])
        .drop(columns=["mem"])
        .stack()
        .rename("mem_use")
        .reset_index()
    )
    mem["mem_gigs"] = mem["mem_use"] / 1e9
    mem["source"] = mem["source"].map(lambda x: relabel_tablesets.get(x, x))

    chart = (
        alt.Chart(mem)
        .mark_line(interpolate="step")
        .encode(
            y="mem_gigs",
            color="source",
            x=alt.X("event", sort=None),
            tooltip=["event", "source", "mem_gigs"],
        )
    )
    return chart
