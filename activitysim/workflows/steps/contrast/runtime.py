from __future__ import annotations

import logging

import altair as alt
import pandas as pd

from ..progression import reset_progress_step
from ..wrapping import workstep

logger = logging.getLogger(__name__)


@workstep
def contrast_runtime(
    combined_timing_log,
    include_runs=(
        "legacy",
        "sharrow",
    ),
    relabel_tablesets=None,
):
    """
    Generate a figure contrasting runtime for each model component.

    Parameters
    ----------
    combined_timing_log
    include_runs
    relabel_tablesets : Mapping[str,str]
        Remap the keys in `include_runs` with these values. Any
        missing values are retained.  This allows you to modify
        the figure to e.g. change "reference" to "v1.0.4" without
        editing the original input data.

    Returns
    -------

    """
    reset_progress_step(description="report model runtime")

    if relabel_tablesets is None:
        relabel_tablesets = {}

    include_runs = list(include_runs)

    logger.info(f"building runtime report from {combined_timing_log}")

    df = pd.read_csv(combined_timing_log, index_col="model_name")
    include_runs = [i for i in include_runs if i in df.columns]
    df1 = (
        df[include_runs]
        .rename_axis(columns="source")
        .unstack()
        .rename("seconds")
        .reset_index()
        .fillna(0)
    )

    def relabel_source(x):
        return relabel_tablesets.get(x, x)

    df1["source"] = df1["source"].map(relabel_source)
    c = alt.Chart(
        df1,
        height={"step": 20},
    )

    if len(include_runs) == 1:
        result = c.mark_bar(size=6,).encode(
            x=alt.X("seconds:Q", stack=None),
            y=alt.Y("model_name", type="nominal", sort=None),
            color="source",
            tooltip=["source", "model_name", "seconds"],
        ) | alt.Chart(df1).mark_bar().encode(
            color="source",
            x="source",
            y=alt.Y("sum(seconds)", title="Total Seconds"),
            tooltip=["source", "sum(seconds)"],
        )

    elif len(include_runs) == 2:
        result = c.mark_bar(yOffset=-3, size=6,).transform_filter(
            (alt.datum.source == relabel_source(include_runs[0]))
        ).encode(
            x=alt.X("seconds:Q", stack=None),
            y=alt.Y("model_name", type="nominal", sort=None),
            color="source",
            tooltip=["source", "model_name", "seconds"],
        ) + c.mark_bar(
            yOffset=4,
            size=6,
        ).transform_filter(
            (alt.datum.source == relabel_source(include_runs[1]))
        ).encode(
            x=alt.X("seconds:Q", stack=None),
            y=alt.Y("model_name", type="nominal", sort=None),
            color="source",
            tooltip=["source", "model_name", "seconds"],
        ) | alt.Chart(
            df1
        ).mark_bar().encode(
            color="source",
            x="source",
            y=alt.Y("sum(seconds)", title="Total Seconds"),
            tooltip=["source", "sum(seconds)"],
        )

    elif len(include_runs) == 3:
        result = c.mark_bar(yOffset=-5, size=4,).transform_filter(
            (alt.datum.source == relabel_source(include_runs[0]))
        ).encode(
            x=alt.X("seconds:Q", stack=None),
            y=alt.Y("model_name", type="nominal", sort=None),
            color="source",
            tooltip=["source", "model_name", "seconds"],
        ) + c.mark_bar(
            yOffset=0,
            size=4,
        ).transform_filter(
            (alt.datum.source == relabel_source(include_runs[1]))
        ).encode(
            x=alt.X("seconds:Q", stack=None),
            y=alt.Y("model_name", type="nominal", sort=None),
            color="source",
            tooltip=["source", "model_name", "seconds"],
        ) + c.mark_bar(
            yOffset=5,
            size=4,
        ).transform_filter(
            (alt.datum.source == relabel_source(include_runs[2]))
        ).encode(
            x=alt.X("seconds:Q", stack=None),
            y=alt.Y("model_name", type="nominal", sort=None),
            color="source",
            tooltip=["source", "model_name", "seconds"],
        ) | alt.Chart(
            df1
        ).mark_bar().encode(
            color="source",
            x=alt.X("source", type="nominal", sort=None),
            y=alt.Y("sum(seconds)", title="Total Seconds"),
            tooltip=["source", "sum(seconds)"],
        )

    else:
        raise ValueError(f"len(include_runs) == {len(include_runs)}")

    return result
