from __future__ import annotations

import logging

import pandas as pd

from activitysim.core import workflow
from activitysim.core.contrast import altair as alt

logger = logging.getLogger(__name__)


def compare_histogram(
    states: dict[str, workflow.State],
    table_name,
    column_name,
    checkpoint_name=None,
    table_filter=None,
    grouping=None,
    relabel_whales=None,
    bins=10,
    bounds=(None, None),
    axis_label=None,
    interpolate="step",
    number_format=",.2f",
    *,
    title=None,
    tickCount=4,
):
    """

    Parameters
    ----------
    states
    skims
    dist_skim_name
    dist_bins
    grouping
    title
    max_dist
    relabel_whales : Mapping[str,str]
        Remap the keys in `states` with these values. Any
        missing values are retained.  This allows you to modify
        the figure to e.g. change "reference" to "v1.0.4" without
        editing the original input data.

    Returns
    -------
    altair.Chart
    """
    if isinstance(alt, Exception):
        raise alt

    if relabel_whales is None:
        relabel_whales = {}

    if grouping:
        groupings = [grouping]
    else:
        groupings = []

    targets = {}
    for key, tableset in states.items():
        if isinstance(tableset, workflow.State):
            df = tableset.get_dataframe(table_name)
        else:
            df = tableset.get_dataframe(table_name, checkpoint_name=checkpoint_name)
        if isinstance(table_filter, str):
            df = df.query(table_filter)
        targets[key] = df[[column_name] + groupings]

    if bins is not None:
        result = pd.concat(targets, names=["source"])
        if bounds[0] is not None:
            result = result[result[column_name] >= bounds[0]]
        if bounds[1] is not None:
            result = result[result[column_name] <= bounds[1]]
        if bounds == (None, None):
            bounds = (result[column_name].min(), result[column_name].max())
        result[column_name] = pd.cut(result[column_name], bins)
        targets = {k: result.loc[k] for k in targets.keys()}

    n = f"n_{table_name}"
    s = f"share_{table_name}"

    d = {}
    for key, dat in targets.items():
        if groupings:
            df = (
                dat.groupby(groupings + [column_name])
                .size()
                .rename(n)
                .unstack(column_name)
                .fillna(0)
                .stack()
                .rename(n)
                .reset_index()
            )
            df[s] = df[n] / df.groupby(groupings)[n].transform("sum")
        else:
            df = dat.groupby(column_name).size().rename(n).reset_index()
            df[s] = df[n] / df[n].sum()
        d[relabel_whales.get(key, key)] = df

    # This is sorted in reverse alphabetical order by source, so that
    # the stroke width for the first line plotted is fattest, and progressively
    # thinner lines are plotted over that, so all data is visible on the figure.
    all_d = (
        pd.concat(d, names=["source"])
        .reset_index()
        .sort_values("source", ascending=False)
    )
    all_d[column_name] = all_d[column_name].apply(lambda x: x.mid)

    if len(states) != 1:
        encode_kwds = dict(
            color="source",
            y=alt.Y(s, axis=alt.Axis(grid=False, title="")),
            x=alt.X(
                f"{column_name}:Q",
                axis=alt.Axis(
                    grid=False,
                    title=axis_label or column_name,
                    format=number_format,
                    tickCount=tickCount,
                ),
            ),
            # opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
            tooltip=[
                "source",
                alt.Tooltip(column_name, format=number_format),
                n,
                alt.Tooltip(f"{s}:Q", format=".2%"),
            ],
            strokeWidth="source",
        )
    else:
        encode_kwds = dict(
            color="source",
            y=alt.Y(s, axis=alt.Axis(grid=False, title="")),
            x=alt.X(
                f"{column_name}:Q",
                axis=alt.Axis(
                    grid=False,
                    title=axis_label or column_name,
                    format=number_format,
                    tickCount=tickCount,
                ),
            ),
            tooltip=[
                alt.Tooltip(column_name, format=number_format),
                n,
                alt.Tooltip(f"{s}:Q", format=".2%"),
            ],
        )

    if grouping:
        encode_kwds["facet"] = alt.Facet(grouping, columns=3)

    if grouping:
        properties_kwds = dict(
            width=200,
            height=120,
        )
    else:
        properties_kwds = dict(
            width=400,
            height=240,
        )

    if bounds[0] is not None and bounds[1] is not None:
        encode_kwds["x"]["scale"] = alt.Scale(domain=bounds)

    if len(states) != 1:
        fig = (
            alt.Chart(all_d)
            .mark_line(
                interpolate=interpolate,
            )
            .encode(**encode_kwds)
            .properties(**properties_kwds)
        )
    else:
        fig = (
            alt.Chart(all_d)
            .mark_bar(binSpacing=0)
            .encode(**encode_kwds)
            .properties(**properties_kwds)
        )

    if title:
        fig = fig.properties(title=title).configure_title(
            fontSize=20,
            anchor="start",
            color="black",
        )

    return fig
