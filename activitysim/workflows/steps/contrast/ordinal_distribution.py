import logging

import altair as alt
import pandas as pd
from pypyr.context import Context

from ..progression import reset_progress_step
from ..wrapping import workstep

logger = logging.getLogger(__name__)


@workstep
def ordinal_distribution(
    tablesets,
    tablename,
    ordinal_col,
    facet_grouping="primary_purpose",
    title=None,
    axis_label=None,
    count_label=None,
    share_label=None,
    interpolate="monotone",
    plot_type="share",
    value_format=None,
):
    if count_label is None:
        count_label = f"# of {tablename}"

    if share_label is None:
        share_label = f"share of {tablename}"

    if axis_label is None:
        axis_label = ordinal_col
    groupings = []
    if isinstance(facet_grouping, str):
        groupings = [facet_grouping]

    d = {}
    for key, tableset in tablesets.items():
        df = (
            tableset[tablename]
            .groupby(groupings + [ordinal_col])
            .size()
            .rename(count_label)
            .unstack(ordinal_col)
            .fillna(0)
            .stack()
            .rename(count_label)
            .reset_index()
        )
        df[share_label] = df[count_label] / df.groupby(groupings)[
            count_label
        ].transform("sum")
        d[key] = df

    # This is sorted in reverse alphabetical order by source, so that
    # the stroke width for the first line plotted is fattest, and progressively
    # thinner lines are plotted over that, so all data is visible on the figure.
    all_d = (
        pd.concat(d, names=["source"])
        .reset_index()
        .sort_values("source", ascending=False)
    )

    if plot_type == "count":
        y = alt.Y(count_label, axis=alt.Axis(grid=False, title=""), stack=None)
    elif plot_type == "share":
        y = alt.Y(share_label, axis=alt.Axis(grid=False, title=""), stack=None)
    else:
        raise ValueError(f"unknown plot_type {plot_type}")

    val_format = {}
    if value_format is not None:
        val_format["format"] = value_format

    fig = (
        alt.Chart(all_d)
        .mark_area(
            interpolate=interpolate,
            fillOpacity=0.3,
            line=True,
        )
        .encode(
            color="source",
            y=y,
            x=alt.X(
                ordinal_col, axis=alt.Axis(grid=False, title=axis_label, **val_format)
            ),
            tooltip=[
                alt.Tooltip(ordinal_col, **val_format),
                "source",
                count_label,
                alt.Tooltip(f"{share_label}:Q", format=".2%"),
            ],
            facet=alt.Facet(facet_grouping, columns=3),
            strokeWidth="source",
        )
        .properties(
            width=200,
            height=120,
        )
    )

    if title:
        fig = fig.properties(title=title).configure_title(
            fontSize=20,
            anchor="start",
            color="black",
        )

    return fig
