import logging

import altair as alt
import pandas as pd
from pypyr.context import Context

from ..progression import reset_progress_step
from ..wrapping import workstep

logger = logging.getLogger(__name__)


def parse_grouping(g):
    if isinstance(g, str):
        return g, {"shorthand": g}
    elif isinstance(g, dict):
        return g.get("field"), g
    elif g is None:
        return None, None
    else:
        raise ValueError(g)


@workstep
def compare_nominal_choice(
    tablesets,
    tablename,
    nominal_col,
    row_grouping=None,
    col_grouping=None,
    count_label=None,
    share_label=None,
    axis_label="Share",
    title=None,
    ordinal=False,
    plot_type="share",
    relabel_tablesets=None,
):
    """
    Parameters
    ----------
    tablesets : Mapping
    title : str, optional
    grouping : str
    relabel_tablesets : Mapping[str,str]
        Remap the keys in `tablesets` with these values. Any
        missing values are retained.  This allows you to modify
        the figure to e.g. change "reference" to "v1.0.4" without
        editing the original input data.
    """
    if count_label is None:
        count_label = f"# of {tablename}"
    if share_label is None:
        share_label = f"share of {tablename}"
    if relabel_tablesets is None:
        relabel_tablesets = {}

    row_g, row_g_kwd = parse_grouping(row_grouping)
    col_g, col_g_kwd = parse_grouping(col_grouping)

    d = {}
    groupings = []
    if row_g is not None:
        groupings.append(row_g)
    if col_g is not None:
        groupings.append(col_g)

    for key, tableset in tablesets.items():
        df = (
            tableset[tablename]
            .groupby(groupings + [nominal_col])
            .size()
            .rename(count_label)
            .reset_index()
        )
        if not groupings:
            df[share_label] = df[count_label] / df[count_label].sum()
        else:
            df[share_label] = df[count_label] / df.groupby(groupings)[
                count_label
            ].transform("sum")
        d[relabel_tablesets.get(key, key)] = df

    all_d = pd.concat(d, names=["source"]).reset_index()

    selection = alt.selection_multi(
        fields=[nominal_col],
        bind="legend",
    )

    if plot_type == "count":
        x = alt.X(
            count_label,
            axis=alt.Axis(grid=False, labels=False, title=axis_label),
        )
    elif plot_type == "share":
        x = alt.X(
            share_label,
            axis=alt.Axis(grid=False, labels=False, title=axis_label),
            scale=alt.Scale(domain=[0.0, 1.0]),
        )
    else:
        raise ValueError(f"unknown plot_type {plot_type}")

    encode = dict(
        color=alt.Color(
            nominal_col,
            type="ordinal" if ordinal else "nominal",
        ),
        y=alt.Y("source", axis=alt.Axis(grid=False, title=""), sort=None),
        x=x,
        opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
        tooltip=[
            nominal_col,
            "source",
            count_label,
            alt.Tooltip(f"{share_label}:Q", format=".2%"),
        ],
    )
    if row_g is not None:
        encode["row"] = alt.Row(**row_g_kwd)
    if col_g is not None:
        encode["column"] = alt.Column(**col_g_kwd)

    fig = (
        alt.Chart(all_d)
        .mark_bar()
        .encode(
            **encode,
        )
        .add_selection(
            selection,
        )
    )

    if title:
        fig = fig.properties(title=title).configure_title(
            fontSize=20,
            anchor="start",
            color="black",
        )

    if col_grouping is not None:
        fig = fig.properties(
            width=100,
        )

    return fig
