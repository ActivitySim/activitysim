from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from activitysim.core import workflow
from activitysim.core.contrast import altair as alt

logger = logging.getLogger(__name__)


def _parse_grouping(g):
    if isinstance(g, str):
        return g, {"shorthand": g}
    elif isinstance(g, dict):
        return g.get("field"), g
    elif g is None:
        return None, None
    else:
        raise ValueError(g)


class NominalTarget:
    def __init__(self, counts: dict):
        total = sum(counts[i] for i in counts)
        self._shares = {k: v / total for (k, v) in counts.items()}
        self._counts = counts

    def as_dataframe(self, table_name, column_name):
        targets = {}
        if self._shares is not None:
            targets[f"share of {table_name}"] = self._shares
        if self._counts is not None:
            targets[f"# of {table_name}"] = self._counts
        return pd.DataFrame(targets).rename_axis(column_name, axis=0).reset_index()


def compare_nominal(
    states: dict[str, workflow.State],
    table_name: str,
    column_name: str,
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
    states : Mapping[str, BasicState]
    """
    if isinstance(alt, Exception):
        raise alt

    if isinstance(states, workflow.State):
        states = {"results": states}

    if count_label is None:
        count_label = f"# of {table_name}"
    if share_label is None:
        share_label = f"share of {table_name}"
    if relabel_tablesets is None:
        relabel_tablesets = {}

    row_g, row_g_kwd = _parse_grouping(row_grouping)
    col_g, col_g_kwd = _parse_grouping(col_grouping)

    d = {}
    groupings = []
    if row_g is not None:
        groupings.append(row_g)
    if col_g is not None:
        groupings.append(col_g)

    for key, state in states.items():
        if isinstance(state, workflow.State):
            raw = state.get_dataarray(table_name, column_name)
            df = (
                state.get_dataframe(table_name)
                .groupby(groupings + [column_name])
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
        elif isinstance(state, NominalTarget):
            d[relabel_tablesets.get(key, key)] = state.as_dataframe(
                table_name, column_name
            )
        else:
            raise TypeError(f"states cannot be {type(state)!r}")

    all_d = pd.concat(d, names=["source"]).reset_index()

    selection = alt.selection_multi(
        fields=[column_name],
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
            column_name,
            type="ordinal" if ordinal else "nominal",
        ),
        y=alt.Y("source", axis=alt.Axis(grid=False, title=""), sort=None),
        x=x,
        opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
        tooltip=[
            column_name,
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
