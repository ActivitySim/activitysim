
import logging
import pandas as pd
import altair as alt
from pypyr.context import Context
from ..progression import reset_progress_step
from ..wrapping import report_step

logger = logging.getLogger(__name__)


@report_step
def compare_nominal_choice_shares(
        tablesets,
        tablename,
        nominal_col,
        grouping='primary_purpose',
        count_label=None,
        share_label=None,
        share_axis_label='Share',
        title=None,
):
    """
    Parameters
    ----------
    tablesets : Mapping
    title : str, optional
    grouping : str
    """
    if count_label is None:
        count_label = f"# of {tablename}"

    if share_label is None:
        share_label = f"share of {tablename}"

    d = {}
    if grouping is None:
        groupings = None
    else:
        groupings = [grouping,]

    for key, tableset in tablesets.items():
        if groupings is None:
            df = tableset[tablename].groupby(
                [nominal_col]
            ).size().rename(count_label).reset_index()
            df[share_label] = df[count_label] / df[count_label].sum()
            d[key] = df
        else:
            df = tableset[tablename].groupby(
                groupings + [nominal_col]
            ).size().rename(count_label).reset_index()
            df[share_label] = df[count_label] / df.groupby(groupings)[count_label].transform('sum')
            d[key] = df

    all_d = pd.concat(d, names=['source']).reset_index()

    selection = alt.selection_multi(
        fields=[nominal_col], bind='legend',
    )

    fig = alt.Chart(
        all_d
    ).mark_bar(
    ).encode(
        color=nominal_col,
        y=alt.Y('source', axis=alt.Axis(grid=False, title='')),
        x=alt.X(share_label, axis=alt.Axis(grid=False, labels=False, title=share_axis_label), scale=alt.Scale(domain=[0., 1.])),
        row=grouping,
        opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
        tooltip = [nominal_col, 'source', count_label, alt.Tooltip(f'{share_label}:Q', format='.2%')],
    ).add_selection(
        selection,
    )

    if title:
        fig = fig.properties(
            title=title
        ).configure_title(
            fontSize=20,
            anchor='start',
            color='black',
        )

    return fig
