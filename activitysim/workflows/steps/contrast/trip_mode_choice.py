
import logging
import pandas as pd
import altair as alt
from pypyr.context import Context

logger = logging.getLogger(__name__)


def run_step(context: Context) -> None:

    context.assert_key_has_value(key='report', caller=__name__)
    report = context.get('report')
    fig = context.get('fig')

    contrast_data = context.get('contrast_data')
    grouping = context.get_formatted('grouping')
    title = context.get_formatted('title')

    with report:
        report << fig(title)
        report << compare_trip_mode_choice(contrast_data, title=None, grouping=grouping)




def compare_trip_mode_choice(tablesets, title="Trip Mode Choice", grouping='primary_purpose'):

    d = {}
    groupings = [grouping,]

    for key, tableset in tablesets.items():
        df = tableset['trips'].groupby(
            groupings + ['trip_mode']
        ).size().rename('n_trips').reset_index()
        df['share_trips'] = df['n_trips'] / df.groupby(groupings)['n_trips'].transform('sum')
        d[key] = df

    all_d = pd.concat(d, names=['source']).reset_index()

    selection = alt.selection_multi(
        fields=['trip_mode'], bind='legend',
    )

    fig = alt.Chart(
        all_d
    ).mark_bar(
    ).encode(
        color='trip_mode',
        y=alt.Y('source', axis=alt.Axis(grid=False, title='')),
        x=alt.X('share_trips', axis=alt.Axis(grid=False, labels=False, title='Mode Share')),
        row=grouping,
        opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
        tooltip = ['trip_mode', 'source', 'n_trips', alt.Tooltip('share_trips:Q', format='.2%')],
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
