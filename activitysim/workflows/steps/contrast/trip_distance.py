import logging
import pandas as pd
import altair as alt
from pypyr.context import Context
from ..progression import reset_progress_step

logger = logging.getLogger(__name__)


def run_step(context: Context) -> None:

    context.assert_key_has_value(key='report', caller=__name__)
    report = context.get('report')
    fig = context.get('fig')

    contrast_data = context.get('contrast_data')
    skims = context.get('skims')
    dist_skim_name = context.get_formatted('dist_skim_name')
    grouping = context.get_formatted('grouping')
    title = context.get_formatted('title') or "Trip Length Distribution"
    title_level = context.get('title_level', None)

    otaz_col = context.get_formatted('otaz_col')
    dtaz_col = context.get_formatted('dtaz_col')
    time_col = context.get_formatted('time_col')
    dist_bins = context.get_formatted('dist_bins')
    max_dist = context.get_formatted_or_default('max_dist', None)

    reset_progress_step(description=f"report trip mode choice / {grouping}")

    with report:
        report << fig(title, level=title_level)
        report << compare_trip_distance(
            contrast_data,
            skims,
            dist_skim_name,
            otaz_col=otaz_col,
            dtaz_col=dtaz_col,
            time_col=time_col,
            dist_bins=dist_bins,
            grouping=grouping,
            title=None,
            max_dist=max_dist,
        )





def compare_trip_distance(
    tablesets,
    skims,
    dist_skim_name,
    otaz_col='origin',
    dtaz_col='destination',
    time_col='depart',
    dist_bins=20,
    grouping='primary_purpose',
    title="Trip Length Distribution",
    max_dist=None,
):
    groupings = [grouping]
    if not isinstance(skims, dict):
        skims = {i: skims for i in tablesets.keys()}

    distances = {}
    for key, tableset in tablesets.items():
        skim_dist = skims[key][[dist_skim_name]]

        zone_ids = tableset['land_use'].index
        if zone_ids.is_monotonic_increasing and zone_ids[-1] == len(zone_ids) + zone_ids[0] - 1:
            offset = zone_ids[0]
            looks = [
                tableset['trips'][otaz_col].rename('otaz') - offset,
                tableset['trips'][dtaz_col].rename('dtaz') - offset,
            ]
        else:
            remapper = dict(zip(zone_ids, pd.RangeIndex(len(zone_ids))))
            looks = [
                tableset['trips'][otaz_col].rename('otaz').apply(remapper.get),
                tableset['trips'][dtaz_col].rename('dtaz').apply(remapper.get),
            ]
        if 'time_period' in skim_dist.dims:
            looks.append(
                tableset['trips'][time_col].apply(skims[key].attrs['time_period_imap'].get).rename('time_period'),
            )
        look = pd.concat(looks, axis=1)
        distances[key] = skims[key][[dist_skim_name]].iat.df(look)

    if dist_bins is not None:
        result = pd.concat(distances, names=['source'])
        if max_dist is not None:
            result = result[result <= max_dist]
        result = pd.cut(result.iloc[:, 0], dist_bins).to_frame()
        distances = {k:result.loc[k] for k in tablesets.keys()}

    data = {}
    for key, tableset in tablesets.items():
        data[key] = tableset['trips'].assign(**{'distance': distances[key]})

    d = {}
    for key, dat in data.items():
        df = dat.groupby(
            groupings + ['distance']
        ).size().rename('n_trips').unstack('distance').fillna(0).stack().rename('n_trips').reset_index()
        df['share_trips'] = df['n_trips'] / df.groupby(groupings)['n_trips'].transform('sum')
        d[key] = df

    all_d = pd.concat(d, names=['source']).reset_index()
    all_d['distance'] = all_d['distance'].apply(lambda x: x.mid)

    fig = alt.Chart(
        all_d
    ).mark_line(
        interpolate='monotone',
    ).encode(
        color='source',
        y=alt.Y('share_trips', axis=alt.Axis(grid=False, title='')),
        x=alt.X('distance', axis=alt.Axis(grid=False, title='Distance')),
        #opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
        #tooltip = ['trip_mode', 'source', 'n_trips', alt.Tooltip('share_trips:Q', format='.2%')],
        facet=alt.Facet(grouping, columns=3),
        strokeWidth = 'source',
    ).properties(
        width=200,
        height=120,
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
