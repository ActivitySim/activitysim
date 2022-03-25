import altair as alt
import pandas as pd
import os
from .pipeline import load_checkpointed_tables

def load_pipelines(pipelines, tables=None, checkpoint_name=None):
    """
    Parameters
    ----------
    pipelines : Dict[Str, Path-like]
        Mapping run name to path of pipeline file.
    checkpoint : str
        Name of checkpoint to load for all pipelines
    """
    return {
        key: load_checkpointed_tables(
            pth,
            tables=tables,
            checkpoint_name=checkpoint_name,
        )
        for key, pth in pipelines.items()
    }


def load_final_tables(output_dirs, tables=None, index_cols=None):
    result = {}
    for key, pth in output_dirs.items():
        result[key] = {}
        for tname, tfile in tables.items():
            tpath = os.path.join(pth, tfile)
            kwargs = {}
            if index_cols is not None and tname in index_cols:
                kwargs['index_col'] = index_cols[tname]
            result[key][tname] = pd.read_csv(tpath, **kwargs)
    return result


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

    selection = alt.selection_point(
        fields=['trip_mode'], bind='legend',
    )

    fig = alt.Chart(
        all_d
    ).mark_bar(
    ).encode(
        color='trip_mode',
        y=alt.Y('source', axis=alt.Axis(grid=False, title='')),
        x=alt.X('share_trips', axis=alt.Axis(grid=False, labels=False, title='Mode Share')),
        row='primary_purpose',
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
        looks = [
            tableset['trips'][otaz_col].rename('otaz'),
            tableset['trips'][dtaz_col].rename('dtaz'),
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

