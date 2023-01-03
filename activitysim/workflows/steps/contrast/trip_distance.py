import logging

import altair as alt
import pandas as pd

from ..wrapping import workstep

logger = logging.getLogger(__name__)


@workstep
def trip_distance(
    tablesets,
    skims,
    dist_skim_name,
    dist_bins=20,
    grouping="primary_purpose",
    title=None,
    max_dist=None,
    relabel_tablesets=None,
    tablename="trips",
):
    """

    Parameters
    ----------
    tablesets
    skims
    dist_skim_name
    dist_bins
    grouping
    title
    max_dist
    relabel_tablesets : Mapping[str,str]
        Remap the keys in `tablesets` with these values. Any
        missing values are retained.  This allows you to modify
        the figure to e.g. change "reference" to "v1.0.4" without
        editing the original input data.

    Returns
    -------
    altair.Chart
    """
    if relabel_tablesets is None:
        relabel_tablesets = {}

    groupings = [grouping]
    if not isinstance(skims, dict):
        skims = {i: skims for i in tablesets.keys()}

    distances = {}
    for key, tableset in tablesets.items():
        distances[key] = tableset[tablename][dist_skim_name]

    if dist_bins is not None:
        result = pd.concat(distances, names=["source"])
        if max_dist is not None:
            result = result[result <= max_dist]
        result = pd.cut(result, dist_bins).to_frame()
        distances = {k: result.loc[k] for k in tablesets.keys()}

    data = {}
    for key, tableset in tablesets.items():
        data[key] = tableset[tablename].assign(**{"distance": distances[key]})

    d = {}
    for key, dat in data.items():
        df = (
            dat.groupby(groupings + ["distance"])
            .size()
            .rename("n_trips")
            .unstack("distance")
            .fillna(0)
            .stack()
            .rename("n_trips")
            .reset_index()
        )
        df["share_trips"] = df["n_trips"] / df.groupby(groupings)["n_trips"].transform(
            "sum"
        )
        d[relabel_tablesets.get(key, key)] = df

    # This is sorted in reverse alphabetical order by source, so that
    # the stroke width for the first line plotted is fattest, and progressively
    # thinner lines are plotted over that, so all data is visible on the figure.
    all_d = (
        pd.concat(d, names=["source"])
        .reset_index()
        .sort_values("source", ascending=False)
    )
    all_d["distance"] = all_d["distance"].apply(lambda x: x.mid)

    fig = (
        alt.Chart(all_d)
        .mark_line(
            interpolate="monotone",
        )
        .encode(
            color="source",
            y=alt.Y("share_trips", axis=alt.Axis(grid=False, title="")),
            x=alt.X("distance", axis=alt.Axis(grid=False, title="Distance")),
            # opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
            # tooltip = ['trip_mode', 'source', 'n_trips', alt.Tooltip('share_trips:Q', format='.2%')],
            facet=alt.Facet(grouping, columns=3),
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
