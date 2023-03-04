from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from activitysim.core import workflow
from activitysim.core.contrast import altair as alt

logger = logging.getLogger(__name__)


def compare_histogram(
    states: dict[str, workflow.State],
    table_name,
    column_name,
    *,
    checkpoint_name=None,
    table_filter=None,
    grouping=None,
    bins: int | str = 10,
    bounds=(None, None),
    axis_label=None,
    interpolate="step",
    number_format=",.2f",
    title=None,
    tickCount=4,
    style="histogram",
    bandwidth=1,
    kde_support=100,
    relabel_states=None,
):
    """

    Parameters
    ----------
    states
    bins : int or str, default 10
        If an integer, then the range of data will be divided into this many
        bins.  If a string, no binning is undertaken, but the values are
        converted to this datatype, usually "int" to achieve the general effect
        of binning.
    relabel_states : Mapping[str,str]
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

    if relabel_states is None:
        relabel_states = {}

    if bins == "int" and number_format == ",.2f":
        number_format = ",d"

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
            try:
                df = df.query(table_filter)
            except NotImplementedError:
                # pandas.eval can't handle, try sharrow
                import sharrow as sh

                q = (
                    sh.DataTree(base=df)
                    .setup_flow({"out": table_filter})
                    .load(dtype=np.bool_)
                )
                df = df.loc[q]
        targets[key] = df[[column_name] + groupings]

    result = pd.concat(targets, names=["source"])
    if bounds[0] is not None:
        result = result[result[column_name] >= bounds[0]]
    if bounds[1] is not None:
        result = result[result[column_name] <= bounds[1]]
    lower_bound = result[column_name].min()
    upper_bound = result[column_name].max()
    if isinstance(bins, str):
        bin_width = 0
        result[column_name] = result[column_name].astype(bins)
    else:
        bin_width = (upper_bound - lower_bound) / bins
        if style == "histogram":
            result[column_name] = pd.cut(result[column_name], bins)
    targets = {k: result.loc[k] for k in targets.keys()}

    n = f"n_{table_name}"
    s = f"share_{table_name}"

    d = {}
    if style == "histogram":
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

            if bin_width:
                if groupings:
                    dummy = df.groupby(groupings).size().index.to_frame()
                else:
                    dummy = pd.DataFrame(index=[0])
                df[column_name] = df[column_name].apply(lambda x: x.mid)
                lower_edge = lower_bound - (bin_width / 2)
                upper_edge = upper_bound + (bin_width / 2)
                df = pd.concat(
                    [
                        dummy.assign(**{column_name: lower_edge, n: 0, s: 0}),
                        df,
                        dummy.assign(**{column_name: upper_edge, n: 0, s: 0}),
                    ]
                ).reset_index(drop=True)
            d[relabel_states.get(key, key)] = df
    elif style == "kde":
        for key, dat in targets.items():
            df, bw = _kde(dat[column_name], bandwidth=bandwidth, n=kde_support)
            d[relabel_states.get(key, key)] = df

    # This is sorted in reverse alphabetical order by source, so that
    # the stroke width for the first line plotted is fattest, and progressively
    # thinner lines are plotted over that, so all data is visible on the figure.
    all_d = (
        pd.concat(d, names=["source"])
        .reset_index()
        .sort_values("source", ascending=False)
    )

    if style == "histogram":
        if len(states) != 1:
            encode_kwds = dict(
                color="source",
                y=alt.Y(s, axis=alt.Axis(grid=False, title="")),
                x=alt.X(
                    f"{column_name}:Q" if bin_width else f"{column_name}:O",
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
                    alt.Tooltip(s, format=".2%"),
                ],
                strokeWidth="source",
            )
        else:
            encode_kwds = dict(
                color="source",
                y=alt.Y(s, axis=alt.Axis(grid=False, title="")),
                x=alt.X(
                    f"{column_name}:Q" if bin_width else f"{column_name}:O",
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
                    alt.Tooltip(s, format=".2%"),
                ],
            )
    elif style == "kde":
        if len(states) != 1:
            encode_kwds = dict(
                color="source",
                y=alt.Y("density", axis=alt.Axis(grid=False, title="")),
                x=alt.X(
                    f"{column_name}:Q",
                    axis=alt.Axis(
                        grid=False,
                        title=axis_label or column_name,
                        format=number_format,
                        tickCount=tickCount,
                    ),
                ),
                strokeWidth="source",
            )
        else:
            encode_kwds = dict(
                color="source",
                y=alt.Y("density", axis=alt.Axis(grid=False, title="")),
                x=alt.X(
                    f"{column_name}:Q",
                    axis=alt.Axis(
                        grid=False,
                        title=axis_label or column_name,
                        format=number_format,
                        tickCount=tickCount,
                    ),
                ),
            )
    else:
        raise ValueError(f"unknown {style=}")

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
            .mark_line(interpolate=interpolate)
            .encode(**encode_kwds)
            .properties(**properties_kwds)
        )
    else:
        if bin_width:
            fig = (
                alt.Chart(all_d)
                .mark_area(interpolate=interpolate)
                .encode(**encode_kwds)
                .properties(**properties_kwds)
            )
        else:
            fig = (
                alt.Chart(all_d)
                .mark_bar()
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


def _kde(values, n=5, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    from sklearn.neighbors import KernelDensity

    x = np.asarray(values)

    if isinstance(bandwidth, (float, int)):
        kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
        kde_skl.fit(x[:, np.newaxis])
    else:
        from sklearn.model_selection import GridSearchCV

        grid = GridSearchCV(
            KernelDensity(), {"bandwidth": bandwidth}, cv=3
        )  # 20-fold cross-validation
        grid.fit(x[:, None])
        bandwidth = grid.best_params_["bandwidth"]
        kde_skl = grid.best_estimator_

    x_grid = np.linspace(values.min(), values.max(), n)

    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    name = getattr(values, "name", "x")
    return (pd.DataFrame({name: x_grid, "density": np.exp(log_pdf)}), bandwidth)
