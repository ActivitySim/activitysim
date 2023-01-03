import os
import warnings

import altair as alt
import pandas as pd

from .data_dictionary import check_data_dictionary
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
        )[1]
        for key, pth in pipelines.items()
    }


def load_final_tables(output_dirs, tables=None, index_cols=None):
    result = {}
    for key, pth in output_dirs.items():
        if not os.path.exists(pth):
            warnings.warn(f"{key} directory does not exist: {pth}")
            continue
        result[key] = {}
        for tname, tfile in tables.items():
            tpath = os.path.join(pth, tfile)
            kwargs = {}
            if index_cols is not None and tname in index_cols:
                kwargs["index_col"] = index_cols[tname]
            if os.path.exists(tpath):
                result[key][tname] = pd.read_csv(tpath, **kwargs)
        if len(result[key]) == 0:
            # no tables were loaded, delete the entire group
            del result[key]
    return result


def compare_trip_mode_choice(
    tablesets, title="Trip Mode Choice", grouping="primary_purpose"
):

    d = {}
    groupings = [
        grouping,
    ]

    for key, tableset in tablesets.items():
        df = (
            tableset["trips"]
            .groupby(groupings + ["trip_mode"])
            .size()
            .rename("n_trips")
            .reset_index()
        )
        df["share_trips"] = df["n_trips"] / df.groupby(groupings)["n_trips"].transform(
            "sum"
        )
        d[key] = df

    all_d = pd.concat(d, names=["source"]).reset_index()

    selection = alt.selection_multi(
        fields=["trip_mode"],
        bind="legend",
    )

    fig = (
        alt.Chart(all_d)
        .mark_bar()
        .encode(
            color="trip_mode",
            y=alt.Y("source", axis=alt.Axis(grid=False, title=""), sort=None),
            x=alt.X(
                "share_trips",
                axis=alt.Axis(grid=False, labels=False, title="Mode Share"),
            ),
            row="primary_purpose",
            opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
            tooltip=[
                "trip_mode",
                "source",
                "n_trips",
                alt.Tooltip("share_trips:Q", format=".2%"),
            ],
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

    return fig


def compare_trip_distance(
    tablesets,
    skims,
    dist_skim_name,
    otaz_col="origin",
    dtaz_col="destination",
    time_col="depart",
    dist_bins=20,
    grouping="primary_purpose",
    title="Trip Length Distribution",
    max_dist=None,
):
    groupings = [grouping]
    if not isinstance(skims, dict):
        skims = {i: skims for i in tablesets.keys()}

    distances = {}
    for key, tableset in tablesets.items():
        skim_dist = skims[key][[dist_skim_name]]

        zone_ids = tableset["land_use"].index
        if (
            zone_ids.is_monotonic_increasing
            and zone_ids[-1] == len(zone_ids) + zone_ids[0] - 1
        ):
            offset = zone_ids[0]
            looks = [
                tableset["trips"][otaz_col].rename("otaz") - offset,
                tableset["trips"][dtaz_col].rename("dtaz") - offset,
            ]
        else:
            remapper = dict(zip(zone_ids, pd.RangeIndex(len(zone_ids))))
            looks = [
                tableset["trips"][otaz_col].rename("otaz").apply(remapper.get),
                tableset["trips"][dtaz_col].rename("dtaz").apply(remapper.get),
            ]
        if "time_period" in skim_dist.dims:
            looks.append(
                tableset["trips"][time_col]
                .apply(skims[key].attrs["time_period_imap"].get)
                .rename("time_period"),
            )
        look = pd.concat(looks, axis=1)
        distances[key] = skims[key][[dist_skim_name]].iat.df(look)

    if dist_bins is not None:
        result = pd.concat(distances, names=["source"])
        if max_dist is not None:
            result = result[result <= max_dist]
        result = pd.cut(result.iloc[:, 0], dist_bins).to_frame()
        distances = {k: result.loc[k] for k in tablesets.keys()}

    data = {}
    for key, tableset in tablesets.items():
        data[key] = tableset["trips"].assign(**{"distance": distances[key]})

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
        d[key] = df

    all_d = pd.concat(d, names=["source"]).reset_index()
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


def compare_work_district(
    tablesets,
    district_id,
    label="district",
    hometaz_col="home_zone_id",
    worktaz_col="workplace_zone_id",
    data_dictionary=None,
):
    data_dictionary = check_data_dictionary(data_dictionary)

    d = {}
    h = f"home_{label}"
    w = f"work_{label}"

    for key, tableset in tablesets.items():
        persons = tableset["persons"]
        workers = persons[persons[worktaz_col] >= 0].copy()
        district_map = tableset["land_use"][district_id]
        # workers[f"home_{label}_"] = workers[hometaz_col].map(district_map)
        # workers[f"work_{label}_"] = workers[worktaz_col].map(district_map)
        home_district = workers[hometaz_col].map(district_map).rename(h)
        work_district = workers[worktaz_col].map(district_map).rename(w)
        df = (
            workers.groupby(
                [home_district, work_district]
                # [f"home_{label}_", f"work_{label}_"]
            )
            .size()
            .rename("n_workers")
        )
        d[key] = df

    all_d = pd.concat(d, names=["source"]).reset_index()

    district_names = data_dictionary.get("land_use", {}).get(district_id, None)
    if district_names is not None:
        all_d[h] = all_d[h].map(district_names)
        all_d[w] = all_d[w].map(district_names)

    selection = alt.selection_multi(
        fields=[w],
        bind="legend",
    )

    fig = (
        alt.Chart(all_d)
        .mark_bar()
        .encode(
            color=f"{w}:N",
            y=alt.Y("source", axis=alt.Axis(grid=False, title=""), sort=None),
            x=alt.X("n_workers", axis=alt.Axis(grid=False)),
            row=f"{h}:N",
            opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
            tooltip=[f"{h}:N", f"{w}:N", "source", "n_workers"],
        )
        .add_selection(
            selection,
        )
    )

    return fig


def compare_runtime(combo_timing_log):
    df = pd.read_csv(combo_timing_log, index_col="model_name")
    df1 = (
        df[["sharrow", "legacy"]]
        .rename_axis(columns="source")
        .unstack()
        .rename("seconds")
        .reset_index()
    )
    c = alt.Chart(
        df1,
        height={"step": 20},
    )

    result = c.mark_bar(yOffset=-3, size=6,).transform_filter(
        (alt.datum.source == "legacy")
    ).encode(
        x=alt.X("seconds:Q", stack=None),
        y=alt.Y("model_name", type="nominal", sort=None),
        color="source",
        tooltip=["source", "model_name", "seconds"],
    ) + c.mark_bar(
        yOffset=4,
        size=6,
    ).transform_filter(
        (alt.datum.source == "sharrow")
    ).encode(
        x=alt.X("seconds:Q", stack=None),
        y=alt.Y("model_name", type="nominal", sort=None),
        color="source",
        tooltip=["source", "model_name", "seconds"],
    ) | alt.Chart(
        df1
    ).mark_bar().encode(
        color="source", x="source", y="sum(seconds)", tooltip=["source", "sum(seconds)"]
    )

    return result
