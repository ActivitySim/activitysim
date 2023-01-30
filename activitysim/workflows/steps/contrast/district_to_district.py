import logging

import altair as alt
import pandas as pd
from pypyr.context import Context

from ....standalone.data_dictionary import check_data_dictionary
from ..progression import reset_progress_step
from ..wrapping import workstep

logger = logging.getLogger(__name__)


@workstep
def compare_district_to_district(
    tablesets,
    tablename,
    district_id,
    orig_col="home_zone_id",
    dest_col="workplace_zone_id",
    orig_label="home_district",
    dest_label="work_district",
    filter=None,
    data_dictionary=None,
    size_label="n_workers",
    viz_engine="altair",
):
    data_dictionary = check_data_dictionary(data_dictionary)

    d = {}

    for key, tableset in tablesets.items():
        if filter is not None:
            subtable = tableset[tablename].query(filter)
        else:
            subtable = tableset[tablename]
        district_map = tableset["land_use"][district_id]
        orig_district = subtable[orig_col].map(district_map).rename(orig_label)
        dest_district = subtable[dest_col].map(district_map).rename(dest_label)
        df = subtable.groupby([orig_district, dest_district]).size().rename(size_label)
        d[key] = df

    all_d = pd.concat(d, names=["source"]).reset_index()

    district_names = data_dictionary.get("land_use", {}).get(district_id, None)
    if district_names is not None:
        all_d[orig_label] = all_d[orig_label].map(district_names)
        all_d[dest_label] = all_d[dest_label].map(district_names)

    if viz_engine is None:
        return all_d

    selection = alt.selection_multi(
        fields=[dest_label],
        bind="legend",
    )

    fig = (
        alt.Chart(all_d)
        .mark_bar()
        .encode(
            color=f"{dest_label}:N",
            y=alt.Y("source", axis=alt.Axis(grid=False, title=""), sort=None),
            x=alt.X(size_label, axis=alt.Axis(grid=False)),
            row=f"{orig_label}:N",
            opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
            tooltip=[f"{orig_label}:N", f"{dest_label}:N", "source", size_label],
        )
        .add_selection(
            selection,
        )
    )

    return fig
