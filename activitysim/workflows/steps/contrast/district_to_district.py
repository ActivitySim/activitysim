import logging
import pandas as pd
import altair as alt
from pypyr.context import Context
from ..progression import reset_progress_step
from ....standalone.data_dictionary import check_data_dictionary
from ..wrapping import report_step

logger = logging.getLogger(__name__)


# def run_step(context: Context) -> None:
#
#     context.assert_key_has_value(key='report', caller=__name__)
#     report = context.get('report')
#     fig = context.get('fig')
#
#     contrast_data = context.get('contrast_data')
#     grouping = context.get_formatted('grouping')
#     title = context.get_formatted('title') or "Trip Length Distribution"
#     title_level = context.get('title_level', None)
#
#     tablename = context.get_formatted('tablename')
#     district_id = context.get_formatted('district_id')
#     filter = context.get_formatted('filter')
#     size_label = context.get_formatted('size_label')
#
#     reset_progress_step(description=f"report trip distance / {grouping}")
#
#     with report:
#         report << fig(title, level=title_level)
#         report << compare_district_to_district(
#                 contrast_data,
#                 tablename,
#                 district_id,
#                 orig_col='home_zone_id',
#                 dest_col='workplace_zone_id',
#                 orig_label='home_district',
#                 dest_label='work_district',
#                 filter=filter,
#                 data_dictionary=None,
#                 size_label=size_label,
#                 viz_engine='altair',
#         )
#


@report_step
def compare_district_to_district(
        tablesets,
        tablename,
        district_id,
        orig_col='home_zone_id',
        dest_col='workplace_zone_id',
        orig_label='home_district',
        dest_label='work_district',
        filter=None,
        data_dictionary=None,
        size_label='n_workers',
        viz_engine='altair',
):
    data_dictionary = check_data_dictionary(data_dictionary)

    d = {}

    for key, tableset in tablesets.items():
        if filter is not None:
            subtable = tableset[tablename].query(filter)
        else:
            subtable = tableset[tablename]
        district_map = tableset['land_use'][district_id]
        orig_district = subtable[orig_col].map(district_map).rename(orig_label)
        dest_district = subtable[dest_col].map(district_map).rename(dest_label)
        df = subtable.groupby(
            [orig_district, dest_district]
        ).size().rename(size_label)
        d[key] = df

    all_d = pd.concat(d, names=['source']).reset_index()

    district_names = data_dictionary.get('land_use', {}).get(district_id, None)
    if district_names is not None:
        all_d[orig_label] = all_d[orig_label].map(district_names)
        all_d[dest_label] = all_d[dest_label].map(district_names)

    if viz_engine is None:
        return all_d

    selection = alt.selection_multi(
        fields=[dest_label], bind='legend',
    )

    fig = alt.Chart(
        all_d
    ).mark_bar(
    ).encode(
        color=f'{dest_label}:N',
        y=alt.Y('source', axis=alt.Axis(grid=False, title='')),
        x=alt.X(size_label, axis=alt.Axis(grid=False)),
        row=f'{orig_label}:N',
        opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
        tooltip = [f'{orig_label}:N', f'{dest_label}:N', 'source', size_label],
    ).add_selection(
        selection,
    )

    return fig
