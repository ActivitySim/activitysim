# ActivitySim
# See full license in LICENSE.txt.
import logging
import os

import numpy as np
import pandas as pd
from activitysim.abm.models.trip_matrices import annotate_trips
from activitysim.core import config, expressions, inject, pipeline

logger = logging.getLogger(__name__)


def wrap_skims(
    network_los: pipeline.Pipeline, trips_merged: pd.DataFrame
) -> dict[str, object]:
    skim_dict = network_los.get_default_skim_dict()

    trips_merged['start_tour_period'] = network_los.skim_time_period_label(
        trips_merged['start']
    )
    trips_merged['end_tour_period'] = network_los.skim_time_period_label(
        trips_merged['end']
    )
    trips_merged['trip_period'] = network_los.skim_time_period_label(
        trips_merged['depart']
    )

    tour_odt_skim_stack_wrapper = skim_dict.wrap_3d(
        orig_key='origin_tour',
        dest_key='destination_tour',
        dim3_key='start_tour_period',
    )
    tour_dot_skim_stack_wrapper = skim_dict.wrap_3d(
        orig_key='destination_tour', dest_key='origin_tour', dim3_key='end_tour_period'
    )
    trip_odt_skim_stack_wrapper = skim_dict.wrap_3d(
        orig_key='origin_trip', dest_key='destination_trip', dim3_key='trip_period'
    )

    tour_od_skim_stack_wrapper = skim_dict.wrap('origin_tour', 'destination_tour')
    trip_od_skim_stack_wrapper = skim_dict.wrap('origin_trip', 'destination_trip')

    return {
        "tour_odt_skims": tour_odt_skim_stack_wrapper,
        "tour_dot_skims": tour_dot_skim_stack_wrapper,
        "trip_odt_skims": trip_odt_skim_stack_wrapper,
        "tour_od_skims": tour_od_skim_stack_wrapper,
        "trip_od_skims": trip_od_skim_stack_wrapper,
    }


DEFAULT_BIN_LABEL_FORMAT = "{left:,.2f} - {right:,.2f}"


def construct_bin_labels(bins: pd.Series, label_format: str) -> pd.Series:
    left = bins.apply(lambda x: x.left)
    mid = bins.apply(lambda x: x.mid)
    right = bins.apply(lambda x: x.right)
    # Get integer ranks of bins (e.g., 1st, 2nd ... nth quantile)
    rank = mid.map(
        {
            x: sorted(mid.unique().tolist()).index(x) + 1 if pd.notnull(x) else np.nan
            for x in mid.unique()
        },
        na_action='ignore',
    )

    def construct_label(label_format, bounds_dict):
        # parts = [part for part in ['left', 'right'] if part in label_format]
        bounds_dict = {
            x: bound for x, bound in bounds_dict.items() if x in label_format
        }
        return label_format.format(**bounds_dict)

    labels = pd.Series(
        [
            construct_label(label_format, {'left': lt, 'mid': md, 'right': rt, 'rank': rk})
            for lt, md, rt, rk in zip(left, mid, right, rank)
        ],
        index=bins.index,
    )
    # Convert to numeric if possible
    labels = pd.to_numeric(labels, errors='ignore')
    return labels


def quantiles(
    data: pd.Series, bins: pd.Series, label_format: str = DEFAULT_BIN_LABEL_FORMAT
) -> pd.Series:
    vals = data.sort_values()
    # qcut a ranking instead of raw values to deal with high frequencies of the same value
    # (e.g., many 0 values) that may span multiple bins
    ranks = vals.rank(method='first')
    bins = pd.qcut(ranks, bins, duplicates='drop')
    bins = construct_bin_labels(bins, label_format)
    return bins


def spaced_intervals(
    data: pd.Series,
    lower_bound: str,
    interval: float,
    label_format: str = DEFAULT_BIN_LABEL_FORMAT,
) -> pd.Series:
    if lower_bound == 'min':
        lower_bound = data.min()
    breaks = np.arange(lower_bound, data.max() + interval, interval)
    bins = pd.cut(data, breaks, include_lowest=True)
    bins = construct_bin_labels(bins, label_format)
    return bins


def equal_intervals(
    data: pd.Series, bins: int, label_format: str = DEFAULT_BIN_LABEL_FORMAT
) -> pd.Series:
    bins = pd.cut(data, bins, include_lowest=True)
    bins = construct_bin_labels(bins, label_format)
    return bins


def manual_breaks(
    data, bin_breaks, labels: list = None, label_format=DEFAULT_BIN_LABEL_FORMAT
) -> pd.Series:
    if isinstance(labels, list):
        return pd.cut(data, bin_breaks, labels=labels, include_lowest=True)
    else:
        bins = pd.cut(data, bin_breaks, include_lowest=True)
        bins = construct_bin_labels(bins, label_format)
        return bins


@inject.step()
def summarize(
    network_los: pipeline.Pipeline,
    persons_merged: pd.DataFrame,
    trips: pd.DataFrame,
    tours_merged: pd.DataFrame,
):
    """
    summarize is a standard model which uses expression files
    to reduce tables
    """
    trace_label = 'summarize'
    model_settings_file_name = 'summarize.yaml'
    model_settings = config.read_model_settings(model_settings_file_name)

    output_location = (
        model_settings['OUTPUT'] if 'OUTPUT' in model_settings else 'summaries'
    )
    os.makedirs(config.output_file_path(output_location), exist_ok=True)

    spec = pd.read_csv(
        config.config_file_path(model_settings['SPECIFICATION']), comment='#'
    )

    persons_merged = persons_merged.to_frame()
    trips = trips.to_frame()
    tours_merged = tours_merged.to_frame()

    # - trips_merged - merge trips and tours_merged
    trips_merged = pd.merge(
        trips,
        tours_merged.drop(columns=['person_id', 'household_id']),
        left_on='tour_id',
        right_index=True,
        suffixes=['_trip', '_tour'],
        how="left",
    )

    locals_d = {'trips_merged': trips_merged, 'persons_merged': persons_merged}

    skims = wrap_skims(network_los, trips_merged)

    expressions.annotate_preprocessors(
        trips_merged, locals_d, skims, model_settings, 'summarize'
    )

    for table_name, df in locals_d.items():
        meta = model_settings[table_name]
        df = eval(table_name)

        if 'AGGREGATE' in meta and meta['AGGREGATE']:
            for agg in meta['AGGREGATE']:
                assert set(('column', 'label', 'map')) <= agg.keys()
                df[agg['label']] = (
                    df[agg['column']].map(agg['map']).fillna(df[agg['column']])
                )

        if 'SLICERS' in meta and meta['SLICERS']:
            for slicer in meta['SLICERS']:
                if slicer['type'] == 'manual_breaks':
                    # df[slicer['label']] = pd.cut(df[slicer['column']], slicer['bin_breaks'],
                    #                              labels=slicer['bin_labels'], include_lowest=True)
                    df[slicer['label']] = manual_breaks(
                        df[slicer['column']], slicer['bin_breaks'], slicer['bin_labels']
                    )

                elif slicer['type'] == 'quantiles':
                    df[slicer['label']] = quantiles(
                        df[slicer['column']], slicer['bins'], slicer['label_format']
                    )

                elif slicer['type'] == 'spaced_intervals':
                    df[slicer['label']] = spaced_intervals(
                        df[slicer['column']],
                        slicer['lower_bound'],
                        slicer['interval'],
                        slicer['label_format'],
                    )

                elif slicer['type'] == 'equal_intervals':
                    df[slicer['label']] = equal_intervals(
                        df[slicer['column']], slicer['bins'], slicer['label_format']
                    )

    locals_d.update(skims)

    # Add classification functions to locals
    locals_d.update(
        {
            'quantiles': quantiles,
            'spaced_intervals': spaced_intervals,
            'equal_intervals': equal_intervals,
            'manual_breaks': manual_breaks,
        }
    )

    # Save merged tables for expression development
    locals_d['trips_merged'].to_csv(
        config.output_file_path(os.path.join(output_location, f'trips_merged.csv'))
    )
    locals_d['persons_merged'].to_csv(
        config.output_file_path(os.path.join(output_location, f'persons_merged.csv'))
    )

    for i, row in spec.iterrows():

        out_file = row['Output']
        expr = row['Expression']

        logger.debug(f'Summary: {expr} -> {out_file}.csv')

        resultset = eval(expr, globals(), locals_d)
        resultset.to_csv(
            config.output_file_path(os.path.join(output_location, f'{out_file}.csv')),
            index=False,
        )
