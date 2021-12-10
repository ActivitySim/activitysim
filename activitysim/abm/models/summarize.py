# ActivitySim
# See full license in LICENSE.txt.
import os
import logging
import pandas as pd

from activitysim.core import pipeline
from activitysim.core import expressions
from activitysim.core import inject
from activitysim.core import config

from activitysim.abm.models.trip_matrices import annotate_trips

logger = logging.getLogger(__name__)


def wrap_skims(network_los, trips_merged):
    skim_dict = network_los.get_default_skim_dict()

    trips_merged['start_tour_period'] = network_los.skim_time_period_label(trips_merged['start'])
    trips_merged['end_tour_period'] = network_los.skim_time_period_label(trips_merged['end'])
    trips_merged['trip_period'] = network_los.skim_time_period_label(trips_merged['depart'])

    tour_odt_skim_stack_wrapper = skim_dict.wrap_3d(
        orig_key='origin_tour', dest_key='destination_tour', dim3_key='start_tour_period')
    tour_dot_skim_stack_wrapper = skim_dict.wrap_3d(
        orig_key='destination_tour', dest_key='origin_tour', dim3_key='end_tour_period')
    trip_odt_skim_stack_wrapper = skim_dict.wrap_3d(
        orig_key='origin_trip', dest_key='destination_trip', dim3_key='trip_period')

    tour_od_skim_stack_wrapper = skim_dict.wrap('origin_tour', 'destination_tour')
    trip_od_skim_stack_wrapper = skim_dict.wrap('origin_trip', 'destination_trip')

    return {
        "tour_odt_skims": tour_odt_skim_stack_wrapper,
        "tour_dot_skims": tour_dot_skim_stack_wrapper,
        "trip_odt_skims": trip_odt_skim_stack_wrapper,
        "tour_od_skims": tour_od_skim_stack_wrapper,
        "trip_od_skims": trip_od_skim_stack_wrapper,
    }

@inject.step()
def summarize(network_los, persons_merged, trips, tours_merged):
    """
    summarize is a standard model which uses expression files
    to reduce tables
    """
    trace_label = 'summarize'
    model_settings_file_name = 'summarize.yaml'
    model_settings = config.read_model_settings(model_settings_file_name)

    output_location = model_settings['OUTPUT'] if 'OUTPUT' in model_settings  else 'summaries'
    os.makedirs(config.output_file_path(output_location), exist_ok=True)

    spec = pd.read_csv(config.config_file_path(model_settings['SPECIFICATION']), comment='#')

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
        how="left")

    locals_d = {
        'trips_merged': trips_merged,
        'persons_merged': persons_merged
    }

    for table_name, df in locals_d.items():
        meta = model_settings[table_name]
        df = eval(table_name)

        if 'AGGREGATE' in meta and meta['AGGREGATE']:
            for agg in meta['AGGREGATE']:
                assert set(('column', 'label', 'map')) <= agg.keys()
                df[agg['label']] = df[agg['column']].map(agg['map']).fillna(df[agg['column']])

        if 'SLICERS' in meta and meta['SLICERS']:
            for slicer in meta['SLICERS']:
                df[slicer['label']] = pd.cut(df[slicer['column']], slicer['bin_breaks'],
                                             labels=slicer['bin_labels'], include_lowest=True)

        # Get merged trips and annotate them
        # model_settings = config.read_model_settings('write_trip_matrices.yaml')
        # trips = inject.get_table('trips_merged', None)
        # locals_d['trips_merged'] = annotate_trips(trips, network_los, model_settings)

        # locals_d['persons'] = inject.get_table('persons_merged', None).to_frame()

    skims = wrap_skims(network_los,trips_merged)

    expressions.annotate_preprocessors(trips_merged, locals_d, skims, model_settings, 'summarize')

    locals_d.update(skims)

    for i, row in spec.iterrows():

        out_file = row['Output']
        expr = row['Expression']

        logger.info(f'Summary: {expr} -> {out_file}.csv')

        resultset = eval(expr, globals(), locals_d)
        resultset.to_csv(config.output_file_path(os.path.join(output_location, f'{out_file}.csv')), index=False)

