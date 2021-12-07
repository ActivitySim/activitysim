# ActivitySim
# See full license in LICENSE.txt.
import os
import logging
import pandas as pd

from activitysim.core import pipeline
from activitysim.core import inject
from activitysim.core import config

from activitysim.abm.models.trip_matrices import annotate_trips

logger = logging.getLogger(__name__)


@inject.step()
def summarize(network_los, persons_merged):
    """
    summarize is a standard model which uses expression files
    to reduce tables
    """
    trace_label = 'summarize'
    model_settings_file_name = 'summarize.yaml'
    model_settings = config.read_model_settings(model_settings_file_name)

    output_location = model_settings['OUTPUT'] if 'OUTPUT' in model_settings  else 'summaries'
    os.makedirs(config.output_file_path(output_location), exist_ok=True)

    persons_merged = persons_merged.to_frame()

    for table_name, meta in model_settings['TABLES'].items():
        if table_name == 'persons_merged':
            df = persons_merged
        else:
            df = inject.get_table(table_name).to_frame()

        if 'AGGREGATE' in meta and meta['AGGREGATE']:
            for agg in meta['AGGREGATE']:
                assert set(('column', 'label', 'map')) <= agg.keys()
                df[agg['label']] = df[agg['column']].map(agg['map']).fillna(df[agg['column']])

        if 'SLICERS' in meta and meta['SLICERS']:
            for slicer in meta['SLICERS']:
                df[slicer['label']] = pd.cut(df[slicer['column']], slicer['bins'],
                                             labels=slicer['bin_labels'], include_lowest=True)

        # Get merged trips and annotate them
        # model_settings = config.read_model_settings('write_trip_matrices.yaml')
        # trips = inject.get_table('trips_merged', None)
        # locals_d['trips_merged'] = annotate_trips(trips, network_los, model_settings)

        # locals_d['persons'] = inject.get_table('persons_merged', None).to_frame()

        # Save table for testing
        # locals_d['trips'].to_csv(config.output_file_path(os.path.join(output_location, f'trips_merged.csv')))
        # locals_d['persons'].to_csv(config.output_file_path(os.path.join(output_location, f'persons_merged.csv')))

        if 'SPECIFICATION' in meta:
            spec = pd.read_csv(config.config_file_path(meta['SPECIFICATION']))

            locals_d = {
                table_name: df,
                'persons_merged': persons_merged
            }

            for i, row in spec.iterrows():

                out_file = row['Output']
                expr = row['Expression']

                logger.info(f'Summary: {expr} -> {out_file}.csv')

                resultset = eval(expr, globals(), locals_d)
                resultset.to_csv(config.output_file_path(os.path.join(output_location, f'{out_file}.csv')), index=False)

