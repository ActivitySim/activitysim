# ActivitySim
# See full license in LICENSE.txt.
import os
import logging
import pandas as pd

from activitysim.core import pipeline
from activitysim.core import inject
from activitysim.core import config

logger = logging.getLogger(__name__)

@inject.step()
def summarize():
    """
    summarize is a standard model which uses expression files
    to reduce tables
    """
    trace_label = 'summarize'
    model_settings_file_name = 'summarize.yaml'
    model_settings = config.read_model_settings(model_settings_file_name)

    for segment in model_settings['SPEC_SEGMENTS']:
        table = segment['table']
        spec_name = segment['spec']
        output_location = segment['output'] if 'output' in segment else 'summaries'
        os.makedirs(config.output_file_path(output_location), exist_ok=True)

        # Go get specified tables from the pipeline unless they are supplied as a parameter
        locals_d = {
            table: pipeline.get_table(table)
        }

        spec = pd.read_csv(config.config_file_path(spec_name))

        for i, row in spec.iterrows():

            out_file = row['Output']
            expr = row['Expression']

            resultset = eval(expr, globals(), locals_d)
            resultset.to_csv(config.output_file_path(os.path.join(output_location, f'{out_file}.csv')))

