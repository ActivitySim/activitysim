import os
import pytest

import pandas as pd
from activitysim.core import pipeline
from activitysim.core.los import Network_LOS as los
import orca


@pytest.fixture(scope='module')
def initialize_pipeline(module, tables, initialize_network_los):
    # Read in the input test dataframes
    for dataframe_name in tables:
        df = pd.read_csv(os.path.join('data', module, f'{dataframe_name}.csv'))
        orca.add_table(dataframe_name, df)

    if initialize_network_los:
        net_los = los()
        net_los.load_data()
        orca.add_injectable('network_los', net_los)

    # Add the dataframes to the pipeline
    pipeline.open_pipeline()
    pipeline.add_checkpoint(module)
    pipeline.close_pipeline()

    # By convention, this method needs to yield something
    yield pipeline._PIPELINE

    # pytest teardown code
    pipeline.close_pipeline()
    pipeline_file_path = os.path.join('output', 'pipeline.h5')
    os.unlink(pipeline_file_path)
