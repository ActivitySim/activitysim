import os

import orca
import pandas as pd
import pytest

from activitysim.core import pipeline
from activitysim.core.los import Network_LOS as los


@pytest.fixture(scope="module")
def initialize_pipeline(
    module: str, tables: dict[str, str], initialize_network_los: bool
) -> pipeline.Pipeline:
    test_dir = os.path.join("test", module)
    configs_dir = os.path.join(test_dir, "configs")
    data_dir = os.path.join(test_dir, "data")
    output_dir = os.path.join(test_dir, "output")

    if os.path.isdir(configs_dir):
        orca.add_injectable("configs_dir", configs_dir)

    if os.path.isdir(data_dir):
        orca.add_injectable("data_dir", data_dir)

    if os.path.isdir(test_dir):
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        orca.add_injectable("output_dir", output_dir)

    # Read in the input test dataframes
    for dataframe_name, idx_name in tables.items():
        df = pd.read_csv(
            os.path.join("test", module, "data", f"{dataframe_name}.csv"),
            index_col=idx_name,
        )
        orca.add_table(dataframe_name, df)

    if initialize_network_los:
        net_los = los()
        net_los.load_data()
        orca.add_injectable("network_los", net_los)

    # Add the dataframes to the pipeline
    pipeline.open_pipeline()
    pipeline.add_checkpoint(module)
    pipeline.close_pipeline()

    # By convention, this method needs to yield something
    yield pipeline._PIPELINE

    # pytest teardown code
    pipeline.close_pipeline()
    pipeline_file_path = os.path.join(output_dir, "pipeline.h5")
    os.unlink(pipeline_file_path)
