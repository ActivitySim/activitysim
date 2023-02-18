import os
from pathlib import Path

import pandas as pd
import pytest

from activitysim.core import los, workflow


@pytest.fixture(scope="module")
def initialize_pipeline(
    module: str,
    tables: dict[str, str],
    initialize_network_los: bool,
    base_dir: Path,
) -> workflow.Whale:
    if base_dir is None:
        base_dir = Path("test").joinpath(module)
    configs_dir = base_dir.joinpath("configs")
    data_dir = base_dir.joinpath("data")
    output_dir = base_dir.joinpath("output")

    whale = (
        workflow.Whale()
        .initialize_filesystem(
            configs_dir=configs_dir,
            data_dir=data_dir,
            output_dir=output_dir,
        )
        .load_settings()
    )

    # Read in the input test dataframes
    for dataframe_name, idx_name in tables.items():
        df = pd.read_csv(
            data_dir.joinpath(f"{dataframe_name}.csv"),
            index_col=idx_name,
        )
        whale.add_table(dataframe_name, df)

    if initialize_network_los:
        net_los = los.Network_LOS(whale)
        net_los.load_data()
        whale.add_injectable("network_los", net_los)

    # Add the dataframes to the pipeline
    whale.open_pipeline()
    whale.checkpoint.add(module)
    whale.close_pipeline()

    # By convention, this method needs to yield something
    yield whale

    # pytest teardown code
    whale.close_pipeline()
    pipeline_file_path = os.path.join(output_dir, "pipeline.h5")
    if os.path.exists(pipeline_file_path):
        os.unlink(pipeline_file_path)
