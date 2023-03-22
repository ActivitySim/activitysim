from __future__ import annotations

import os

import pandas as pd
import pytest

from activitysim.core import workflow
from activitysim.core.los import Network_LOS as los


def _initialize_pipeline(
    module: str,
    tables: dict[str, str],
    initialize_network_los: bool,
    load_checkpoint: str = None,
) -> workflow.State:
    test_dir = os.path.join("test", module)
    configs_dir = os.path.join(test_dir, "configs")
    data_dir = os.path.join(test_dir, "data")
    output_dir = os.path.join(test_dir, "output")

    state = workflow.State()
    state.initialize_filesystem(
        configs_dir=(configs_dir,),
        data_dir=(data_dir,),
        output_dir=output_dir,
    )

    assert not (load_checkpoint and tables)

    # Read in the input test dataframes
    for dataframe_name, idx_name in tables.items():
        df = pd.read_csv(
            os.path.join("test", module, "data", f"{dataframe_name}.csv"),
            index_col=idx_name,
        )
        state.add_table(dataframe_name, df)

    if initialize_network_los:
        net_los = los(state)
        net_los.load_data()
        state.add_injectable("network_los", net_los)

    # Add an output directory in current working directory if it's not already there
    try:
        os.makedirs("output")
    except FileExistsError:
        # directory already exists
        pass

    if load_checkpoint:
        state.checkpoint.restore(resume_after=load_checkpoint)
    else:
        # Add the dataframes to the pipeline
        state.checkpoint.open_store()
        state.checkpoint.add(module)

    # By convention, this method needs to yield something
    yield state

    state.close_pipeline()


@pytest.fixture(scope="module")
def initialize_pipeline(
    module: str, tables: dict[str, str], initialize_network_los: bool
) -> workflow.State:
    yield from _initialize_pipeline(module, tables, initialize_network_los)


@pytest.fixture(scope="module")
def reconnect_pipeline(
    module: str, initialize_network_los: bool, load_checkpoint: str
) -> workflow.State:
    yield from _initialize_pipeline(module, {}, initialize_network_los, load_checkpoint)
