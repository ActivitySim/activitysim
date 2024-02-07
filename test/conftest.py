from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from activitysim.core import workflow
from activitysim.core.los import Network_LOS as los


@pytest.fixture(scope="module")
def tmp_path_module(request, tmp_path_factory):
    """A tmpdir fixture for the module scope. Persists throughout the module."""
    return tmp_path_factory.mktemp(request.module.__name__)


def _initialize_pipeline(
    module: str,
    tables: dict[str, str],
    initialize_network_los: bool,
    load_checkpoint: str = None,
    *,
    prepared_module_inputs: Path,
) -> workflow.State:
    local_dir = Path(__file__).parent

    module_test_dir = local_dir.joinpath(module)
    configs_dir = module_test_dir.joinpath("configs")
    data_dir = module_test_dir.joinpath("data")
    output_dir = module_test_dir.joinpath("output")

    state = workflow.State.make_default(
        configs_dir=(configs_dir,),
        data_dir=(data_dir,),
        output_dir=output_dir,
    )

    assert not (load_checkpoint and tables)

    # Read in the input test dataframes
    for dataframe_name, idx_name in tables.items():
        if prepared_module_inputs.joinpath(f"{dataframe_name}.csv").exists():
            df = pd.read_csv(
                prepared_module_inputs.joinpath(f"{dataframe_name}.csv"),
                index_col=idx_name,
            )
        elif prepared_module_inputs.joinpath(f"{dataframe_name}.csv.gz").exists():
            df = pd.read_csv(
                prepared_module_inputs.joinpath(f"{dataframe_name}.csv.gz"),
            )
            try:
                df = df.set_index(idx_name)
            except Exception:
                print(df.info(1))
                raise
        else:
            raise FileNotFoundError(data_dir.joinpath(f"{dataframe_name}.csv"))
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
    module: str,
    tables: dict[str, str],
    initialize_network_los: bool,
    prepare_module_inputs: Path,
) -> workflow.State:
    yield from _initialize_pipeline(
        module,
        tables,
        initialize_network_los,
        prepared_module_inputs=prepare_module_inputs,
    )


@pytest.fixture(scope="module")
def reconnect_pipeline(
    module: str,
    initialize_network_los: bool,
    load_checkpoint: str,
    prepare_module_inputs: Path,
) -> workflow.State:
    yield from _initialize_pipeline(
        module,
        {},
        initialize_network_los,
        load_checkpoint,
        prepared_module_inputs=prepare_module_inputs,
    )
