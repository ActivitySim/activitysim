# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging
import os

import pytest
import tables

from activitysim.core import workflow
from activitysim.core.test.extensions import steps

# set the max households for all tests (this is to limit memory use on travis)
HOUSEHOLDS_SAMPLE_SIZE = 100
HH_ID = 961042


@pytest.fixture
def state():
    configs_dir = os.path.join(os.path.dirname(__file__), "configs")
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    data_dir = os.path.join(os.path.dirname(__file__), "data")

    state = (
        workflow.State()
        .initialize_filesystem(
            configs_dir=(configs_dir,),
            output_dir=output_dir,
            data_dir=(data_dir,),
        )
        .load_settings()
    )

    state.logging.config_logger()
    return state


def close_handlers():
    loggers = logging.Logger.manager.loggerDict
    for name in loggers:
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True
        logger.setLevel(logging.NOTSET)


# @pytest.mark.filterwarnings('ignore::tables.NaturalNameWarning')
def test_pipeline_run(state):
    # workflow.step(steps.step1, step_name="step1")
    # workflow.step(steps.step2, step_name="step2")
    # workflow.step(steps.step3, step_name="step3")
    # workflow.step(steps.step_add_col, step_name="step_add_col")

    _MODELS = [
        "step1",
        "step2",
        "step3",
        "step_add_col.table_name=table2;column_name=c2",
    ]

    state.run(models=_MODELS, resume_after=None)

    checkpoints = state.checkpoint.get_inventory()
    print("checkpoints\n", checkpoints)

    c2 = state.checkpoint.load_dataframe("table2").c2

    # get table from
    state.checkpoint.load_dataframe("table1", checkpoint_name="step3")

    # try to get a table from a step before it was checkpointed
    with pytest.raises(RuntimeError) as excinfo:
        state.checkpoint.load_dataframe("table2", checkpoint_name="step1")
    assert "not in checkpoint 'step1'" in str(excinfo.value)

    # try to get a non-existant table
    with pytest.raises(RuntimeError) as excinfo:
        state.checkpoint.load_dataframe("bogus")
    assert "never checkpointed" in str(excinfo.value)

    # try to get an existing table from a non-existant checkpoint
    with pytest.raises(RuntimeError) as excinfo:
        state.checkpoint.load_dataframe("table1", checkpoint_name="bogus")
    assert "not in checkpoints" in str(excinfo.value)

    state.checkpoint.close_store()

    close_handlers()


def test_pipeline_checkpoint_drop(state):
    # workflow.step(steps.step1, step_name="step1")
    # workflow.step(steps.step2, step_name="step2")
    # workflow.step(steps.step3, step_name="step3")
    # workflow.step(steps.step_add_col, step_name="step_add_col")
    # workflow.step(steps.step_forget_tab, step_name="step_forget_tab")

    _MODELS = [
        "step1",
        "_step2",
        "_step_add_col.table_name=table2;column_name=c2",
        "_step_forget_tab.table_name=table2",
        "step3",
        "step_forget_tab.table_name=table3",
    ]
    state.run(models=_MODELS, resume_after=None)

    checkpoints = state.checkpoint.get_inventory()
    print("checkpoints\n", checkpoints)

    state.checkpoint.load_dataframe("table1")

    with pytest.raises(RuntimeError) as excinfo:
        state.checkpoint.load_dataframe("table2")
    # assert "never checkpointed" in str(excinfo.value)

    # can't get a dropped table from current checkpoint
    with pytest.raises(RuntimeError) as excinfo:
        state.checkpoint.load_dataframe("table3")
    # assert "was dropped" in str(excinfo.value)

    # ensure that we can still get table3 from a checkpoint at which it existed
    state.checkpoint.load_dataframe("table3", checkpoint_name="step3")

    state.checkpoint.close_store()
    close_handlers()


# if __name__ == "__main__":
#
#     print "\n\ntest_pipeline_run"
#     test_pipeline_run()
#     teardown_function(None)
#
#     print "\n\ntest_pipeline_checkpoint_drop"
#     test_pipeline_checkpoint_drop()
#     teardown_function(None)
