# ActivitySim
# See full license in LICENSE.txt.
import logging
import os

import pytest
import tables

from activitysim.core import inject, pipeline, tracing

from .extensions import steps

# set the max households for all tests (this is to limit memory use on travis)
HOUSEHOLDS_SAMPLE_SIZE = 100
HH_ID = 961042


def setup_function():

    inject.reinject_decorated_tables()

    inject.remove_injectable("skim_dict")
    inject.remove_injectable("skim_stack")

    configs_dir = os.path.join(os.path.dirname(__file__), "configs")
    inject.add_injectable("configs_dir", configs_dir)

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    inject.add_injectable("output_dir", output_dir)

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    inject.add_injectable("data_dir", data_dir)

    inject.clear_cache()

    tracing.config_logger()


def teardown_function(func):
    inject.clear_cache()
    inject.reinject_decorated_tables()


def close_handlers():

    loggers = logging.Logger.manager.loggerDict
    for name in loggers:
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True
        logger.setLevel(logging.NOTSET)


# @pytest.mark.filterwarnings('ignore::tables.NaturalNameWarning')
def test_pipeline_run():

    inject.add_step("step1", steps.step1)
    inject.add_step("step2", steps.step2)
    inject.add_step("step3", steps.step3)
    inject.add_step("step_add_col", steps.step_add_col)
    inject.dump_state()

    _MODELS = [
        "step1",
        "step2",
        "step3",
        "step_add_col.table_name=table2;column_name=c2",
    ]

    pipeline.run(models=_MODELS, resume_after=None)

    checkpoints = pipeline.get_checkpoints()
    print("checkpoints\n", checkpoints)

    c2 = pipeline.get_table("table2").c2

    # get table from
    pipeline.get_table("table1", checkpoint_name="step3")

    # try to get a table from a step before it was checkpointed
    with pytest.raises(RuntimeError) as excinfo:
        pipeline.get_table("table2", checkpoint_name="step1")
    assert "not in checkpoint 'step1'" in str(excinfo.value)

    # try to get a non-existant table
    with pytest.raises(RuntimeError) as excinfo:
        pipeline.get_table("bogus")
    assert "never checkpointed" in str(excinfo.value)

    # try to get an existing table from a non-existant checkpoint
    with pytest.raises(RuntimeError) as excinfo:
        pipeline.get_table("table1", checkpoint_name="bogus")
    assert "not in checkpoints" in str(excinfo.value)

    pipeline.close_pipeline()

    close_handlers()


def test_pipeline_checkpoint_drop():

    inject.add_step("step1", steps.step1)
    inject.add_step("step2", steps.step2)
    inject.add_step("step3", steps.step3)
    inject.add_step("step_add_col", steps.step_add_col)
    inject.add_step("step_forget_tab", steps.step_forget_tab)

    _MODELS = [
        "step1",
        "_step2",
        "_step_add_col.table_name=table2;column_name=c2",
        "_step_forget_tab.table_name=table2",
        "step3",
        "step_forget_tab.table_name=table3",
    ]
    pipeline.run(models=_MODELS, resume_after=None)

    checkpoints = pipeline.get_checkpoints()
    print("checkpoints\n", checkpoints)

    pipeline.get_table("table1")

    with pytest.raises(RuntimeError) as excinfo:
        pipeline.get_table("table2")
    assert "never checkpointed" in str(excinfo.value)

    # can't get a dropped table from current checkpoint
    with pytest.raises(RuntimeError) as excinfo:
        pipeline.get_table("table3")
    assert "was dropped" in str(excinfo.value)

    # ensure that we can still get table3 from a checkpoint at which it existed
    pipeline.get_table("table3", checkpoint_name="step3")

    pipeline.close_pipeline()
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
