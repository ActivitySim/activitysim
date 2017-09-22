# ActivitySim
# See full license in LICENSE.txt.

import os
import tempfile
import logging

import numpy as np
import orca
import pandas as pd
import pandas.util.testing as pdt
import pytest
import yaml

from . import extensions

from activitysim.core import tracing
from activitysim.core import pipeline
from activitysim.core import inject

# set the max households for all tests (this is to limit memory use on travis)
HOUSEHOLDS_SAMPLE_SIZE = 100
HH_ID = 961042


def teardown_function(func):
    orca.clear_cache()
    inject.reinject_decorated_tables()


def close_handlers():

    loggers = logging.Logger.manager.loggerDict
    for name in loggers:
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True
        logger.setLevel(logging.NOTSET)


def test_pipeline_run():

    orca.orca._INJECTABLES.pop('skim_dict', None)
    orca.orca._INJECTABLES.pop('skim_stack', None)

    configs_dir = os.path.join(os.path.dirname(__file__), 'configs')
    orca.add_injectable("configs_dir", configs_dir)

    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    orca.add_injectable("output_dir", output_dir)

    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    orca.add_injectable("data_dir", data_dir)

    orca.clear_cache()

    tracing.config_logger()

    _MODELS = [
        'step1',
    ]

    pipeline.run(models=_MODELS, resume_after=None)

    table1 = pipeline.get_table("table1").column1

    # test that model arg is passed to step
    pipeline.run_model('step2.table_name=table2')

    table2 = pipeline.get_table("table2").column1

    # try to get a non-existant table
    with pytest.raises(RuntimeError) as excinfo:
        pipeline.get_table("bogus")
    assert "not in checkpointed tables" in str(excinfo.value)

    # try to get an existing table from a non-existant checkpoint
    with pytest.raises(RuntimeError) as excinfo:
        pipeline.get_table("table1", checkpoint_name="bogus")
    assert "not in checkpoints" in str(excinfo.value)

    pipeline.close_pipeline()
    orca.clear_cache()

    close_handlers()
