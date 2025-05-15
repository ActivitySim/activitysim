# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging
import os

import numpy as np
import numpy.testing as npt
import openmatrix as omx
import pandas as pd
import pandas.testing as pdt
import pkg_resources
import pytest
import yaml

from activitysim.core import config, random, tracing, workflow

# set the max households for all tests (this is to limit memory use on travis)
HOUSEHOLDS_SAMPLE_SIZE = 50
HOUSEHOLDS_SAMPLE_RATE = 0.01  # HOUSEHOLDS_SAMPLE_RATE / 5000 households

# household with mandatory, non mandatory, atwork_subtours, and joint tours
HH_ID = 257341

#  [ 257341 1234246 1402915 1511245 1931827 1931908 2307195 2366390 2408855
# 2518594 2549865  982981 1594365 1057690 1234121 2098971]

# SKIP_FULL_RUN = True
SKIP_FULL_RUN = False


def example_path(dirname):
    resource = os.path.join("examples", "prototype_mtc", dirname)
    return pkg_resources.resource_filename("activitysim", resource)


def setup_dirs(ancillary_configs_dir=None, data_dir=None):
    # ancillary_configs_dir is used by run_mp to test multiprocess

    # test_pipeline_configs_dir = os.path.join(os.path.dirname(__file__), "configs")
    example_configs_dir = example_path("configs")
    # configs_dir = [test_pipeline_configs_dir, example_configs_dir]
    configs_dir = [example_configs_dir]

    if ancillary_configs_dir is not None:
        configs_dir = [ancillary_configs_dir] + configs_dir

    output_dir = os.path.join(os.path.dirname(__file__), "output")

    if not data_dir:
        data_dir = example_path("data")

    state = workflow.State.make_default(
        configs_dir=configs_dir,
        output_dir=output_dir,
        data_dir=data_dir,
    )

    state.logging.config_logger()

    state.tracing.delete_output_files("csv")
    state.tracing.delete_output_files("txt")
    state.tracing.delete_output_files("yaml")
    state.tracing.delete_output_files("omx")

    return state


def close_handlers():
    loggers = logging.Logger.manager.loggerDict
    for name in loggers:
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True
        logger.setLevel(logging.NOTSET)
