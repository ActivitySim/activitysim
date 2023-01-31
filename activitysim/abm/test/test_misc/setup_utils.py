# ActivitySim
# See full license in LICENSE.txt.
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

from activitysim.core import config, inject, pipeline, random, tracing

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

    test_pipeline_configs_dir = os.path.join(os.path.dirname(__file__), "configs")
    example_configs_dir = example_path("configs")
    configs_dir = [test_pipeline_configs_dir, example_configs_dir]

    if ancillary_configs_dir is not None:
        configs_dir = [ancillary_configs_dir] + configs_dir

    inject.add_injectable("configs_dir", configs_dir)

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    inject.add_injectable("output_dir", output_dir)

    if not data_dir:
        data_dir = example_path("data")

    inject.add_injectable("data_dir", data_dir)

    inject.clear_cache()

    tracing.config_logger()

    tracing.delete_output_files("csv")
    tracing.delete_output_files("txt")
    tracing.delete_output_files("yaml")
    tracing.delete_output_files("omx")


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


def inject_settings(**kwargs):

    settings = config.read_settings_file("settings.yaml", mandatory=True)

    for k in kwargs:
        settings[k] = kwargs[k]

    inject.add_injectable("settings", settings)

    return settings
