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

from .setup_utils import inject_settings, setup_dirs

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


def test_load_cached_accessibility():

    inject.clear_cache()
    inject.reinject_decorated_tables()

    data_dir = [os.path.join(os.path.dirname(__file__), "data"), example_path("data")]
    setup_dirs(data_dir=data_dir)

    #
    # add OPTIONAL ceched table accessibility to input_table_list
    # activitysim.abm.tables.land_use.accessibility() will load this table if listed here
    # presumably independently calculated outside activitysim or a cached copy created during a previous run
    #
    settings = config.read_settings_file("settings.yaml", mandatory=True)
    input_table_list = settings.get("input_table_list")
    input_table_list.append(
        {
            "tablename": "accessibility",
            "filename": "cached_accessibility.csv",
            "index_col": "zone_id",
        }
    )
    inject_settings(
        households_sample_size=HOUSEHOLDS_SAMPLE_SIZE, input_table_list=input_table_list
    )

    _MODELS = [
        "initialize_landuse",
        # 'compute_accessibility',  # we load accessibility table ordinarily created by compute_accessibility
        "initialize_households",
    ]

    pipeline.run(models=_MODELS, resume_after=None)

    accessibility_df = pipeline.get_table("accessibility")

    assert "auPkRetail" in accessibility_df

    pipeline.close_pipeline()
    inject.clear_cache()
    close_handlers()


if __name__ == "__main__":
    from activitysim import abm  # register injectables

    print("running test_ftest_load_cached_accessibilityull_run1")
    test_load_cached_accessibility()
    # teardown_function(None)
