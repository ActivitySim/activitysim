# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402

import os

import pandas as pd
import pandas.util.testing as pdt

from activitysim.core import tracing
from activitysim.core import pipeline
from activitysim.core import inject
from activitysim.core import mp_tasks

# set the max households for all tests (this is to limit memory use on travis)
HOUSEHOLDS_SAMPLE_SIZE = 100

# household with mandatory, non mandatory, atwork_subtours, and joint tours
HH_ID = 1482966

# [1062107 1098395 1115294 1482578 1482698 1482715 1482920
# 1482966 1809757 2022863 2123123 2123582 2591312]


# def teardown_function(func):
#     inject.clear_cache()
#     inject.reinject_decorated_tables()
#
#
# def close_handlers():
#
#     loggers = logging.Logger.manager.loggerDict
#     for name in loggers:
#         logger = logging.getLogger(name)
#         logger.handlers = []
#         logger.propagate = True
#         logger.setLevel(logging.NOTSET)


def regress_mini_auto():

    # regression test: these are among the first 10 households in households table
    hh_ids = [702445, 93713, 2525286, 945700]
    choices = [1, 1, 1, 0]
    expected_choice = pd.Series(choices, index=pd.Index(hh_ids, name="household_id"),
                                name='auto_ownership')

    auto_choice = pipeline.get_table("households").auto_ownership
    print("auto_choice\n", auto_choice.head(4))

    auto_choice = auto_choice.reindex(hh_ids)

    """
    auto_choice
     household_id
    702445     1
    93713      1
    2525286    1
    945700     0
    Name: auto_ownership, dtype: int64
    """
    pdt.assert_series_equal(auto_choice, expected_choice)


def test_mp_run():

    mp_configs_dir = os.path.join(os.path.dirname(__file__), 'configs_mp')
    configs_dir = os.path.join(os.path.dirname(__file__), 'configs')
    inject.add_injectable('configs_dir', [mp_configs_dir, configs_dir])

    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    inject.add_injectable("output_dir", output_dir)

    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    inject.add_injectable("data_dir", data_dir)

    tracing.config_logger()

    run_list = mp_tasks.get_run_list()
    mp_tasks.print_run_list(run_list)

    # do this after config.handle_standard_args, as command line args may override injectables
    injectables = ['data_dir', 'configs_dir', 'output_dir']
    injectables = {k: inject.get_injectable(k) for k in injectables}

    # pipeline.run(models=run_list['models'], resume_after=run_list['resume_after'])

    mp_tasks.run_multiprocess(run_list, injectables)
    pipeline.open_pipeline('_')
    regress_mini_auto()
    pipeline.close_pipeline()


if __name__ == '__main__':

    test_mp_run()
