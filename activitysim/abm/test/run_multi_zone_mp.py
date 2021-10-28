# ActivitySim
# See full license in LICENSE.txt.
import os

import pandas as pd
import pandas.testing as pdt

from activitysim.core import pipeline
from activitysim.core import inject
from activitysim.core import mp_tasks

from test_multi_zone import example_path
from test_multi_zone import setup_dirs
from test_multi_zone import regress_3_zone

# set the max households for all tests (this is to limit memory use on travis)
HOUSEHOLDS_SAMPLE_SIZE = 100


def test_mp_run():

    configs_dir = [example_path('configs_3_zone'), example_path('configs')]
    data_dir = example_path('data_3')

    setup_dirs(configs_dir, data_dir)
    inject.add_injectable('settings_file_name', 'settings_mp.yaml')

    run_list = mp_tasks.get_run_list()
    mp_tasks.print_run_list(run_list)

    # do this after config.handle_standard_args, as command line args may override injectables
    injectables = ['data_dir', 'configs_dir', 'output_dir', 'settings_file_name']
    injectables = {k: inject.get_injectable(k) for k in injectables}

    mp_tasks.run_multiprocess(run_list, injectables)
    pipeline.open_pipeline('_')
    regress_3_zone()
    pipeline.close_pipeline()


if __name__ == '__main__':

    test_mp_run()
