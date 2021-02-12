# ActivitySim
# See full license in LICENSE.txt.
import os

import pandas as pd
import pandas.testing as pdt

from activitysim import abm  # register injectables

from activitysim.core import pipeline
from activitysim.core import inject
from activitysim.core import mp_tasks

from test_pipeline import setup_dirs

# set the max households for all tests (this is to limit memory use on travis)
HOUSEHOLDS_SAMPLE_SIZE = 100

# household with mandatory, non mandatory, atwork_subtours, and joint tours
HH_ID = 1396417


def regress_mini_auto():

    # regression test: these are among the middle households in households table
    # should be the same results as in test_pipeline (single-threaded) tests
    hh_ids = [932147, 982875, 983048, 1024353]
    choices = [1, 1, 1, 0]
    expected_choice = pd.Series(choices, index=pd.Index(hh_ids, name="household_id"),
                                name='auto_ownership')

    auto_choice = pipeline.get_table("households").sort_index().auto_ownership

    offset = HOUSEHOLDS_SAMPLE_SIZE // 2  # choose something midway as hh_id ordered by hh size
    print("auto_choice\n", auto_choice.head(offset).tail(4))

    auto_choice = auto_choice.reindex(hh_ids)

    """
    auto_choice
     household_id
    932147     1
    982875     1
    983048     1
    1024353    0
    Name: auto_ownership, dtype: int64
    """
    pdt.assert_series_equal(auto_choice, expected_choice)


def test_mp_run():

    mp_configs_dir = os.path.join(os.path.dirname(__file__), 'configs_mp')

    setup_dirs(ancillary_configs_dir=mp_configs_dir)

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
