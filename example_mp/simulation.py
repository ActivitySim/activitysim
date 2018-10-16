# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, unicode_literals)

from builtins import *

from future.standard_library import install_aliases
install_aliases()  # noqa: E402

import logging
from io import open
import sys

from activitysim.core import inject
from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import pipeline
# from activitysim import abm

import tasks

logger = logging.getLogger('activitysim')


def cleanup_output_files():

    active_log_files = \
        [h.baseFilename for h in logger.root.handlers if isinstance(h, logging.FileHandler)]
    tracing.delete_output_files('log', ignore=active_log_files)

    tracing.delete_output_files('h5')
    tracing.delete_output_files('csv')
    tracing.delete_output_files('txt')
    tracing.delete_output_files('yaml')


if __name__ == '__main__':

    # inject.add_injectable('data_dir', '/Users/jeff.doyle/work/activitysim-data/mtc_tm1/data')
    inject.add_injectable('data_dir', '../example/data')
    inject.add_injectable('configs_dir', ['configs', '../example/configs'])
    # inject.add_injectable('configs_dir', '../example/configs')

    config.handle_standard_args()
    tracing.config_logger()

    t0 = tracing.print_elapsed_time()

    injectables = ['data_dir', 'configs_dir', 'output_dir']
    injectables = {k: inject.get_injectable(k) for k in injectables}

    # cleanup if not resuming
    if not config.setting('resume_after', False):
        cleanup_output_files()

    run_list = tasks.get_run_list()

    mode = 'wb' if sys.version_info < (3,) else 'w'
    with open(config.output_file_path('run_list.txt'), mode) as f:
        tasks.print_run_list(run_list, f)

    tasks.print_run_list(run_list)
    bug

    if run_list['multiprocess']:
        logger.info("run multiprocess simulation")

        tasks.run_multiprocess(run_list, injectables)

    else:
        logger.info("run single process simulation")

        # tasks.run_simulation(run_list['models'], run_list['resume_after'])
        pipeline.run(models=run_list['models'], resume_after=run_list['resume_after'])

        # tables will no longer be available after pipeline is closed
        pipeline.close_pipeline()

    t0 = tracing.print_elapsed_time("everything", t0)
