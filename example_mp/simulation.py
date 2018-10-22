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
from activitysim.core import mp_tasks

# from activitysim import abm


logger = logging.getLogger('activitysim')


def cleanup_output_files():

    active_log_files = \
        [h.baseFilename for h in logger.root.handlers if isinstance(h, logging.FileHandler)]
    tracing.delete_output_files('log', ignore=active_log_files)

    tracing.delete_output_files('h5')
    tracing.delete_output_files('csv')
    tracing.delete_output_files('csv', subdir='trace')
    tracing.delete_output_files('txt')
    tracing.delete_output_files('yaml')


def run(run_list, injectables=None):

    if run_list['multiprocess']:
        logger.info("run multiprocess simulation")
        mp_tasks.run_multiprocess(run_list, injectables)
    else:
        logger.info("run single process simulation")
        pipeline.run(models=run_list['models'], resume_after=run_list['resume_after'])
        pipeline.close_pipeline()


if __name__ == '__main__':

    # inject.add_injectable('data_dir', '/Users/jeff.doyle/work/activitysim-data/mtc_tm1/data')
    inject.add_injectable('data_dir', '../example/data')
    inject.add_injectable('configs_dir', ['configs', '../example/configs'])

    config.handle_standard_args()
    tracing.config_logger()

    t0 = tracing.print_elapsed_time()

    # cleanup if not resuming
    if not config.setting('resume_after', False):
        cleanup_output_files()

    run_list = mp_tasks.get_run_list()

    if run_list['multiprocess']:
        # do this after config.handle_standard_args, as command line args may override injectables
        injectables = ['data_dir', 'configs_dir', 'output_dir']
        injectables = {k: inject.get_injectable(k) for k in injectables}
    else:
        injectables = None

    if config.setting('profile', False):
        import cProfile
        cProfile.runctx('run(run_list, injectables)',
                        globals(), locals(), filename=config.output_file_path('simulation.prof'))
    else:
        run(run_list, injectables)

    t0 = tracing.print_elapsed_time("everything", t0)
