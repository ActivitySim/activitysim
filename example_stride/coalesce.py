# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402

import sys
import logging
import argparse
import os

from activitysim.core import mem
from activitysim.core import inject
from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import pipeline
from activitysim.core import mp_tasks
from activitysim.core import chunk

# from activitysim import abm


logger = logging.getLogger('activitysim')


if __name__ == '__main__':

    inject.add_injectable('configs_dir', ['configs', '../example/configs'])

    config.handle_standard_args()

    mp_tasks.filter_warnings()
    tracing.config_logger()

    t0 = tracing.print_elapsed_time()

    coalesce_rules = config.setting('coalesce')

    mp_tasks.coalesce_pipelines(coalesce_rules['names'], coalesce_rules['slice'], use_prefix=False)

    checkpoints_df = pipeline.get_checkpoints()
    file_path = config.output_file_path('coalesce_checkpoints.csv')
    checkpoints_df.to_csv(file_path, index=True)

    t0 = tracing.print_elapsed_time("everything", t0)
