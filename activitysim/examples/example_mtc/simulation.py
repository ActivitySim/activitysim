# ActivitySim
# See full license in LICENSE.txt.

# import sys
# if not sys.warnoptions:  # noqa: E402
#     import warnings
#     warnings.filterwarnings('error', category=Warning)
#     warnings.filterwarnings('ignore', category=PendingDeprecationWarning, module='future')
#     warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')

import logging
import argparse

# activitysim.abm imported for its side-effects (dependency injection)
from activitysim import abm

from activitysim.core.config import setting
from activitysim.core import pipeline
from activitysim.core import inject
from activitysim import cli

logger = logging.getLogger('activitysim')


def run():

    inject.add_injectable('data_dir', ['../example_data_sf'])
    inject.add_injectable('configs_dir', ['override_configs', 'configs'])

    cli.config.setup()

    # If you provide a resume_after argument to pipeline.run
    # the pipeline manager will attempt to load checkpointed tables from the checkpoint store
    # and resume pipeline processing on the next submodel step after the specified checkpoint
    resume_after = setting('resume_after', None)

    # cleanup if not resuming
    if not resume_after:
        cli.run.cleanup_output_files()

    pipeline.run(models=setting('models'), resume_after=resume_after)

    # tables will no longer be available after pipeline is closed
    pipeline.close_pipeline()


if __name__ == '__main__':
    run()
