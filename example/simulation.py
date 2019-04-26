# ActivitySim
# See full license in LICENSE.txt.

from __future__ import print_function

# import sys
# if not sys.warnoptions:  # noqa: E402
#     import warnings
#     warnings.filterwarnings('error', category=Warning)
#     warnings.filterwarnings('ignore', category=PendingDeprecationWarning, module='future')
#     warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')

import logging

# activitysim.abm imported for its side-effects (dependency injection)
from activitysim import abm

from activitysim.core import tracing
from activitysim.core import config
from activitysim.core.config import setting
from activitysim.core import pipeline

logger = logging.getLogger('activitysim')


def run():
    config.handle_standard_args()

    # specify None for a pseudo random base seed
    # inject.add_injectable('rng_base_seed', 0)

    tracing.config_logger()
    config.filter_warnings()

    tracing.delete_csv_files()

    # If you provide a resume_after argument to pipeline.run
    # the pipeline manager will attempt to load checkpointed tables from the checkpoint store
    # and resume pipeline processing on the next submodel step after the specified checkpoint
    resume_after = setting('resume_after', None)

    if resume_after:
        print("resume_after", resume_after)

    pipeline.run(models=setting('models'), resume_after=resume_after)

    # tables will no longer be available after pipeline is closed
    pipeline.close_pipeline()


if __name__ == '__main__':
    run()
