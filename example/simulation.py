import os
import logging
import time
from random import randint

import numpy as np
import multiprocessing as mp

from activitysim.core import inject
from activitysim.core import tracing

from activitysim.core.tracing import print_elapsed_time
from activitysim.core.config import handle_standard_args
from activitysim.core.config import setting


from activitysim import abm
from activitysim.core import pipeline

logger = logging.getLogger('activitysim')


def run():

    handle_standard_args()

    # specify None for a pseudo random base seed
    # inject.add_injectable('rng_base_seed', 0)

    tracing.config_logger()
    tracing.delete_csv_files()

    t0 = print_elapsed_time()

    MODELS = setting('models')

    # If you provide a resume_after argument to pipeline.run
    # the pipeline manager will attempt to load checkpointed tables from the checkpoint store
    # and resume pipeline processing on the next submodel step after the specified checkpoint
    resume_after = setting('resume_after', None)

    if resume_after:
        print "resume_after", resume_after

    pipeline.run(models=MODELS, resume_after=resume_after)

    # tables will no longer be available after pipeline is closed
    pipeline.close_pipeline()

    t0 = print_elapsed_time("all models", t0)


if __name__ == '__main__':
    run()
