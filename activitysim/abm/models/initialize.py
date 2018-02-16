# ActivitySim
# See full license in LICENSE.txt.

import logging
import os

import pandas as pd
import numpy as np

from activitysim.core import assign
from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import inject


logger = logging.getLogger(__name__)


@inject.step()
def initialize():
    """
    Because random seed is set differently for each step, the sampling of households depends
    on which step they are initially loaded in so we force them to load here and they get
    stored to the pipeline,
    """

    t0 = tracing.print_elapsed_time()
    inject.get_table('land_use').to_frame()
    t0 = tracing.print_elapsed_time("preload land_use", t0, debug=True)

    inject.get_table('households').to_frame()
    t0 = tracing.print_elapsed_time("preload households", t0, debug=True)

    inject.get_table('persons').to_frame()
    t0 = tracing.print_elapsed_time("preload persons", t0, debug=True)

    inject.get_table('person_windows').to_frame()
    t0 = tracing.print_elapsed_time("preload person_windows", t0, debug=True)


@inject.injectable(cache=True)
def preload_injectables():
    """
    preload bulky injectables up front - stuff that isn't inserted into eh pipeline
    """

    logger.info("preload_injectables")

    t0 = tracing.print_elapsed_time()

    if inject.get_injectable('skim_dict', None) is not None:
        t0 = tracing.print_elapsed_time("preload skim_dict", t0, debug=True)

    if inject.get_injectable('skim_stack', None) is not None:
        t0 = tracing.print_elapsed_time("preload skim_stack", t0, debug=True)
