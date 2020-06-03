# ActivitySim
# See full license in LICENSE.txt.
from builtins import range
from builtins import int

import sys
import os
import logging
import multiprocessing

from collections import OrderedDict
from functools import reduce
from operator import mul

import numpy as np
import openmatrix as omx

from activitysim.core import skim
from activitysim.core import skim_loader
from activitysim.core import inject
from activitysim.core import util
from activitysim.core import config
from activitysim.core import tracing

logger = logging.getLogger(__name__)

"""
Read in the omx files and create the skim objects
"""


@inject.injectable(cache=True)
def skim_dict():
    return skim_loader.create_skim_dict('skims_file')


@inject.injectable(cache=True)
def skim_stack(skim_dict):

    logger.debug("loading skim_stack injectable")
    return skim.SkimStack(skim_dict)
