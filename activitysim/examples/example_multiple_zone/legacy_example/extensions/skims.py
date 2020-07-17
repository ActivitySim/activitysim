# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import numpy as np
import openmatrix as omx

from activitysim.core import skim
from activitysim.core import config
from activitysim.core import inject


logger = logging.getLogger('activitysim')

"""
Read in the omx files and create the skim objects
"""


@inject.injectable(cache=True)
def taz_skim_dict(network_los):

    logger.info("loading taz_skim_dict")

    return network_los.skim_dict('taz')


@inject.injectable(cache=True)
def tap_skim_dict(network_los):

    logger.info("loading tap_skim_dict")

    return network_los.skim_dict('tap')
