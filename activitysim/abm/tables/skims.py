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
def skim_dicts():

    skims_manifest = skim_loader.get_skims_manifest()

    for skim_tag, skim_settings in skims_manifest.items():

        skim_settings['skims'] = skim_loader.create_skim_dict(skim_tag)

    return skims_manifest


@inject.injectable(cache=True)
def skim_dict(skim_dicts):

    logger.debug("loading skim_dict injectable")

    assert 'TAZ' in skim_dicts

    return skim_dicts.get('TAZ').get('skims')


@inject.injectable(cache=True)
def skim_stack(skim_dict):

    logger.debug("loading skim_stack injectable")
    return skim.SkimStack(skim_dict)
