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
from activitysim.core import skim_maz
from activitysim.core import los
from activitysim.core import inject


from activitysim.abm.models.util.transit_virtual_path_builder import TransitVirtualPathBuilder


logger = logging.getLogger(__name__)

"""
Read in the omx files and create the skim objects
"""


@inject.injectable(cache=True)
def network_los():

    logger.debug("loading network_los injectable")
    nw_los = los.Network_LOS()
    nw_los.load_data()

    return nw_los



@inject.injectable(cache=True)
def path_builder(network_los):

    logger.debug("loading network_los injectable")
    tvpb = TransitVirtualPathBuilder(network_los)

    return tvpb


#
# @inject.injectable(cache=True)
# def skim_dict(network_los):
#
#     taz_skim_dict = network_los.get_skim_dict('taz')
#
#     if network_los.zone_system == los.ONE_ZONE:
#         logger.debug("loading skim_dict injectable (TAZ)")
#         return taz_skim_dict
#     else:
#         logger.debug("loading skim_dict injectable (MAZ)")
#         return skim_maz.MazSkimDict(network_los)


@inject.injectable(cache=True)
def skim_dict(network_los):
    return network_los.get_default_skim_dict()



@inject.injectable(cache=True)
def skim_stack(network_los, skim_dict):

    logger.debug("loading skim_stack injectable")

    if network_los.zone_system == los.ONE_ZONE:
        return skim.SkimStack(skim_dict)
    else:
        return skim.SkimStack(skim_dict.get_taz_skim_dict())
