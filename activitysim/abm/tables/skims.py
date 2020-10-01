# ActivitySim
# See full license in LICENSE.txt.

import logging

from activitysim.core import skim
from activitysim.core import los
from activitysim.core import inject


from activitysim.core.transit_virtual_path_builder import TransitVirtualPathBuilder


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
