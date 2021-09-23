# ActivitySim
# See full license in LICENSE.txt.

import logging

from activitysim.core import config, inject, los
from activitysim.core.pathbuilder import TransitVirtualPathBuilder

logger = logging.getLogger(__name__)

"""
Read in the omx files and create the skim objects
"""


@inject.injectable(cache=True)
def network_los_preload():

    # when multiprocessing with shared data mp_tasks has to call network_los methods
    # allocate_shared_skim_buffers() and load_shared_data() BEFORE network_los.load_data()
    logger.debug("loading network_los_without_data_loaded injectable")
    nw_los = los.Network_LOS()

    return nw_los


@inject.injectable(cache=True)
def network_los(network_los_preload):

    logger.debug("loading network_los injectable")
    network_los_preload.load_data()
    return network_los_preload


@inject.injectable(cache=True)
def skim_dict(network_los):
    return network_los.get_default_skim_dict()


@inject.injectable()
def log_settings():

    # abm settings to log on startup
    return [
        "households_sample_size",
        "chunk_size",
        "chunk_method",
        "chunk_training_mode",
        "multiprocess",
        "num_processes",
        "resume_after",
        "trace_hh_id",
    ]
