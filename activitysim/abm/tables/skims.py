# ActivitySim
# See full license in LICENSE.txt.

import logging

from activitysim.core import config, inject, los
from activitysim.core.pathbuilder import TransitVirtualPathBuilder

from ...core.pipeline import Whale
from ...core.workflow import workflow_cached_object

logger = logging.getLogger(__name__)

"""
Read in the omx files and create the skim objects
"""


@workflow_cached_object
def network_los_preload(whale) -> los.Network_LOS:

    # when multiprocessing with shared data mp_tasks has to call network_los methods
    # allocate_shared_skim_buffers() and load_shared_data() BEFORE network_los.load_data()
    logger.debug("loading network_los_without_data_loaded injectable")
    nw_los = los.Network_LOS()

    return nw_los


@workflow_cached_object
def network_los(whale, network_los_preload: los.Network_LOS) -> los.Network_LOS:

    logger.debug("loading network_los injectable")
    network_los_preload.load_data()
    return network_los_preload


@workflow_cached_object
def skim_dict(whale, network_los):
    return network_los.get_default_skim_dict()


@workflow_cached_object
def log_settings(whale):

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
        "memory_profile",
        "instrument",
    ]
