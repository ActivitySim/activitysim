# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import openmatrix as omx
import orca

from activitysim import skim as askim
from activitysim import tracing

logger = logging.getLogger(__name__)

"""
Read in the omx files and create the skim objects
"""


def add_to_skim_dict(skim_dict, omx_file, cache_skim_key_values, offset):

    skims_in_omx = omx_file.listMatrices()
    for skim_name in skims_in_omx:
        key, sep, key2 = skim_name.partition('__')
        if not sep:
            # no separator - this is a simple 2d skim - we load them all
            skim_dict.set(key, askim.Skim(omx_file[skim_name], offset=offset))
        else:
            # there may be more time periods in the skim than are used by the model
            # cache_skim_key_values is a list of time periods (from settings) that are used
            # FIXME - assumes that the only types of key2 are time_periods
            if key2 in cache_skim_key_values:
                skim_dict.set((key, key2), askim.Skim(omx_file[skim_name], offset=offset))



@orca.injectable(cache=True)
def taz_skim_dict(data_dir, settings):

    logger.info("loading taz_skim_dict")

    skims_file = os.path.join(data_dir, settings["taz_skims_file"])
    cache_skim_key_values = settings['time_periods']['labels']

    skim_dict = askim.SkimDict()

    with omx.open_file(skims_file) as omx_file:
        add_to_skim_dict(skim_dict, omx_file, cache_skim_key_values, offset=None)

    return skim_dict


@orca.injectable(cache=True)
def tap_skim_dict(data_dir, settings):

    logger.info("loading tap_skim_dict")

    cache_skim_key_values = settings['time_periods']['labels']
    skim_dict = askim.SkimDict()

    for skims_file in settings["tap_skims_files"]:
        skims_file_path = os.path.join(data_dir, skims_file)
        with omx.open_file(skims_file_path) as omx_file:
            add_to_skim_dict(skim_dict, omx_file, cache_skim_key_values, offset=None)

    return skim_dict

