# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import openmatrix as omx
import orca

from activitysim.core import skim as askim
from activitysim.core import tracing
from activitysim.core import pipeline

logger = logging.getLogger(__name__)

"""
Read in the omx files and create the skim objects
"""


# cache this so we don't open it again and again - skim code is not closing it....
@orca.injectable(cache=True)
def omx_file(data_dir, settings):
    logger.debug("opening omx file")

    fname = os.path.join(data_dir, settings["skims_file"])
    file = omx.open_file(fname)

    pipeline.close_on_exit(file, fname)

    return file


@orca.injectable(cache=True)
def skim_dict(omx_file, cache_skim_key_values):

    logger.info("skims injectable loading skims")

    skim_dict = askim.SkimDict()
    skim_dict.offset_mapper.set_offset_int(-1)

    skims_in_omx = omx_file.listMatrices()
    for skim_name in skims_in_omx:
        key, sep, key2 = skim_name.partition('__')
        skim_data = omx_file[skim_name]
        if not sep:
            # no separator - this is a simple 2d skim - we load them all
            skim_dict.set(key, skim_data)
        else:
            # there may be more time periods in the skim than are used by the model
            # cache_skim_key_values is a list of time periods (frem settings) that are used
            # FIXME - assumes that the only types of key2 are time_periods
            if key2 in cache_skim_key_values:
                skim_dict.set((key, key2), skim_data)

    return skim_dict


@orca.injectable(cache=True)
def skim_stack(skim_dict):

    logger.debug("loading skim_stack")
    return askim.SkimStack(skim_dict)
