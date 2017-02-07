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


# cache this so we don't open it again and again - skim code is not closing it....
@orca.injectable(cache=True)
def omx_file(data_dir, settings):
    logger.debug("opening omx file")

    return omx.open_file(os.path.join(data_dir, settings["skims_file"]))


@orca.injectable()
def distance_skim(skim_dict):
    # want the Skim
    return skim_dict.get('DISTANCE')


@orca.injectable()
def sovam_skim(skim_dict):
    # want the Skim
    return skim_dict.get(('SOV_TIME', 'AM'))


@orca.injectable()
def sovmd_skim(skim_dict):
    # want the Skim
    return skim_dict.get(('SOV_TIME', 'MD'))


@orca.injectable()
def sovpm_skim(skim_dict):
    # want the Skim
    return skim_dict.get(('SOV_TIME', 'PM'))


@orca.injectable(cache=True)
def skim_dict(omx_file, preload_3d_skims, cache_skim_key_values):

    logger.info("loading skims (preload_3d_skims: %s)" % preload_3d_skims)

    skim_dict = askim.SkimDict()
    skim_dict.set('DISTANCE', askim.Skim(omx_file['DIST'], offset=-1))
    skim_dict.set('DISTBIKE', askim.Skim(omx_file['DISTBIKE'], offset=-1))
    skim_dict.set('DISTWALK', askim.Skim(omx_file['DISTWALK'], offset=-1))

    if preload_3d_skims:
        logger.info("skims injectable preloading preload_3d_skims")
        # FIXME - assumes that the only types of key2 are time_periods
        # there may be more time periods in the skim than are used by the model
        skims_in_omx = omx_file.listMatrices()
        for skim_name in skims_in_omx:
            # logger.debug("skims injectable preloading skim %s" % skim_name)
            key, sep, key2 = skim_name.partition('__')
            if key2 and key2 in cache_skim_key_values:
                skim_dict.set((key, key2), askim.Skim(omx_file[skim_name], offset=-1))
    else:
        # need to load these for the injectables above
        skim_dict.set(('SOV_TIME', 'AM'), askim.Skim(omx_file['SOV_TIME__AM'], offset=-1))
        skim_dict.set(('SOV_TIME', 'PM'), askim.Skim(omx_file['SOV_TIME__PM'], offset=-1))
        skim_dict.set(('SOV_TIME', 'MD'), askim.Skim(omx_file['SOV_TIME__MD'], offset=-1))

    return skim_dict


@orca.injectable(cache=True)
def skim_stack(skim_dict):

    logger.info("loading skim_stack")
    return askim.SkimStack(skim_dict)
