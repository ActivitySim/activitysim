# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import openmatrix as omx
import orca

from activitysim import skim
from activitysim import tracing

logger = logging.getLogger(__name__)

"""
Read in the omx files and create the skim objects
"""


# cache this so we don't open it again and again - skim code is not closing it....
@orca.injectable(cache=True)
def omx_file(data_dir, settings):
    logger.debug("opening omx file")
    return omx.openFile(os.path.join(data_dir, 'data', settings["skims_file"]))


@orca.injectable()
def distance_skim(skims):
    # want the Skim
    return skims.get_skim('DISTANCE')


@orca.injectable()
def sovam_skim(skims):
    # want the Skim
    return skims.get_skim(('SOV_TIME', 'AM'))


@orca.injectable()
def sovmd_skim(skims):
    # want the Skim
    return skims.get_skim(('SOV_TIME', 'MD'))


@orca.injectable()
def sovpm_skim(skims):
    # want the Skim
    return skims.get_skim(('SOV_TIME', 'PM'))


@orca.injectable(cache=True)
def skims(omx_file, preload_3d_skims, cache_skim_key_values):

    logger.info("loading skims (preload_3d_skims: %s)" % preload_3d_skims)

    skims = skim.Skims()
    skims['DISTANCE'] = skim.Skim(omx_file['DIST'], offset=-1)
    skims['DISTBIKE'] = skim.Skim(omx_file['DISTBIKE'], offset=-1)
    skims['DISTWALK'] = skim.Skim(omx_file['DISTWALK'], offset=-1)

    if preload_3d_skims:
        logger.info("skims injectable preloading preload_3d_skims")
        # FIXME - assumes that the only types of key2 are time_periods
        # there may be more time periods in the skim than are used by the model
        skims_in_omx = omx_file.listMatrices()
        for skim_name in skims_in_omx:
            logger.debug("skims injectable preloading skim %s" % skim_name)
            key, sep, key2 = skim_name.partition('__')
            if key2 and key2 in cache_skim_key_values:
                skims.set_3d(key, key2, skim.Skim(omx_file[skim_name], offset=-1))
    else:
        # need to load these for the injectables above
        skims.set_3d('SOV_TIME', 'AM', skim.Skim(omx_file['SOV_TIME__AM'], offset=-1))
        skims.set_3d('SOV_TIME', 'PM', skim.Skim(omx_file['SOV_TIME__PM'], offset=-1))
        skims.set_3d('SOV_TIME', 'MD', skim.Skim(omx_file['SOV_TIME__MD'], offset=-1))

    return skims


@orca.injectable(cache=True)
def stacked_skims(skims):

    logger.info("loading stacked_skims")
    return skim.SkimStack(skims)
