# ActivitySim
# See full license in LICENSE.txt.

import os

import openmatrix as omx
import orca

from activitysim import skim

"""
Read in the omx files and create the skim objects
"""


# cache this so we don't open it again and again - skim code is not closing it....
@orca.injectable(cache=True)
def omx_file(data_dir):
    print "opening omx file"
    return omx.openFile(os.path.join(data_dir, 'data', "nonmotskm.omx"))


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

    skims = skim.Skims()
    skims['DISTANCE'] = skim.Skim(omx_file['DIST'], offset=-1)
    skims['DISTBIKE'] = skim.Skim(omx_file['DISTBIKE'], offset=-1)
    skims['DISTWALK'] = skim.Skim(omx_file['DISTWALK'], offset=-1)

    if preload_3d_skims:

        print "preload_3d_skims"

        # FIXME - assumes that the only types of key2 are time_periods
        # there may be more time periods in the skim than are used by the model
        skims_in_omx = omx_file.listMatrices()
        for skim_name in skims_in_omx:
            key, sep, key2 = skim_name.partition('__')
            if key2 and key2 in cache_skim_key_values:
                skim_key_tuple = (key, key2)
                # print skim_name, "(%s, %s)" % skim_key_tuple
                skims[skim_key_tuple] = skim.Skim(omx_file[skim_name], offset=-1)

    else:
        # need to load these for the injectables above
        skims[('SOV_TIME', 'AM')] = skim.Skim(omx_file['SOV_TIME__AM'], offset=-1)
        skims[('SOV_TIME', 'PM')] = skim.Skim(omx_file['SOV_TIME__PM'], offset=-1)
        skims[('SOV_TIME', 'MD')] = skim.Skim(omx_file['SOV_TIME__MD'], offset=-1)

    return skims
