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
    return omx.openFile(os.path.join(data_dir, 'data', "nonmotskm.omx"))


@orca.injectable(cache=True)
def distance_skim(skims):
    return skims['DISTANCE']


@orca.injectable()
def sovam_skim(skims):
    # FIXME use the right skim
    return skims['DISTANCE']


@orca.injectable()
def sovmd_skim(skims):
    # FIXME use the right skim
    return skims['DISTANCE']


@orca.injectable()
def sovpm_skim(skims):
    # FIXME use the right skim
    return skims['DISTANCE']


@orca.injectable(cache=True)
def skims(omx_file):
    skims = skim.Skims()
    skims['DISTANCE'] = skim.Skim(omx_file['DIST'], offset=-1)
    skims['DISTBIKE'] = skim.Skim(omx_file['DISTBIKE'], offset=-1)
    skims['DISTWALK'] = skim.Skim(omx_file['DISTWALK'], offset=-1)

    return skims
