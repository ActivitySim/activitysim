import os

import openmatrix as omx
import orca

from activitysim import skim

"""
Read in the omx files and create the skim objects
"""


@orca.injectable()
def nonmotskm_omx(data_dir):
    return omx.openFile(os.path.join(data_dir, 'data', "nonmotskm.omx"))


@orca.injectable()
def nonmotskm_matrix(nonmotskm_omx):
    return nonmotskm_omx['DIST']


@orca.injectable()
def distance_skim(nonmotskm_matrix):
    return skim.Skim(nonmotskm_matrix, offset=-1)


@orca.injectable()
def sovam_skim(nonmotskm_matrix):
    # FIXME use the right omx file
    return skim.Skim(nonmotskm_matrix, offset=-1)


@orca.injectable()
def sovmd_skim(nonmotskm_matrix):
    # FIXME use the right omx file
    return skim.Skim(nonmotskm_matrix, offset=-1)


@orca.injectable()
def sovpm_skim(nonmotskm_matrix):
    # FIXME use the right omx file
    return skim.Skim(nonmotskm_matrix, offset=-1)


@orca.injectable(cache=True)
def skims():
    skims = skim.Skims()
    # FIXME - this is reusing the same skim as all the different kinds of skims
    for typ in ["SOV_TIME", "SOVTOLL_TIME", "HOV2_TIME",
                "SOV_DIST", "SOVTOLL_DIST", "HOV2_DIST",
                "SOV_BTOLL", "SOVTOLL_BTOLL", "HOV2_BTOLL",
                "SOVTOLL_VTOLL"]:
        for period in ["AM", "MD", "PM"]:
            skims[(typ, period)] = orca.get_injectable("distance_skim")
    skims['DISTANCE'] = orca.get_injectable("distance_skim")
    return skims
