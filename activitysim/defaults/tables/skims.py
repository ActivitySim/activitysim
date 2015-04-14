import urbansim.sim.simulation as sim
import openmatrix as omx
from activitysim import skim
import os

"""
Read in the omx files and create the skim objects
"""


@sim.injectable()
def nonmotskm_omx(data_dir):
    return omx.openFile(os.path.join(data_dir, 'data', "nonmotskm.omx"))


@sim.injectable()
def nonmotskm_matrix(nonmotskm_omx):
    return nonmotskm_omx['DIST']


@sim.injectable()
def distance_skim(nonmotskm_matrix):
    return skim.Skim(nonmotskm_matrix, offset=-1)


@sim.injectable()
def sovam_skim(nonmotskm_matrix):
    # FIXME use the right omx file
    return skim.Skim(nonmotskm_matrix, offset=-1)


@sim.injectable()
def sovmd_skim(nonmotskm_matrix):
    # FIXME use the right omx file
    return skim.Skim(nonmotskm_matrix, offset=-1)


@sim.injectable()
def sovpm_skim(nonmotskm_matrix):
    # FIXME use the right omx file
    return skim.Skim(nonmotskm_matrix, offset=-1)


@sim.injectable(cache=True)
def skims():
    skims = skim.Skims()
    # FIXME - this is reusing the same skim as all the different kinds of skims
    for typ in ["SOV_TIME", "SOVTOLL_TIME", "HOV2_TIME", "HOV2TOLL_TIME",
                "SOV_DIST", "SOVTOLL_DIST", "HOV2_DIST", "HOV2TOLL_DIST",
                "SOV_BTOLL", "SOVTOLL_BTOLL", "HOV2_BTOLL", "HOV2TOLL_BTOLL",
                "SOVTOLL_VTOLL", "HOV2TOLL_VTOLL",
                "HOV3_TIME",
                "HOV3_DIST",
                "HOV3_BTOLL"]:
        for period in ["AM", "MD", "PM"]:
            skims[(typ, period)] = sim.get_injectable("distance_skim")
    skims['DISTANCE'] = sim.get_injectable("distance_skim")
    return skims
