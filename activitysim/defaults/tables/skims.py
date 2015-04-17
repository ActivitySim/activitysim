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


@sim.injectable(cache=True)
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
    for typ in ["SOV_TIME"]:
        """, "SOVTOLL_TIME", "HOV2_TIME", "HOV2TOLL_TIME",
                "SOV_DIST", "SOVTOLL_DIST", "HOV2_DIST", "HOV2TOLL_DIST",
                "SOV_BTOLL", "SOVTOLL_BTOLL", "HOV2_BTOLL", "HOV2TOLL_BTOLL",
                "SOVTOLL_VTOLL", "HOV2TOLL_VTOLL",

                "WLK_LOC_WLK_TOTIVT", "WLK_LOC_WLK_IWAIT", "WLK_LOC_WLK_XWAIT",
                "WLK_LOC_WLK_BOARDS", "WLK_LOC_WLK_WAUX", "WLK_LOC_WLK_FAR",

                "WLK_LRF_WLK_FERRYIVT", "WLK_LRF_WLK_KEYIVT",
                "WLK_LRF_WLK_TOTIVT", "WLK_LRF_WLK_IWAIT", "WLK_LRF_WLK_XWAIT",
                "WLK_LRF_WLK_BOARDS", "WLK_LRF_WLK_WAUX", "WLK_LRF_WLK_FAR",

                "WLK_EXP_WLK_KEYIVT",
                "WLK_EXP_WLK_TOTIVT", "WLK_EXP_WLK_IWAIT", "WLK_EXP_WLK_XWAIT",
                "WLK_EXP_WLK_BOARDS", "WLK_EXP_WLK_WAUX", "WLK_EXP_WLK_FAR",

                "WLK_HVY_WLK_KEYIVT",
                "WLK_HVY_WLK_TOTIVT", "WLK_HVY_WLK_IWAIT", "WLK_HVY_WLK_XWAIT",
                "WLK_HVY_WLK_BOARDS", "WLK_HVY_WLK_WAUX", "WLK_HVY_WLK_FAR",

                "WLK_COM_WLK_KEYIVT",
                "WLK_COM_WLK_TOTIVT", "WLK_COM_WLK_IWAIT", "WLK_COM_WLK_XWAIT",
                "WLK_COM_WLK_BOARDS", "WLK_COM_WLK_WAUX", "WLK_COM_WLK_FAR",

                "DRV_LOC_WLK_TOTIVT", "WLK_LOC_DRV_TOTIVT",
                "DRV_LOC_WLK_IWAIT",  "WLK_LOC_DRV_IWAIT",
                "DRV_LOC_WLK_XWAIT",  "WLK_LOC_DRV_XWAIT",
                "DRV_LOC_WLK_BOARDS", "WLK_LOC_DRV_BOARDS",
                "DRV_LOC_WLK_DTIM",   "WLK_LOC_DRV_DTIM",
                "DRV_LOC_WLK_WAUX",   "WLK_LOC_DRV_WAUX",
                "DRV_LOC_WLK_DDIST",  "WLK_LOC_DRV_DDIST",
                "DRV_LOC_WLK_FAR",    "WLK_LOC_DRV_FAR",

                             "HOV3TOLL_VTOLL",
                "HOV3_TIME", "HOV3TOLL_TIME",
                "HOV3_DIST", "HOV3TOLL_DIST",
                "HOV3_BTOLL", "HOV3TOLL_BTOLL"]:"""
        for period in ["AM", "MD", "PM"]:
            skims[(typ, period)] = sim.get_injectable("distance_skim")
    skims['DISTANCE'] = sim.get_injectable("distance_skim")
    return skims
