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


@orca.injectable(cache=True)
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
    def_skim = orca.get_injectable("distance_skim")
    # FIXME - this is reusing the same skim as all the different kinds of skims
    for typ in ["SOV_TIME", "SOVTOLL_TIME", "HOV2_TIME", "HOV2TOLL_TIME",
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

                "DRV_COM_WLK_BOARDS", "WLK_COM_DRV_BOARDS",
                "DRV_COM_WLK_DDIST", "WLK_COM_DRV_DDIST",
                "DRV_COM_WLK_FAR", "WLK_COM_DRV_FAR",
                "DRV_COM_WLK_DTIM", "WLK_COM_DRV_DTIM",
                "DRV_COM_WLK_DDIST", "WLK_COM_DRV_DDIST",
                "DRV_COM_WLK_IWAIT", "WLK_COM_DRV_IWAIT",
                "DRV_COM_WLK_XWAIT", "DRV_COM_WLK_BOARDS",
                "DRV_COM_WLK_KEYIVT", "WLK_COM_DRV_KEYIVT",
                "DRV_COM_WLK_TOTIVT", "WLK_COM_DRV_TOTIVT",
                "DRV_COM_WLK_WAUX", "WLK_COM_DRV_WAUX",
                "DRV_COM_WLK_XWAIT", "WLK_COM_DRV_XWAIT",
                "DRV_COM_WLK_WAUX",

                "DRV_EXP_WLK_TOTIVT", "WLK_EXP_DRV_TOTIVT",
                "DRV_EXP_WLK_KEYIVT", "WLK_EXP_DRV_KEYIVT",
                "DRV_EXP_WLK_IWAIT", "WLK_EXP_DRV_IWAIT",
                "DRV_EXP_WLK_XWAIT", "WLK_EXP_DRV_XWAIT",
                "DRV_EXP_WLK_WAUX", "WLK_EXP_DRV_WAUX",
                "DRV_EXP_WLK_DTIM", "WLK_EXP_DRV_DTIM",
                "DRV_EXP_WLK_DDIST", "WLK_EXP_DRV_DDIST",
                "DRV_EXP_WLK_BOARDS", "WLK_EXP_DRV_BOARDS",
                "DRV_EXP_WLK_XWAIT", "WLK_EXP_DRV_XWAIT",
                "DRV_EXP_WLK_FAR", "WLK_EXP_DRV_FAR",

                "DRV_HVY_WLK_TOTIVT", "WLK_HVY_DRV_TOTIVT",
                "DRV_HVY_WLK_KEYIVT", "WLK_HVY_DRV_KEYIVT",
                "DRV_HVY_WLK_IWAIT", "WLK_HVY_DRV_IWAIT",
                "DRV_HVY_WLK_XWAIT", "WLK_HVY_DRV_XWAIT",
                "DRV_HVY_WLK_WAUX", "WLK_HVY_DRV_WAUX",
                "DRV_HVY_WLK_DTIM", "WLK_HVY_DRV_DTIM",
                "DRV_HVY_WLK_DDIST", "WLK_HVY_DRV_DDIST",
                "DRV_HVY_WLK_BOARDS", "WLK_HVY_DRV_BOARDS",
                "DRV_COM_WLK_XWAIT", "WLK_COM_DRV_XWAIT",
                "DRV_HVY_WLK_FAR", "WLK_HVY_DRV_FAR",

                "DRV_LRF_WLK_TOTIVT", "WLK_LRF_DRV_TOTIVT",
                "DRV_LRF_WLK_KEYIVT", "WLK_LRF_DRV_KEYIVT",
                "DRV_LRF_WLK_IWAIT", "WLK_LRF_DRV_IWAIT",
                "DRV_LRF_WLK_XWAIT", "WLK_LRF_DRV_XWAIT",
                "DRV_LRF_WLK_WAUX", "WLK_LRF_DRV_WAUX",
                "DRV_LRF_WLK_DTIM", "WLK_LRF_DRV_DTIM",
                "DRV_LRF_WLK_DDIST", "WLK_LRF_DRV_DDIST",
                "DRV_LRF_WLK_BOARDS", "WLK_LRF_DRV_BOARDS",
                "DRV_LRF_WLK_XWAIT", "WLK_LRF_DRV_XWAIT",
                "DRV_LRF_WLK_FAR", "WLK_LRF_DRV_FAR",
                "DRV_LRF_WLK_FERRYIVT", "WLK_LRF_DRV_FERRYIVT",

                             "HOV3TOLL_VTOLL",
                "HOV3_TIME", "HOV3TOLL_TIME",
                "HOV3_DIST", "HOV3TOLL_DIST",
                "HOV3_BTOLL", "HOV3TOLL_BTOLL"]:
        for period in ["AM", "MD", "PM"]:
            skims[(typ, period)] = def_skim
    skims['DISTANCE'] = orca.get_injectable("distance_skim")
    return skims
