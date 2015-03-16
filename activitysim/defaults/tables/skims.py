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
    skims[("SOV", "AM")] = sim.get_injectable("sovam_skim")
    skims[("SOV", "MD")] = sim.get_injectable("sovmd_skim")
    skims[("SOV", "PM")] = sim.get_injectable("sovpm_skim")
    skims["DISTANCE"] = sim.get_injectable("distance_skim")
    return skims
