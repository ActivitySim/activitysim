# ActivitySim
# Copyright (C) 2014-2015 Synthicity, LLC
# Copyright (C) 2015 Autodesk
# See full license in LICENSE.txt.

import os

import openmatrix as omx
import orca

from activitysim import skim

"""
Read in the omx files and create the skim objects
"""


@orca.injectable()
def omx_file(data_dir):
    return omx.openFile(os.path.join(data_dir, 'data', "nonmotskm.omx"))


@orca.injectable()
def nonmotskm_matrix(omx_file):
    return omx_file['DIST']


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
    skims['DISTANCE'] = orca.get_injectable("distance_skim")
    return skims
