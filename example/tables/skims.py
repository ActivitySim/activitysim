"""
Read in the omx files and create the skim objects
"""


@sim.injectable()
def nonmotskm_omx():
    return omx.openFile(os.path.join('data', "nonmotskm.omx"))


@sim.injectable()
def distance_skim(nonmotskm_omx):
    return skim.Skim(nonmotskm_omx['DIST'], offset=-1)


@sim.injectable()
def sovam_skim(nonmotskm_omx):
    # FIXME use the right omx file
    return skim.Skim(nonmotskm_omx['DIST'], offset=-1)


@sim.injectable()
def sovmd_skim(nonmotskm_omx):
    # FIXME use the right omx file
    return skim.Skim(nonmotskm_omx['DIST'], offset=-1)


@sim.injectable()
def sovpm_skim(nonmotskm_omx):
    # FIXME use the right omx file
    return skim.Skim(nonmotskm_omx['DIST'], offset=-1)
