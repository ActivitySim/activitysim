# ActivitySim
# See full license in LICENSE.txt.

import numpy as np
import pandas as pd
import openmatrix as omx

# input files, SF county is zones 1 to 190, output files
data_file = "mtc_asim.h5"
skims_file = "nonmotskm.omx"


def create_subset(data_file_out, skims_file_out, maxZone, households_sample_size=0):

    # process all data tables
    print 'skims/accessibility'
    df = pd.read_hdf(data_file, 'skims/accessibility')
    df = df[df.index <= maxZone]
    df.to_hdf(data_file_out, 'skims/accessibility')

    print 'land_use/taz_data'
    df = pd.read_hdf(data_file, 'land_use/taz_data')
    df = df[df.index <= maxZone]
    df.to_hdf(data_file_out, 'land_use/taz_data')

    print 'households'
    df = pd.read_hdf(data_file, 'households')
    df = df[df.TAZ <= maxZone]
    if households_sample_size:
        df = df.take(np.random.choice(len(df), size=households_sample_size, replace=False))
    df.to_hdf(data_file_out, 'households')

    print 'persons'
    per = pd.read_hdf(data_file, 'persons')
    per = per[per.household_id.isin(df.index)]
    per.to_hdf(data_file_out, 'persons')

    # process all skims
    skims = omx.openFile(skims_file)
    skims_out = omx.openFile(skims_file_out, 'a')

    skimsToProcess = skims.listMatrices()
    for skimName in skimsToProcess:
        print skimName
        skims_out[skimName] = skims[skimName][0:maxZone, 0:maxZone]
        skims_out[skimName].attrs.TITLE = ''  # remove funny character for OMX viewer


create_subset(data_file_out="mtc_asim_sf.h5",
              skims_file_out="nonmotskm_sf.omx",
              maxZone=190
              )

create_subset(data_file_out="mtc_asim_sf_test.h5",
              skims_file_out="nonmotskm_sf_test.omx",
              maxZone=25,
              households_sample_size=5000
              )
