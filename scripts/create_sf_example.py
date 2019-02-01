# ActivitySim
# See full license in LICENSE.txt.

import os

import numpy as np
import pandas as pd
import openmatrix as omx

# input files, SF county is zones 1 to 190, output files
source_store = "/Users/jeff.doyle/work/activitysim-data/mtc_tm1/data/mtc_asim.h5"
source_skims = '/Users/jeff.doyle/work/activitysim-data/mtc_tm1/data/skims.omx'

dest_data_dir = "/Users/jeff.doyle/work/activitysim-data"

def create_subset(dest_store, dest_skims, maxZone, households_sample_size=0):

    dest_store_path = os.path.join(dest_data_dir, dest_store)
    dest_skims_path = os.path.join(dest_data_dir, dest_skims)

    print 'land_use/taz_data'
    df = pd.read_hdf(source_store, 'land_use/taz_data')
    df = df[df.index <= maxZone]
    df.to_hdf(dest_store_path, 'land_use/taz_data')
    del df

    print 'households'
    hh_df = pd.read_hdf(source_store, 'households')
    hh_df = hh_df[hh_df.TAZ <= maxZone]
    if households_sample_size:
        hh_df = hh_df.take(np.random.choice(len(hh_df), size=households_sample_size, replace=False))
    hh_df.to_hdf(dest_store_path, 'households')

    print 'persons'
    per_df = pd.read_hdf(source_store, 'persons')
    per_df = per_df[per_df.household_id.isin(hh_df.index)]
    per_df.to_hdf(dest_store_path, 'persons')

    # process all skims
    skims = omx.open_file(source_skims)
    skims_out = omx.open_file(dest_skims_path, 'w')

    skimsToProcess = skims.list_matrices()
    for skimName in skimsToProcess:
        print skimName
        skims_out[skimName] = skims[skimName][0:maxZone, 0:maxZone]
        skims_out[skimName].attrs.TITLE = ''  # remove funny character for OMX viewer


create_subset(dest_store='mtc_tm1_sf/data/mtc_asim.h5',
              dest_skims='mtc_tm1_sf/data/skims.omx',
              maxZone=190
              )

create_subset(dest_store='mtc_tm1_test/data/mtc_asim.h5',
              dest_skims='mtc_tm1_test/data/skims.omx',
              maxZone=25,
              households_sample_size=5000
              )
