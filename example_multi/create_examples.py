# ActivitySim
# See full license in LICENSE.txt.

import numpy as np
import pandas as pd
import openmatrix as omx

input_folder = "/Users/jeff.doyle/work/activitysim-data/sandag_zone/output/"
output_folder = "/Users/jeff.doyle/work/activitysim-data/sandag_zone/output/"

data_file = input_folder + "NetworkData.h5"
skims_file = input_folder + 'bikelogsum.omx'



def create_subset(data_file_out, skims_file_out, maxZone, households_sample_size=0):

    with pd.HDFStore(data_file, mode='r') as hdf:

        taz_df = hdf['/TAZ']
        taz_df = taz_df[taz_df.TAZ <= maxZone]
        taz_df.to_hdf(data_file_out, 'TAZ')

        tap_df = hdf['/TAP']
        tap_df = tap_df[tap_df.TAZ <= maxZone]
        tap_df.to_hdf(data_file_out, 'TAP')

        maz_df = hdf['/MAZ']
        maz_df = maz_df[maz_df.TAZ <= maxZone]
        maz_df.to_hdf(data_file_out, 'MAZ')

        maz2tap_df = hdf['/MAZtoTAP']
        maz2tap_df = maz2tap_df[maz2tap_df.TAZ <= maxZone]
        maz2tap_df.to_hdf(data_file_out, 'MAZtoTAP')

        maz2maz_df = hdf['/MAZtoMAZ']
        x = (maz2maz_df.DMAZ in maz_df.index)
        maz2maz_df = maz2tap_df[maz2tap_df.TAZ <= maxZone]
        maz2maz_df.to_hdf(data_file_out, 'MAZtoMAZ')

        # print "taz_df.ptype.unique()", taz_df.ptype.unique()
        # print "taz_df.ptype.value_counts()", taz_df.ptype.value_counts()
        # print "taz_df.terminal_time.value_counts()", taz_df.terminal_time.value_counts()

        for key in hdf.keys():
            df = hdf[key]

            print "\n========== %s\n" % key
            print df.columns.values

    # print 'households'
    # df = pd.read_hdf(data_file, 'households')
    # df = df[df.TAZ <= maxZone]
    # if households_sample_size:
    #     df = df.take(np.random.choice(len(df), size=households_sample_size, replace=False))
    # df.to_hdf(data_file_out, 'households')
    #
    # print 'persons'
    # per = pd.read_hdf(data_file, 'persons')
    # per = per[per.household_id.isin(df.index)]
    # per.to_hdf(data_file_out, 'persons')

    # process all skims
    skims = omx.open_file(skims_file)
    skims_out = omx.open_file(skims_file_out, 'a')

    skimsToProcess = skims.listMatrices()
    for skimName in skimsToProcess:
        print skimName
        # skims_out[skimName] = skims[skimName][0:maxZone, 0:maxZone]
        # skims_out[skimName].attrs.TITLE = ''  # remove funny character for OMX viewer


# create_subset(data_file_out='mtc_asim_sf.h5',
#               skims_file_out='skims_sf.omx',
#               maxZone=190
#               )
#
create_subset(data_file_out=output_folder+'NetworkData_test.h5',
              skims_file_out=output_folder+'bikelogsum_test.omx',
              maxZone=25
              )


