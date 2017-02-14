# ActivitySim
# See full license in LICENSE.txt.

import numpy as np
import pandas as pd
import openmatrix as omx

input_folder = "/Users/jeff.doyle/work/activitysim-data/sandag_zone/output/"
output_folder = "/Users/jeff.doyle/work/activitysim-data/sandag_zone/output/"

data_file = input_folder + "NetworkData.h5"
skims_file = input_folder + 'bikelogsum.omx'



def create_subset(output_folder, data_file_out, skims_file_out, maxZone, households_sample_size=0):

    data_file_out = output_folder + data_file_out
    skims_file_out = output_folder + skims_file_out

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
        x = maz2maz_df.OMAZ.isin(maz_df.index) & maz2maz_df.DMAZ.isin(maz_df.index)
        maz2maz_df = maz2maz_df[x]
        maz2maz_df.to_hdf(data_file_out, 'MAZtoMAZ')

        for key in hdf.keys():
            df = hdf[key]
            print "\n========== %s\n" % key
            print df.columns.values

    # process all skims
    skims = omx.open_file(skims_file)
    skims_out = omx.open_file(skims_file_out, 'a')

    skimsToProcess = skims.listMatrices()
    for skimName in skimsToProcess:
        print skimName
        skims_out[skimName] = skims[skimName][0:maxZone, 0:maxZone]
        skims_out[skimName].attrs.TITLE = ''  # remove funny character for OMX viewer

def dump_subset(folder, data_file, skim_files):

    if data_file:
        with pd.HDFStore(folder+data_file, mode='r') as hdf:

            df = hdf['/TAZ']
            df.to_csv(folder+'taz.csv', index=True)

            df = hdf['/TAP']
            df.to_csv(folder+'tap.csv', index=True)

            for key in hdf.keys():
                print "\n========== %s\n" % key
                df = hdf[key]

                print "len", len(df.index)

                print df.columns.values

                for c in ['TAZ', 'TAP', 'MAZ', 'OMAZ', 'DMAZ']:
                    if c in df.columns:
                        print "%s min: %s max: %s" % (c, df[c].min(), df[c].max())

                if 'TAZ'in df.columns:
                    print df.TAZ.value_counts().head(20)
                #print df


    # process all skims
    for skim_file in skim_files:
        with omx.open_file(folder+skim_file) as skims:
            #skims = omx.open_file(folder+skim_file)
            skimsToProcess = skims.listMatrices()
            print "\n##### %s %s" % (skim_file, skims.shape())
            for skimName in skimsToProcess:
                print skimName
            #skims.close()


#
# create_subset(output_folder=output_folder,
#               data_file_out='NetworkData_example.h5',
#               skims_file_out='bikelogsum_example.omx',
#               maxZone=190
#               )
#
# create_subset(output_folder=output_folder,
#               data_file_out='NetworkData_test.h5',
#               skims_file_out='bikelogsum_test.omx',
#               maxZone=25
#               )

skim_files=['bikelogsum.omx']

dump_subset(folder=output_folder,
            data_file='NetworkData.h5',
            skim_files=[]
              )

# taz_skim_files=['impdan_AM.omx', 'impdat_AM.omx', 'Trip_AM.omx']
#
# tap_skim_files=['implocl_AM.omx', 'implocl_AMo.omx', 'impprem_AM.omx', 'impprem_AMo.omx', 'tranTotalTrips_AM.omx',]
#
#
# dump_subset(folder="/Users/jeff.doyle/work/activitysim-data/sandag_zone/",
#             data_file='',
#             skim_files=taz_skim_files
#               )
#
# print "\n#### other ###\n"
#
# dump_subset(folder="/Users/jeff.doyle/work/activitysim-data/sandag_zone/",
#             data_file='',
#             skim_files=tap_skim_files
#               )
