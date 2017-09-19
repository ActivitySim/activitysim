# ActivitySim
# See full license in LICENSE.txt.

import numpy as np
import pandas as pd
import openmatrix as omx


input_folder = "/Users/jeff.doyle/work/activitysim-data/sandag_zone/output/"
output_folder = "./output/"

data_file = 'NetworkData.h5'
skim_files = ['taz_skims.omx', 'tap_skims_locl.omx', 'tap_skims_prem.omx']


if __name__ == "__main__":

    if data_file:
        with pd.HDFStore(input_folder+data_file, mode='r') as hdf:

            df = hdf['/TAZ']
            df.to_csv(output_folder+'taz.csv', index=True)

            df = hdf['/TAP']
            df.to_csv(output_folder+'tap.csv', index=True)

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
                # print df

    # process all skims
    for skim_file in skim_files:
        with omx.open_file(input_folder+skim_file) as skims:
            # skims = omx.open_file(folder+skim_file)

            print "\n##### %s %s" % (skim_file, skims.shape())

            print "mappings:", skims.listMappings()

            skimsToProcess = skims.listMatrices()
            for skimName in skimsToProcess:
                print skimName
            # skims.close()
