import os

import openmatrix
import pandas as pd


def patch_example_sandag_1_zone(example_dir):

    cwd = os.getcwd()
    try:
        os.chdir(example_dir)
        skims = openmatrix.open_file("data_1/skims1.omx", mode="a")
        skims_lookup = skims.root["lookup"]

        zone_name = None
        if len(skims_lookup._v_children) == 1:
            zone_name = list(skims_lookup._v_children)[0]
            zone_data = skims_lookup[zone_name]
            rezone = pd.Series(
                pd.RangeIndex(1, zone_data.shape[0] + 1),
                index=zone_data[:],
            )
        else:
            rezone = None

        if rezone is not None:
            households = pd.read_csv("data_1/households.csv")
            households["TAZ"] = households["TAZ"].map(rezone)
            households.to_csv("data_1/households.csv", index=False)

            land_use = pd.read_csv("data_1/land_use.csv")
            land_use["TAZ"] = land_use["TAZ"].map(rezone)
            land_use.to_csv("data_1/land_use.csv", index=False)

        if zone_name:
            skims_lookup[zone_name]._f_remove()

        skims.close()

    finally:
        os.chdir(cwd)
