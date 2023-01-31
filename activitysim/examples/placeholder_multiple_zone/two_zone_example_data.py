# Creating the Two Zone Example Data
#
# Transform the TM1 TAZ-based model 25 zone inputs to a two-zone (MAZ and TAZ) set of inputs for software development.
#
# The 25 zones are downtown San Francisco and they are converted to 25 MAZs.
# MAZs 1,2,3,4 are small and adjacent and assigned TAZ 2 and TAP 10002.
# MAZs 13,14,15 are small and adjacent and as signed TAZ 14 and TAP 10014.
# TAZs 1,3,4,13,15 are removed from the final data set.
#
# This script should work for the full TM1 example as well.

import os
import shutil

import numpy as np
import openmatrix as omx
import pandas as pd

# Create example directory

input_data = os.path.join(os.path.dirname(__file__), "data")
output_data = os.path.join(os.path.dirname(__file__), "data_2")
MAZ_MULTIPLIER = 1000

# ### initialize output data directory

# new empty output_dir
if os.path.exists(output_data):
    # shutil.rmtree(output_data)
    # os.makedirs(output_data)
    file_type = ("csv", "omx")
    for file_name in os.listdir(output_data):
        if file_name.endswith(file_type):
            os.unlink(os.path.join(output_data, file_name))
else:
    os.makedirs(output_data)

# ### Convert tazs to mazs and add transit access distance by mode

land_use = pd.read_csv(os.path.join(input_data, "land_use.csv"))

land_use.insert(loc=0, column="MAZ", value=land_use.ZONE)
land_use.insert(loc=1, column="TAZ", value=land_use.ZONE)
land_use.drop(columns=["ZONE"], inplace=True)

land_use.TAZ = land_use.TAZ.replace([1, 2, 3, 4], 2)
land_use.TAZ = land_use.TAZ.replace([13, 14, 15], 14)

# make MAZ indexes different from TAZ to drive MAZ/TAZ confusion errors and omisisons
land_use.MAZ *= MAZ_MULTIPLIER

shortWalk = 0.333  # the tm1 example assumes this distance for transit access
longWalk = 0.667
land_use["access_dist_transit"] = shortWalk

# FIXME - could assign longWalk where maz != taz, but then results wodl differe from one-zone
# land_use['access_dist_transit'] =\
#     np.where(land_use.TAZ*MAZ_MULTIPLIER==land_use.MAZ, shortWalk, longWalk)


land_use.to_csv(os.path.join(output_data, "land_use.csv"), index=False)

# ### Put households in mazs instead of tazs

households = pd.read_csv(os.path.join(input_data, "households.csv"))
households.rename(columns={"TAZ": "MAZ"}, inplace=True)
households.MAZ *= MAZ_MULTIPLIER
households.to_csv(os.path.join(output_data, "households.csv"), index=False)

persons = pd.read_csv(os.path.join(input_data, "persons.csv"))
persons.to_csv(os.path.join(output_data, "persons.csv"), index=False)

# ### Create maz correspondence file

# FIXME - not clear we need this
maz_df = land_use[["MAZ", "TAZ"]]
maz_df.to_csv(os.path.join(output_data, "maz.csv"), index=False)
print("maz.csv\n%s" % (maz_df.head(6),))

# ### Create taz file

# TAZ
# 2
# 5
# 6
# 7

new_zone_labels = np.unique(land_use.TAZ)
new_zone_indexes = new_zone_labels - 1
taz_df = pd.DataFrame({"TAZ": new_zone_labels}, index=new_zone_indexes)
taz_df.to_csv(os.path.join(output_data, "taz.csv"), index=False)
print("taz.csv\n%s" % (taz_df.head(6),))

# currently this has only the one TAZ column, but the legacy table had:
# index TAZ
# offset             int64
# terminal_time    float64  # occasional small integer (1-5), but mostly blank (only if it has a TAP?
# ptype            float64  # parking type at TAP? (rarer than terminal_time, never alone)

# ### Create taz skims

with omx.open_file(
    os.path.join(input_data, "skims.omx"), "r"
) as skims_file, omx.open_file(
    os.path.join(output_data, "taz_skims.omx"), "w"
) as output_skims_file:

    skims = skims_file.list_matrices()
    num_zones = skims_file.shape()[0]

    # assume zones labels were 1-based in skims file
    assert not skims_file.listMappings()
    assert num_zones == len(land_use)

    for skim_name in skims_file.list_matrices():

        old_skim = skims_file[skim_name][:]
        new_skim = old_skim[new_zone_indexes, :][:, new_zone_indexes]
        output_skims_file[skim_name] = new_skim
        print("skim:", skim_name, ": shape", str(new_skim.shape))

    output_skims_file.create_mapping("taz", new_zone_labels)


# ### Create maz to maz time/distance

max_distance_for_walk = 1.0
max_distance_for_bike = 5.0


with omx.open_file(os.path.join(input_data, "skims.omx")) as skims_file:

    # create df with DIST column
    maz_to_maz = pd.DataFrame(np.transpose(skims_file["DIST"])).unstack().reset_index()
    maz_to_maz.columns = ["OMAZ", "DMAZ", "DIST"]
    maz_to_maz["OMAZ"] = (maz_to_maz["OMAZ"] + 1) * MAZ_MULTIPLIER
    maz_to_maz["DMAZ"] = (maz_to_maz["DMAZ"] + 1) * MAZ_MULTIPLIER

    # additional columns
    for c in ["DISTBIKE", "DISTWALK"]:
        maz_to_maz[c] = pd.DataFrame(np.transpose(skims_file[c])).unstack().values

    maz_to_maz.loc[
        maz_to_maz["DIST"] <= max_distance_for_walk, ["OMAZ", "DMAZ", "DISTWALK"]
    ].to_csv(os.path.join(output_data, "maz_to_maz_walk.csv"), index=False)

    maz_to_maz.loc[
        maz_to_maz["DIST"] <= max_distance_for_bike,
        ["OMAZ", "DMAZ", "DIST", "DISTBIKE"],
    ].to_csv(os.path.join(output_data, "maz_to_maz_bike.csv"), index=False)
