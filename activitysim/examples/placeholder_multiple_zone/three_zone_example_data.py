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
output_data = os.path.join(os.path.dirname(__file__), "data_3")
MAZ_MULTIPLIER = 1000
TAP_OFFSET = 90000

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

# ### Create maz file
# one row per maz, currentlyt he only attribute it its containing TAZ

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

taz_zone_ids = np.unique(land_use.TAZ)
taz_zone_indexes = taz_zone_ids - 1
taz_df = pd.DataFrame({"TAZ": taz_zone_ids}, index=taz_zone_indexes)
taz_df.to_csv(os.path.join(output_data, "taz.csv"), index=False)
print("taz.csv\n%s" % (taz_df.head(6),))

# currently this has only the one TAZ column, but the legacy table had:
# index TAZ
# offset             int64
# terminal_time    float64  # occasional small integer (1-5), but mostly blank (only if it has a TAP?
# ptype            float64  # parking type at TAP? (rarer than terminal_time, never alone)


# ### Create maz to maz time/distance

max_distance_for_walk = 1.0
max_distance_for_bike = 5.0


with omx.open_file(os.path.join(input_data, "skims.omx")) as ur_skims:

    # create df with DIST column
    maz_to_maz = pd.DataFrame(ur_skims["DIST"]).unstack().reset_index()
    maz_to_maz.columns = ["OMAZ", "DMAZ", "DIST"]
    maz_to_maz["OMAZ"] = (maz_to_maz["OMAZ"] + 1) * MAZ_MULTIPLIER
    maz_to_maz["DMAZ"] = (maz_to_maz["DMAZ"] + 1) * MAZ_MULTIPLIER

    # additional columns
    for c in ["DISTBIKE", "DISTWALK"]:
        maz_to_maz[c] = pd.DataFrame(ur_skims[c]).unstack().values

    maz_to_maz.loc[
        maz_to_maz["DIST"] <= max_distance_for_walk, ["OMAZ", "DMAZ", "DISTWALK"]
    ].to_csv(os.path.join(output_data, "maz_to_maz_walk.csv"), index=False)

    maz_to_maz.loc[
        maz_to_maz["DIST"] <= max_distance_for_bike,
        ["OMAZ", "DMAZ", "DIST", "DISTBIKE"],
    ].to_csv(os.path.join(output_data, "maz_to_maz_bike.csv"), index=False)


########


# create tap file
# currently the only attribute is its containing maz

taz_zone_labels = taz_df.TAZ.values
tap_zone_labels = taz_zone_labels + TAP_OFFSET
maz_zone_labels = taz_zone_labels * MAZ_MULTIPLIER
tap_df = pd.DataFrame({"TAP": tap_zone_labels, "MAZ": maz_zone_labels})
tap_df.to_csv(os.path.join(output_data, "tap.csv"), index=False)

# create taz_z3 and tap skims
with omx.open_file(
    os.path.join(input_data, "skims.omx"), "r"
) as ur_skims, omx.open_file(
    os.path.join(output_data, "taz_skims.omx"), "w"
) as output_taz_skims_file, omx.open_file(
    os.path.join(output_data, "tap_skims.omx"), "w"
) as output_tap_skims_file:

    for skim_name in ur_skims.list_matrices():

        ur_skim = ur_skims[skim_name][:]
        new_skim = ur_skim[taz_zone_indexes, :][:, taz_zone_indexes]
        # print("skim:", skim_name, ": shape", str(new_skim.shape))

        mode_code = skim_name[0:3]
        is_tap_mode = mode_code == "DRV" or mode_code == "WLK"
        is_taz_mode = not is_tap_mode

        if is_tap_mode:
            # WLK_TRN_WLK_XWAIT__PM
            # 012345678911111111112
            #           01234567890
            access_mode = skim_name[0:3]
            transit_mode = skim_name[4:7]
            egress_mode = skim_name[8:11]
            datum_name = skim_name[12:-4]
            tod = skim_name[-2:]
            if access_mode == "WLK" and egress_mode == "WLK":
                for suffix in ["FAST", "SHORT", "CHEAP"]:
                    if (suffix == "FAST") and (datum_name == "TOTIVT"):
                        random_variation = np.random.rand(*new_skim.shape) * -0.1 + 1.0
                    elif (suffix == "CHEAP") and (datum_name == "FAR"):
                        random_variation = np.random.rand(*new_skim.shape) * -0.5 + 1.0
                    else:
                        random_variation = np.ones_like(new_skim)

                    tap_skim_name = f"{transit_mode}_{datum_name}_{suffix}__{tod}"
                    output_tap_skims_file[tap_skim_name] = new_skim * random_variation
                    print(
                        f"tap skim: {skim_name} tap_skim_name: {tap_skim_name}, "
                        f"shape: {str(output_tap_skims_file.shape())}"
                    )

        if is_taz_mode:
            output_taz_skims_file[skim_name] = new_skim
            print("taz skim:", skim_name, ": shape", str(output_taz_skims_file.shape()))

    output_taz_skims_file.create_mapping("taz", taz_zone_labels)
    output_tap_skims_file.create_mapping("tap", tap_zone_labels)

# Create maz to tap distance file by mode

with omx.open_file(os.path.join(input_data, "skims.omx")) as ur_skims:
    distance_table = pd.DataFrame(np.transpose(ur_skims["DIST"])).unstack()
    distance_table = distance_table.reset_index()
    distance_table.columns = ["MAZ", "TAP", "DIST"]

    distance_table["drive_time"] = (
        pd.DataFrame(np.transpose(ur_skims["SOV_TIME__MD"])).unstack().values
    )

    for c in ["DISTBIKE", "DISTWALK"]:
        distance_table[c] = pd.DataFrame(np.transpose(ur_skims[c])).unstack().values

walk_speed = 3
bike_speed = 10
drive_speed = 25
max_distance_for_nearby_taps_walk = 1.0
max_distance_for_nearby_taps_bike = 5.0
max_distance_for_nearby_taps_drive = 10.0

distance_table["MAZ"] = (distance_table["MAZ"] + 1) * MAZ_MULTIPLIER
distance_table["TAP"] = (distance_table["TAP"] + 1) + TAP_OFFSET

distance_table["walk_time"] = distance_table["DIST"] * (60 / walk_speed)
distance_table["bike_time"] = distance_table["DIST"] * (60 * bike_speed)

# FIXME: we are using SOV_TIME__MD - is that right?
distance_table["drive_time"] = distance_table["DIST"] * (60 * drive_speed)

distance_table = distance_table[distance_table["TAP"].isin(tap_zone_labels)]


distance_table.loc[
    distance_table["DIST"] <= max_distance_for_nearby_taps_walk,
    ["MAZ", "TAP", "DISTWALK", "walk_time"],
].to_csv(os.path.join(output_data, "maz_to_tap_walk.csv"), index=False)

distance_table.loc[
    distance_table["DIST"] <= max_distance_for_nearby_taps_bike,
    ["MAZ", "TAP", "DISTBIKE", "bike_time"],
].to_csv(os.path.join(output_data, "maz_to_tap_bike.csv"), index=False)

distance_table.loc[
    distance_table["DIST"] <= max_distance_for_nearby_taps_drive,
    ["MAZ", "TAP", "DIST", "drive_time"],
].to_csv(os.path.join(output_data, "maz_to_tap_drive.csv"), index=False)
