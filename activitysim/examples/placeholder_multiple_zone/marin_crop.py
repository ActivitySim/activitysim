# crop marin tvpb example data processing to one county
# Ben Stabler, ben.stabler@rsginc.com, 09/17/20
# jeff doyle added code to introduce MAZ_OFFSET to avoid confusion (and detect associated errors) between zone types

import os

import openmatrix as omx
import pandas as pd

# counties = ["Marin"
# counties = ["San Francisco"
counties = ["Marin", "San Francisco"]

input_dir = "./data_3_marin"
output_dir = "./data_3_marin/crop"
MAZ_OFFSET = 100000


def input_path(file_name):
    return os.path.join(input_dir, file_name)


def output_path(file_name):
    return os.path.join(output_dir, file_name)


def patch_maz(df, maz_offset):
    for c in df.columns:
        if c in ["MAZ", "OMAZ", "DMAZ", "mgra", "orig_mgra", "dest_mgra"]:
            df[c] += maz_offset
    return df


def read_csv(file_name):
    df = pd.read_csv(input_path(file_name))
    if MAZ_OFFSET:
        df = patch_maz(df, MAZ_OFFSET)
        print(f"\n\n{file_name}\n{df}")
    return df


def to_csv(df, file_name):
    df.to_csv(output_path(file_name), index=False)


#########
# mazs = read_csv("maz_data_asim.csv")
# taps = read_csv("tap_data.csv")
#
# print(f"max maz {mazs.MAZ.max()}")
# print(f"num maz {len(mazs.MAZ.unique())}")
# print(f"num taz {len(mazs.TAZ.unique())}")
# print(f"num tap {len(taps.TAP.unique())}")
#
# num maz 5952
# num taz 4735
# num tap 6216
#########


# 0 - get county zones

mazs = read_csv("maz_data_asim.csv")

mazs = mazs[mazs["CountyName"].isin(counties)]
to_csv(mazs, "maz_data_asim.csv")

maz_taz = mazs[["MAZ", "TAZ"]]
to_csv(mazs, "maz_taz.csv")

tazs = mazs["TAZ"].unique()
tazs.sort()
tazs_indexes = (tazs - 1).tolist()

taps = read_csv("tap_data.csv")
taps = taps[["TAP", "TAZ"]].sort_values(by="TAP")
taps = taps[taps["TAZ"].isin(tazs)]
to_csv(taps, "tap_data.csv")

# 1-based tap_ids
taps_indexes = (taps["TAP"] - 1).tolist()


# 2 - maz to tap walk, bike

maz_tap_walk = read_csv("maz_tap_walk.csv")
maz_maz_walk = read_csv("maz_maz_walk.csv")
maz_maz_bike = read_csv("maz_maz_bike.csv")

maz_tap_walk = maz_tap_walk[
    maz_tap_walk["MAZ"].isin(mazs["MAZ"]) & maz_tap_walk["TAP"].isin(taps["TAP"])
]
maz_maz_walk = maz_maz_walk[
    maz_maz_walk["OMAZ"].isin(mazs["MAZ"]) & maz_maz_walk["DMAZ"].isin(mazs["MAZ"])
]
maz_maz_bike = maz_maz_bike[
    maz_maz_bike["OMAZ"].isin(mazs["MAZ"]) & maz_maz_bike["DMAZ"].isin(mazs["MAZ"])
]

to_csv(maz_tap_walk, "maz_tap_walk.csv")
to_csv(maz_maz_walk, "maz_maz_walk.csv")
to_csv(maz_maz_bike, "maz_maz_bike.csv")


tap_lines = read_csv("tap_lines.csv")
tap_lines = tap_lines[tap_lines["TAP"].isin(taps["TAP"])]
to_csv(tap_lines, "tap_lines.csv")

# taz to tap drive data

taz_tap_drive = read_csv("maz_taz_tap_drive.csv")
taz_tap_drive = taz_tap_drive[
    taz_tap_drive["MAZ"].isin(mazs["MAZ"]) & taz_tap_drive["TAP"].isin(taps["TAP"])
]
to_csv(taz_tap_drive, "maz_taz_tap_drive.csv")


# 3 - accessibility data

access = read_csv("access.csv")
access = access[access["mgra"].isin(mazs["MAZ"])]
to_csv(access, "access.csv")


# households

households = read_csv("households_asim.csv")
households = households[households["MAZ"].isin(mazs["MAZ"])]
to_csv(households, "households_asim.csv")

# persons

persons = read_csv("persons_asim.csv")
persons = persons[persons["HHID"].isin(households["HHID"])]
to_csv(persons, "persons_asim.csv")

# tours file

work_tours = read_csv("work_tours.csv")
work_tours = work_tours[work_tours["hh_id"].isin(households["HHID"])]
work_tours = work_tours[
    work_tours["orig_mgra"].isin(mazs["MAZ"])
    & work_tours["dest_mgra"].isin(mazs["MAZ"])
]
to_csv(work_tours, "work_tours.csv")

# skims

time_periods = ["AM", "EA", "EV", "MD", "PM"]
for tp in time_periods:
    omx_file_name = "HWYSKM" + tp + "_taz_rename.omx"
    taz_file = omx.open_file(input_path(omx_file_name))
    taz_file_rename = omx.open_file(output_path(omx_file_name), "w")
    taz_file_rename.create_mapping("ZONE", tazs.tolist())
    for mat_name in taz_file.list_matrices():
        taz_file_rename[mat_name] = taz_file[mat_name][tazs_indexes, :][:, tazs_indexes]
        print(mat_name)
    taz_file.close()
    taz_file_rename.close()

for tp in time_periods:
    for skim_set in ["SET1", "SET2", "SET3"]:
        omx_file_name = "transit_skims_" + tp + "_" + skim_set + "_rename.omx"
        tap_file = omx.open_file(input_path(omx_file_name))
        tap_file_rename = omx.open_file(output_path(omx_file_name), "w")
        tap_file_rename.create_mapping("ZONE", taps["TAP"].tolist())
        for mat_name in tap_file.list_matrices():
            tap_file_rename[mat_name] = tap_file[mat_name][taps_indexes, :][
                :, taps_indexes
            ]
            print(mat_name)
        tap_file.close()
        tap_file_rename.close()
