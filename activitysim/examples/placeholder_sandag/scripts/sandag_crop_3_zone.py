import argparse
import os

import numpy as np
import openmatrix as omx
import pandas as pd

MAZ_OFFSET = 0

segments = {
    "test": {"MAZ": np.arange(MAZ_OFFSET + 492, MAZ_OFFSET + 1101)},  # includes univ
    "univ_east": {"MAZ": np.arange(MAZ_OFFSET, MAZ_OFFSET + 1080)},
    "full": {},
}

parser = argparse.ArgumentParser(description="crop SANDAG 3 zone raw_data")
parser.add_argument(
    "segment_name",
    metavar="segment_name",
    type=str,
    nargs=1,
    help=f"geography segmentation (e.g. full)",
)

parser.add_argument(
    "-c",
    "--check_geography",
    default=False,
    action="store_true",
    help="check consistency of MAZ, TAZ, TAP zone_ids and foreign keys & write orphan_households file",
)

args = parser.parse_args()


segment_name = args.segment_name[0]
check_geography = args.check_geography

assert segment_name in segments.keys(), f"Unknown seg: {segment_name}"

input_dir = "./data_raw"
output_dir = f"./data_{segment_name}_3"


print(f"segment_name {segment_name}")

print(f"input_dir {input_dir}")
print(f"output_dir {output_dir}")

print(f"check_geography {check_geography}")

if not os.path.isdir(output_dir):
    print(f"creating output directory {output_dir}")
    os.mkdir(output_dir)


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
    print(f"read {file_name} {df.shape}")
    return df


def to_csv(df, file_name):
    df.to_csv(output_path(file_name), index=False)
    print(f"write {file_name} {df.shape}")


def crop_omx(omx_file_name, zones, num_outfiles=1):

    skim_data_type = np.float32

    omx_in = omx.open_file(input_path(f"{omx_file_name}.omx"))
    print(f"omx_in shape {omx_in.shape()}")

    offset_map = None
    for mapping_name in omx_in.listMappings():
        _offset_map = np.asanyarray(omx_in.mapentries(mapping_name))
        offset_map = _offset_map

    om = pd.Series(offset_map)
    om = om[om.isin(zones.values)]
    indexes = om.index.values

    labels = zones.values  # TAZ zone_ids in omx index order

    # create
    if num_outfiles == 1:
        omx_out = [omx.open_file(output_path(f"{omx_file_name}.omx"), "w")]
    else:
        omx_out = [
            omx.open_file(output_path(f"{omx_file_name}{i + 1}.omx"), "w")
            for i in range(num_outfiles)
        ]

    for omx_file in omx_out:
        omx_file.create_mapping("ZONE", labels)

    iskim = 0
    for mat_name in omx_in.list_matrices():
        # make sure we have a vanilla numpy array, not a CArray
        m = np.asanyarray(omx_in[mat_name]).astype(skim_data_type)
        m = m[indexes, :][:, indexes]
        print(f"{mat_name} {m.shape}")

        omx_file = omx_out[iskim % num_outfiles]
        omx_file[mat_name] = m
        iskim += 1

    omx_in.close()
    for omx_file in omx_out:
        omx_file.close()


# non-standard input file names

LAND_USE = "land_use.csv"
HOUSEHOLDS = "households.csv"
PERSONS = "persons.csv"
MAZ_TAZ = "maz.csv"
TAP_MAZ = "tap.csv"
TAZ = "taz.csv"


if check_geography:

    # ######## check for orphan_households not in any maz in land_use
    land_use = read_csv(LAND_USE)
    land_use = land_use[["MAZ", "TAZ"]]
    land_use = land_use.sort_values(["TAZ", "MAZ"])

    households = read_csv(HOUSEHOLDS)
    orphan_households = households[~households.MAZ.isin(land_use.MAZ)]
    print(f"{len(orphan_households)} orphan_households")

    # write orphan_households to INPUT directory (since it doesn't belong in output)
    if len(orphan_households) > 0:
        file_name = "orphan_households.csv"
        print(
            f"writing {file_name} {orphan_households.shape} to {input_path(file_name)}"
        )
        orphan_households.to_csv(input_path(file_name), index=False)

    # ######## check that land_use and maz and taz tables have same MAZs and TAZs

    # could just build maz and taz files, but want to make sure PSRC data is right

    land_use = read_csv(LAND_USE)
    # assert land_use.set_index('MAZ').index.is_monotonic_increasing

    land_use = land_use.sort_values("MAZ")
    maz = read_csv(MAZ_TAZ).sort_values("MAZ")

    # ### FATAL ###
    if not land_use.MAZ.isin(maz.MAZ).all():
        print(
            f"land_use.MAZ not in maz.MAZ\n{land_use.MAZ[~land_use.MAZ.isin(maz.MAZ)]}"
        )
        raise RuntimeError(f"land_use.MAZ not in maz.MAZ")

    if not maz.MAZ.isin(land_use.MAZ).all():
        print(f"maz.MAZ not in land_use.MAZ\n{maz.MAZ[~maz.MAZ.isin(land_use.MAZ)]}")

    # ### FATAL ###
    if not land_use.TAZ.isin(maz.TAZ).all():
        print(
            f"land_use.TAZ not in maz.TAZ\n{land_use.TAZ[~land_use.TAZ.isin(maz.TAZ)]}"
        )
        raise RuntimeError(f"land_use.TAZ not in maz.TAZ")

    if not maz.TAZ.isin(land_use.TAZ).all():
        print(f"maz.TAZ not in land_use.TAZ\n{maz.TAZ[~maz.TAZ.isin(land_use.TAZ)]}")


# land_use

land_use = read_csv(LAND_USE)

land_use.MAZ = land_use.MAZ.astype(int)

ur_land_use = land_use.copy()

slicer = segments[segment_name]
for slice_col, slice_values in slicer.items():
    # print(f"slice {slice_col}: {slice_values}")
    land_use = land_use[land_use[slice_col].isin(slice_values)]

print(f"land_use shape after slicing {land_use.shape}")
to_csv(land_use, "land_use.csv")


# TAZ

taz = pd.DataFrame({"TAZ": sorted(ur_land_use.TAZ.unique())})
taz = taz[taz.TAZ.isin(land_use["TAZ"])]
to_csv(taz, TAZ)


# maz_taz


maz_taz = read_csv(MAZ_TAZ).sort_values("MAZ")
maz_taz = maz_taz[maz_taz.MAZ.isin(land_use.MAZ)]
to_csv(maz_taz, MAZ_TAZ)

# tap

taps = read_csv(TAP_MAZ)
taps = taps[["TAP", "MAZ"]].sort_values(by="TAP").reset_index(drop=True)
taps = taps[taps["MAZ"].isin(land_use["MAZ"])]
to_csv(taps, "tap.csv")

# maz to tap

maz_tap_walk = read_csv("maz_to_tap_walk.csv").sort_values(["MAZ", "TAP"])
taz_tap_drive = read_csv("maz_to_tap_drive.csv").sort_values(["MAZ", "TAP"])

maz_tap_walk = maz_tap_walk[
    maz_tap_walk["MAZ"].isin(land_use["MAZ"]) & maz_tap_walk["TAP"].isin(taps["TAP"])
]
taz_tap_drive = taz_tap_drive[
    taz_tap_drive["MAZ"].isin(land_use["MAZ"]) & taz_tap_drive["TAP"].isin(taps["TAP"])
]

to_csv(maz_tap_walk, "maz_to_tap_walk.csv")
to_csv(taz_tap_drive, "maz_to_tap_drive.csv")

# maz to mz

maz_maz_walk = read_csv("maz_to_maz_walk.csv").sort_values(["OMAZ", "DMAZ"])
maz_maz_bike = read_csv("maz_to_maz_bike.csv").sort_values(["OMAZ", "DMAZ"])

maz_maz_walk = maz_maz_walk[
    maz_maz_walk["OMAZ"].isin(land_use["MAZ"])
    & maz_maz_walk["DMAZ"].isin(land_use["MAZ"])
]
maz_maz_bike = maz_maz_bike[
    maz_maz_bike["OMAZ"].isin(land_use["MAZ"])
    & maz_maz_bike["DMAZ"].isin(land_use["MAZ"])
]

to_csv(maz_maz_walk, "maz_to_maz_walk.csv")
to_csv(maz_maz_bike, "maz_to_maz_bike.csv")

# tap_lines

tap_lines = read_csv("tap_lines.csv")
tap_lines = tap_lines[tap_lines["TAP"].isin(taps["TAP"])]
to_csv(tap_lines, "tap_lines.csv")

# households

households = read_csv(HOUSEHOLDS)
households = households[households["MAZ"].isin(land_use["MAZ"])]
to_csv(households, "households.csv")

# persons

persons = read_csv(PERSONS)
persons = persons[persons["household_id"].isin(households["HHID"])]
to_csv(persons, "persons.csv")

# skims

crop_omx("taz_skims1", taz.TAZ, num_outfiles=(4 if segment_name == "full" else 1))
crop_omx("tap_skims1", taps.TAP, num_outfiles=(4 if segment_name == "full" else 1))
