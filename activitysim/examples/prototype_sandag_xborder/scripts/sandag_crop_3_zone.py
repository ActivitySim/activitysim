# adapted from
# crop marin tvpb example data processing to one county
# Ben Stabler, ben.stabler@rsginc.com, 09/17/20

import argparse
import glob
import os

import numpy as np
import openmatrix as omx
import pandas as pd

MAZ_OFFSET = 100000

segments = {
    "cropped": {"MAZ": np.arange(MAZ_OFFSET + 500, MAZ_OFFSET + 1080)},
    "full": {},
}

parser = argparse.ArgumentParser(description="crop SANDAG 3 zone raw_data")
parser.add_argument(
    "-s",
    "--segment_name",
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

assert segment_name in segments.keys(), f"Unknown seg: {segment_name}"

input_dir = "../data_raw"
output_dir = f"../data_{segment_name}"

print(f"segment_name {segment_name}")

print(f"input_dir {input_dir}")
print(f"output_dir {output_dir}")

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
LAND_USE = "mazs_xborder.csv"
HOUSEHOLDS = "households_xborder.csv"
PERSONS = "persons_xborder.csv"
TOURS = "tours_xborder.csv"
# MAZ_TAZ = "maz.csv"
# TAP_MAZ = "tap.csv"
TAZ = "taz.csv"


# land_use
land_use = read_csv(LAND_USE)
land_use.MAZ = land_use.MAZ.astype(int)
ur_land_use = land_use.copy()

slicer = segments[segment_name]
for slice_col, slice_values in slicer.items():
    # print(f"slice {slice_col}: {slice_values}")
    poe_mask = land_use["poe_id"] > -1  # preserve mazs with poe data
    slice_mask = land_use[slice_col].isin(slice_values)
    land_use = land_use[(poe_mask) | (slice_mask)]

print(f"land_use shape after slicing {land_use.shape}")
to_csv(land_use, LAND_USE)


# TAZ
taz = pd.DataFrame({"TAZ": sorted(ur_land_use.TAZ.unique())})
taz = taz[taz.TAZ.isin(land_use["TAZ"])]
# to_csv(taz, TAZ)

# # maz_taz
# maz_taz = read_csv(MAZ_TAZ).sort_values('MAZ')
# maz_taz = maz_taz[maz_taz.MAZ.isin(land_use.MAZ)]
# to_csv(maz_taz, MAZ_TAZ)

# # tap
# taps = read_csv(TAP_MAZ)
# taps = taps[['TAP', 'MAZ']].sort_values(by='TAP').reset_index(drop=True)
# taps = taps[taps["MAZ"].isin(land_use["MAZ"])]
# to_csv(taps, "tap.csv")

# maz to tap
maz_tap_walk = read_csv("maz_tap_walk.csv").sort_values(["MAZ", "TAP"])
maz_tap_walk = maz_tap_walk[maz_tap_walk["MAZ"].isin(land_use["MAZ"])]
to_csv(maz_tap_walk, "maz_tap_walk.csv")

# maz to maz
maz_maz_walk = read_csv("maz_maz_walk.csv").sort_values(["OMAZ", "DMAZ"])
maz_maz_walk = maz_maz_walk[
    maz_maz_walk["OMAZ"].isin(land_use["MAZ"])
    & maz_maz_walk["DMAZ"].isin(land_use["MAZ"])
]
to_csv(maz_maz_walk, "maz_maz_walk.csv")

# taps and tap_lines
tap_lines = read_csv("tap_lines.csv")
tap_lines = tap_lines[tap_lines["TAP"].isin(maz_tap_walk["TAP"])]
to_csv(tap_lines, "tap_lines.csv")

taps = read_csv("taps.csv")
taps = taps[taps["TAP"].isin(maz_tap_walk["TAP"])]
to_csv(taps, "taps.csv")

tours = read_csv(TOURS)
to_csv(tours, "tours.csv")

# households
households = read_csv(HOUSEHOLDS)
# households = households[households["MAZ"].isin(land_use["MAZ"])]
to_csv(households, "households.csv")

# persons
persons = read_csv(PERSONS)
# persons = persons[persons["household_id"].isin(households["HHID"])]
to_csv(persons, "persons.csv")

# drive skims
for omx_fpath in glob.glob(input_path("*traffic*xborder*.omx")):
    print(omx_fpath)
    omx_fname = omx_fpath.replace("\\", "/").split("/")[-1].split(".omx")[0]
    crop_omx(omx_fname, taz.TAZ)

# transit skims
crop_omx("transit_skims_xborder", taps.TAP)
