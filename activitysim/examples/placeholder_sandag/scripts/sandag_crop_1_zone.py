import argparse
import os

import numpy as np
import openmatrix as omx
import pandas as pd

segments = {
    "test": "../data_2/taz.csv",  # crop to match 2 zone TAZs, includes univ
    "full": (0, 100000),
}

parser = argparse.ArgumentParser(description="crop SANDAG 1 zone raw_data")
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
    help="check consistency of TAZ zone_ids and foreign keys & write orphan_households file",
)

args = parser.parse_args()


segment_name = args.segment_name[0]
check_geography = args.check_geography

assert segment_name in segments.keys(), f"Unknown seg: {segment_name}"
if isinstance(segments[segment_name], str):
    zone_df = pd.read_csv(segments[segment_name])
    zones = zone_df["TAZ"].values
else:
    zone_min, zone_max = segments[segment_name]
    zones = range(zone_min, zone_max + 1)

input_dir = "./data_raw"
output_dir = f"./data_{segment_name}_1"


print(f"check_geography {check_geography}")

if not os.path.isdir(output_dir):
    print(f"creating output directory {output_dir}")
    os.mkdir(output_dir)


def input_path(file_name):
    return os.path.join(input_dir, file_name)


def output_path(file_name):
    return os.path.join(output_dir, file_name)


def integerize_id_columns(df, table_name):
    columns = ["TAZ", "household_id", "HHID", "taz"]
    for c in df.columns:
        if c in columns:
            print(f"converting {table_name}.{c} to int")
            if df[c].isnull().any():
                print(df[c][df[c].isnull()])
            df[c] = df[c].astype(int)


def read_csv(file_name, integerize=True):
    df = pd.read_csv(input_path(file_name))

    print(f"read {file_name} {df.shape}")

    return df


def to_csv(df, file_name):
    print(f"writing {file_name} {df.shape} {output_path(file_name)}")
    df.to_csv(output_path(file_name), index=False)


if check_geography:

    # ######## check for orphan_households not in any taz in land_use
    land_use = read_csv("land_use.csv")

    households = read_csv("households.csv")
    orphan_households = households[~households.TAZ.isin(land_use.TAZ)]
    print(f"{len(orphan_households)} orphan_households")

    # write orphan_households to INPUT directory (since it doesn't belong in output)
    file_name = "orphan_households.csv"
    print(f"writing {file_name} {orphan_households.shape} to {input_path(file_name)}")
    orphan_households.to_csv(input_path(file_name), index=False)


#
# land_use
#
land_use = read_csv("land_use.csv")
land_use = land_use[land_use["TAZ"].isin(zones)]
integerize_id_columns(land_use, "land_use")
land_use = land_use.sort_values("TAZ")

# move index col to front
land_use.insert(0, "TAZ", land_use.pop("TAZ"))

to_csv(land_use, "land_use.csv")

#
# households
#
households = read_csv("households.csv")
households = households[households["TAZ"].isin(land_use.TAZ)]
integerize_id_columns(households, "households")

to_csv(households, "households.csv")

#
# persons
#
persons = read_csv("persons.csv")
persons = persons[persons["household_id"].isin(households.HHID)]
integerize_id_columns(persons, "persons")

to_csv(persons, "persons.csv")

#
# skims
#
omx_infile_name = "skims.omx"
skim_data_type = np.float32

omx_in = omx.open_file(input_path(omx_infile_name))
print(f"omx_in shape {omx_in.shape()}")


zone = land_use.sort_values("TAZ")[["TAZ"]]
zone.index = zone.TAZ - 1
zone_indexes = zone.index.tolist()  # index of TAZ in skim (zero-based, no mapping)
zone_labels = zone.TAZ.tolist()  # TAZ in omx index order


# create
num_outfiles = 6 if segment_name == "full" else 1
if num_outfiles == 1:
    omx_out = [omx.open_file(output_path(f"skims1.omx"), "w")]
else:
    omx_out = [
        omx.open_file(output_path(f"skims{i+1}.omx"), "w") for i in range(num_outfiles)
    ]

for omx_file in omx_out:
    omx_file.create_mapping("ZONE", zone_labels)

iskim = 0
for mat_name in omx_in.list_matrices():

    # make sure we have a vanilla numpy array, not a CArray
    m = np.asanyarray(omx_in[mat_name]).astype(skim_data_type)
    m = m[zone_indexes, :][:, zone_indexes]
    print(f"{mat_name} {m.shape}")

    omx_file = omx_out[iskim % num_outfiles]
    omx_file[mat_name] = m
    iskim += 1


omx_in.close()
for omx_file in omx_out:
    omx_file.close()
