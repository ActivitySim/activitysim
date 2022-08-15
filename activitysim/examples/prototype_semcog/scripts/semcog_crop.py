import argparse
import os

import numpy as np
import openmatrix as omx
import pandas as pd

MAZ_OFFSET = 0

segments = {
    "test": (149, 215),  # SUPER_DIST_25==1, has univ
    "z500": (0, 500),
    "full": (0, 10000),
}

land_use_zone_col = "ZONE"
hh_zone_col = "zone_id"
num_full_skim_files = 2

parser = argparse.ArgumentParser(description="crop PSRC raw_data")
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
    help="check consistency of MAZ, TAZ zone_ids and foreign keys & write orphan_households file",
)

args = parser.parse_args()


segment_name = args.segment_name[0]
check_geography = args.check_geography

assert segment_name in segments.keys(), f"Unknown seg: {segment_name}"
zone_min, zone_max = segments[segment_name]

input_dir = "./data_raw"
output_dir = f"./data_{segment_name}"


print(f"check_geography {check_geography}")

if not os.path.isdir(output_dir):
    print(f"creating output directory {output_dir}")
    os.mkdir(output_dir)


def input_path(file_name):
    return os.path.join(input_dir, file_name)


def output_path(file_name):
    return os.path.join(output_dir, file_name)


def integerize_id_columns(df, table_name):
    columns = [
        "ZONE",
        "SUPER_DIST_25",
        "zone_id",
        "household_id",
        "person_id",
        "MAZ",
        "TAZ",
    ]
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

    # ######## check for orphan_households not in any maz in land_use
    land_use = read_csv("land_use.csv")

    households = read_csv("households.csv")
    orphan_households = households[
        ~households[hh_zone_col].isin(land_use[land_use_zone_col])
    ]
    print(f"{len(orphan_households)} orphan_households")

    if len(orphan_households) > 0:
        # write orphan_households to INPUT directory (since it doesn't belong in output)
        file_name = "orphan_households.csv"
        print(
            f"writing {file_name} {orphan_households.shape} to {input_path(file_name)}"
        )
        orphan_households.to_csv(input_path(file_name), index=False)


#
# land_use
#
land_use = read_csv("land_use.csv")
land_use = land_use[
    (land_use[land_use_zone_col] >= zone_min)
    & (land_use[land_use_zone_col] <= zone_max)
]
integerize_id_columns(land_use, "land_use")
land_use = land_use.sort_values(land_use_zone_col)

# move index col to front
land_use.insert(0, land_use_zone_col, land_use.pop(land_use_zone_col))

to_csv(land_use, "land_use.csv")

# # make sure we have some HSENROLL and COLLFTE, even for very for small samples
# if land_use['HSENROLL'].sum() == 0:
#     assert segment_name != 'full', f"land_use['HSENROLL'] is 0 for full sample!"
#     land_use['HSENROLL'] = land_use['AGE0519']
#     print(f"\nWARNING: land_use.HSENROLL is 0, so backfilled with AGE0519\n")
#
# if land_use['COLLFTE'].sum() == 0:
#     assert segment_name != 'full', f"land_use['COLLFTE'] is 0 for full sample!"
#     land_use['COLLFTE'] = land_use['HSENROLL']
#     print(f"\nWARNING: land_use.COLLFTE is 0, so backfilled with HSENROLL\n")

#
# households
#
households = read_csv("households.csv")
households = households[households[hh_zone_col].isin(land_use[land_use_zone_col])]
integerize_id_columns(households, "households")

# move index col to front
households.insert(0, "household_id", households.pop("household_id"))

to_csv(households, "households.csv")

#
# persons
#
persons = read_csv("persons.csv")
persons = persons[persons["household_id"].isin(households.household_id)]
integerize_id_columns(persons, "persons")

# move index col to front
persons.insert(0, "person_id", persons.pop("person_id"))


to_csv(persons, "persons.csv")

#
# skims
#
omx_infile_name = "skims.omx"
skim_data_type = np.float32

omx_in = omx.open_file(input_path(omx_infile_name))
print(f"omx_in shape {omx_in.shape()}")

for m in omx_in.listMappings():
    offset_map = omx_in.mapentries(m)
    # otherwise we will have to offset_map zone_indexes we use to slice skim below
    assert (offset_map == np.arange(len(offset_map)) + 1).all()
    # print(f"{m}\n{offset_map}")

zone = land_use.sort_values(land_use_zone_col)[[land_use_zone_col]]
zone.index = zone[land_use_zone_col] - 1
zone_indexes = zone.index.tolist()  # index of TAZ in skim (zero-based, no mapping)
zone_labels = zone[land_use_zone_col].tolist()  # TAZ zone_ids in omx index order

# create
num_outfiles = num_full_skim_files if segment_name == "full" else 1
if num_outfiles == 1:
    omx_out = [omx.open_file(output_path(f"skims.omx"), "w")]
else:
    omx_out = [
        omx.open_file(output_path(f"skims{i+1}.omx"), "w") for i in range(num_outfiles)
    ]

for omx_file in omx_out:
    omx_file.create_mapping(land_use_zone_col, zone_labels)

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
