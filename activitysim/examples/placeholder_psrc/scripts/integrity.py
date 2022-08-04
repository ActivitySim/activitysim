import argparse
import os

import numpy as np
import openmatrix as omx
import pandas as pd

parser = argparse.ArgumentParser(description="check activitysim raw_data")
parser.add_argument(
    "raw_data_directory",
    metavar="raw_data_directory",
    type=str,
    nargs=1,
    help=f"path to raw data directory",
)

parser.add_argument(
    "-o", "--output", type=str, metavar="PATH", help="path to output dir"
)

args = parser.parse_args()


input_dir = args.raw_data_directory[0]
output_dir = args.output

print(f"input_dir {input_dir}")
print(f"output_dir {output_dir}")


def input_path(file_name):
    return os.path.join(input_dir, file_name)


def output_path(file_name):
    return os.path.join(output_dir, file_name)


def integerize_id_columns(df, table_name):
    columns = ["MAZ", "OMAZ", "DMAZ", "TAZ", "zone_id", "household_id", "HHID"]
    for c in df.columns:
        if c in columns:
            bad = ~(df[c] == df[c].astype(int))
            if bad.any():
                print(f"\n### OOPS ### table {table_name} bad integer column {c}\n")
            df[c] = df[c].astype(int)


def read_csv(file_name, integerize=True):
    df = pd.read_csv(input_path(file_name))

    print(f"read {file_name} {df.shape}")

    return df


def to_csv(df, file_name):
    print(f"writing {file_name} {df.shape} {output_path(file_name)}")
    df.to_csv(output_path(file_name), index=False)


def report_baddies(df, tag, fatal=False):

    if len(df) > 0:
        print(f"\n### OOPS ### {len(df)} {tag}\n")

        # print(f"\n{df}\n")

        if output_dir:
            file_name = f"{tag}.csv"
            print(f"writing {tag} {df.shape} to {output_path(file_name)}")
            df.to_csv(output_path(file_name), index=False)

        if fatal:
            raise RuntimeError(tag)
    else:
        print(f"{len(df)} {tag}")


print(f"input_dir {input_dir} output_dir {output_dir}")

if output_dir and not os.path.isdir(output_dir):
    print(f"creating output directory {output_dir}")
    os.mkdir(output_dir)


land_use = read_csv("land_use.csv")

# ### check maz.csv against land_use

land_use = land_use.sort_values("MAZ")
maz = read_csv("maz.csv").sort_values("MAZ")

# fatal
missing = land_use.MAZ[~land_use.MAZ.isin(maz.MAZ)]
report_baddies(missing, "land_use_MAZ_not_in_maz_MAZ", fatal=True)

missing = maz.MAZ[~maz.MAZ.isin(land_use.MAZ)]
report_baddies(missing, "maz_MAZ_not_in_land_use_MAZ")

# fatal
missing = land_use.TAZ[~land_use.TAZ.isin(maz.TAZ)]
report_baddies(missing, "land_use_TAZ_not_in_maz_TAZ", fatal=True)

missing = maz.TAZ[~maz.TAZ.isin(land_use.TAZ)]
report_baddies(missing, "maz_TAZ_not_in_land_use_TAZ")

# ### check taz.csv against land_use

land_use = land_use.sort_values("TAZ")
taz = read_csv("taz.csv").sort_values("TAZ")

if output_dir:
    taz.to_csv(output_path("taz.csv"), index=False)

# fatal
missing = land_use.TAZ[~land_use.TAZ.isin(taz.TAZ)]
report_baddies(missing, "land_use_TAZ_not_in_taz_TAZ", fatal=True)

missing = taz.TAZ[~taz.TAZ.isin(land_use.TAZ)]
report_baddies(missing, "taz_TAZ_not_in_land_use_TAZ")

# #########s

#
# maz
#
maz = read_csv("maz.csv").sort_values(["MAZ", "TAZ"])
maz = maz[maz["MAZ"].isin(land_use.MAZ)]
integerize_id_columns(maz, "maz")

assert land_use.MAZ.isin(maz.MAZ).all()
assert land_use.TAZ.isin(maz.TAZ).all()
assert maz.TAZ.isin(land_use.TAZ).all()

#
# taz
#
taz = read_csv("taz.csv").sort_values(["TAZ"])
taz = taz[taz["TAZ"].isin(land_use.TAZ)]
integerize_id_columns(taz, "taz")

assert land_use.TAZ.isin(taz.TAZ).all()

# print(maz.shape)
# print(f"MAZ {len(maz.MAZ.unique())}")
# print(f"TAZ {len(maz.TAZ.unique())}")

#
# households
#
households = read_csv("households.csv")
missing = households[~households["MAZ"].isin(maz.MAZ)]
report_baddies(missing, "household_MAZ_not_in_maz_MAZ")

integerize_id_columns(households, "households")

#
# persons
#
persons = read_csv("persons.csv")
orphans = persons[~persons["household_id"].isin(households.HHID)]
report_baddies(orphans, "persons_not_in_households")

households = households[households["MAZ"].isin(maz.MAZ)]
orphans = persons[~persons["household_id"].isin(households.HHID)]
report_baddies(orphans, "persons_not_in_households_in_maz_MAZ")

integerize_id_columns(persons, "persons")

#
# maz_to_maz_walk and maz_to_maz_bike
#

m2m = read_csv("maz_to_maz_walk.csv")
missing = m2m[~(m2m.OMAZ.isin(maz.MAZ) & m2m.DMAZ.isin(maz.MAZ))]
report_baddies(missing, "maz_to_maz_walk_OMAZ_or_DMAZ_not_in_maz_MAZ")
integerize_id_columns(m2m, "maz_to_maz_walk")

m2m = read_csv("maz_to_maz_bike.csv")
missing = m2m[~(m2m.OMAZ.isin(maz.MAZ) & m2m.DMAZ.isin(maz.MAZ))]
report_baddies(missing, "maz_to_maz_bike_OMAZ_or_DMAZ_not_in_maz_MAZ")
integerize_id_columns(m2m, "maz_to_maz_bike")


#
# skims
#
omx_infile_name = "skims.omx"
skim_data_type = np.float32

omx_in = omx.open_file(input_path(omx_infile_name), "r")
print(f"omx_in shape {omx_in.shape()}")

print(f"{len(omx_in.listMappings())} mappings in skims")

for m in omx_in.listMappings():
    print(f"found mapping '{m}' in skims")
assert len(omx_in.listMappings()) == 0


# assert omx_in.shape() == (len(taz), len(taz))

omx_in.close()
