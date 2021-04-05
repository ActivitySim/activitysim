# crop marin tvpb example data processing to one county
# Ben Stabler, ben.stabler@rsginc.com, 09/17/20

import os
import pandas as pd
import openmatrix as omx
import argparse
import numpy as np

MAZ_OFFSET = 100000

segments = {
    'test': {'DistName': ["Downtown SF"]},
    'marin_sf': {'CountyName': ["Marin", "San Francisco"]},
    'full': {},
}

parser = argparse.ArgumentParser(description='crop Marin raw_data')
parser.add_argument('segment_name', metavar='segment_name', type=str, nargs=1,
                    help=f"geography segmentation (e.g. full)")

parser.add_argument('-c', '--check_geography',
                    default=False,
                    action='store_true',
                    help='check consistency of MAZ, TAZ, TAP zone_ids and foreign keys & write orphan_households file')

args = parser.parse_args()


segment_name = args.segment_name[0]
check_geography = args.check_geography

assert segment_name in segments.keys(), f"Unknown seg: {segment_name}"

input_dir = './data_raw'
output_dir = f'./data_{segment_name}'


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
        if c in ['MAZ', 'OMAZ', 'DMAZ', 'mgra', 'orig_mgra', 'dest_mgra']:
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


# non-standard input file names
LAND_USE = "maz_data_asim.csv"
HOUSEHOLDS = "households_asim.csv"
PERSONS = "persons_asim.csv"
MAZ_TAZ = "maz_taz.csv"
TAP_MAZ = "tap_data.csv"
ACCESSIBILITY = "access.csv"
WORK_TOURS = "work_tours.csv"

if check_geography:

    # ######## check for orphan_households not in any maz in land_use
    land_use = read_csv(LAND_USE)
    land_use = land_use[['MAZ', 'TAZ']]
    land_use = land_use.sort_values(['TAZ', 'MAZ'])

    households = read_csv(HOUSEHOLDS)
    orphan_households = households[~households.MAZ.isin(land_use.MAZ)]
    print(f"{len(orphan_households)} orphan_households")

    # write orphan_households to INPUT directory (since it doesn't belong in output)
    file_name = "orphan_households.csv"
    print(f"writing {file_name} {orphan_households.shape} to {input_path(file_name)}")
    orphan_households.to_csv(input_path(file_name), index=False)

    # ######## check that land_use and maz and taz tables have same MAZs and TAZs

    # could just build maz and taz files, but want to make sure PSRC data is right

    land_use = read_csv(LAND_USE)
    land_use = land_use.sort_values('MAZ')
    maz = read_csv(MAZ_TAZ).sort_values('MAZ')

    # ### FATAL ###
    if not land_use.MAZ.isin(maz.MAZ).all():
        print(f"land_use.MAZ not in maz.MAZ\n{land_use.MAZ[~land_use.MAZ.isin(maz.MAZ)]}")
        raise RuntimeError(f"land_use.MAZ not in maz.MAZ")

    if not maz.MAZ.isin(land_use.MAZ).all():
        print(f"maz.MAZ not in land_use.MAZ\n{maz.MAZ[~maz.MAZ.isin(land_use.MAZ)]}")

    # ### FATAL ###
    if not land_use.TAZ.isin(maz.TAZ).all():
        print(f"land_use.TAZ not in maz.TAZ\n{land_use.TAZ[~land_use.TAZ.isin(maz.TAZ)]}")
        raise RuntimeError(f"land_use.TAZ not in maz.TAZ")

    if not maz.TAZ.isin(land_use.TAZ).all():
        print(f"maz.TAZ not in land_use.TAZ\n{maz.TAZ[~maz.TAZ.isin(land_use.TAZ)]}")


# land_use

land_use = read_csv(LAND_USE)

slicer = segments[segment_name]
for slice_col, slice_values in slicer.items():
    print(f"slice {slice_col}: {slice_values}")

    land_use = land_use[land_use[slice_col].isin(slice_values)]

print(f"land_use shape after slicing {land_use.shape}")
to_csv(land_use, 'land_use.csv')

# maz_taz, tazs, taps

maz_taz = land_use[['MAZ', 'TAZ']]
to_csv(maz_taz, "maz_taz.csv")

tazs = land_use["TAZ"].unique()
tazs.sort()

taps = read_csv(TAP_MAZ)
taps = taps[['TAP', 'TAZ']].sort_values(by='TAP')
taps = taps[taps["TAZ"].isin(tazs)]
to_csv(taps, "tap.csv")

# maz to tap walk, bike

maz_tap_walk = read_csv("maz_tap_walk.csv")
maz_maz_walk = read_csv("maz_maz_walk.csv")
maz_maz_bike = read_csv("maz_maz_bike.csv")

maz_tap_walk = maz_tap_walk[maz_tap_walk["MAZ"].isin(land_use["MAZ"]) & maz_tap_walk["TAP"].isin(taps["TAP"])]
maz_maz_walk = maz_maz_walk[maz_maz_walk["OMAZ"].isin(land_use["MAZ"]) & maz_maz_walk["DMAZ"].isin(land_use["MAZ"])]
maz_maz_bike = maz_maz_bike[maz_maz_bike["OMAZ"].isin(land_use["MAZ"]) & maz_maz_bike["DMAZ"].isin(land_use["MAZ"])]

to_csv(maz_tap_walk, "maz_tap_walk.csv")
to_csv(maz_maz_walk, "maz_maz_walk.csv")
to_csv(maz_maz_bike, "maz_maz_bike.csv")

tap_lines = read_csv("tap_lines.csv")
tap_lines = tap_lines[tap_lines['TAP'].isin(taps["TAP"])]
to_csv(tap_lines, "tap_lines.csv")

# taz to tap drive data

taz_tap_drive = read_csv("maz_taz_tap_drive.csv")
taz_tap_drive = taz_tap_drive[taz_tap_drive["MAZ"].isin(land_use["MAZ"]) & taz_tap_drive["TAP"].isin(taps["TAP"])]
to_csv(taz_tap_drive, "maz_taz_tap_drive.csv")


# accessibility data

access = read_csv(ACCESSIBILITY)
access = access[access["mgra"].isin(land_use["MAZ"])]
to_csv(access, "accessibility.csv")


# households

households = read_csv(HOUSEHOLDS)
households = households[households["MAZ"].isin(land_use["MAZ"])]
to_csv(households, "households.csv")

# persons

persons = read_csv(PERSONS)
persons = persons[persons["HHID"].isin(households["HHID"])]
to_csv(persons, "persons.csv")

# tours file

work_tours = read_csv(WORK_TOURS)
work_tours = work_tours[work_tours["hh_id"].isin(households["HHID"])]
work_tours = work_tours[work_tours["orig_mgra"].isin(land_use["MAZ"]) & work_tours["dest_mgra"].isin(land_use["MAZ"])]
to_csv(work_tours, "work_tours.csv")

# skims

taz_indexes = (tazs - 1).tolist()  # offset_map
tap_indexes = (taps["TAP"] - 1).tolist()  # offset_map
time_periods = ["AM", "EA", "EV", "MD", "PM"]
skim_data_type = np.float32

# taz skims with skim_data_type np.float32 are under 2GB - otherwise we would need to further segment them

for tp in time_periods:
    in_file_name = f'HWYSKM{tp}_taz_rename.omx'
    taz_file_in = omx.open_file(input_path(in_file_name))
    out_file_name = f'highway_skims_{tp}.omx'
    taz_file_out = omx.open_file(output_path(out_file_name), 'w')
    taz_file_out.create_mapping('ZONE', tazs.tolist())
    for mat_name in taz_file_in.list_matrices():
        # make sure we have a vanilla numpy array, not a CArray
        m = np.asanyarray(taz_file_in[mat_name]).astype(skim_data_type)
        m = m[taz_indexes, :][:, taz_indexes]
        taz_file_out[mat_name] = m
        print(f"taz {mat_name} {m.shape}")
    taz_file_in.close()
    taz_file_out.close()

for skim_set in ["SET1", "SET2", "SET3"]:
    out_file_name = f'transit_skims_{skim_set}.omx'
    tap_file_out = omx.open_file(output_path(out_file_name), 'w')
    tap_file_out.create_mapping('TAP', taps["TAP"].tolist())
    for tp in time_periods:
        in_file_name = f'transit_skims_{tp}_{skim_set}_rename.omx'
        tap_file_in = omx.open_file(input_path(in_file_name))
        for mat_name in tap_file_in.list_matrices():
            # make sure we have a vanilla numpy array, not a CArray
            m = np.asanyarray(tap_file_in[mat_name]).astype(skim_data_type)
            m = m[tap_indexes, :][:, tap_indexes]
            tap_file_out[mat_name] = m
            print(f"tap {skim_set} {mat_name} {m.shape}")
        tap_file_in.close()
    tap_file_out.close()
