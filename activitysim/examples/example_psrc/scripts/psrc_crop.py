import os
import pandas as pd
import openmatrix as omx
import numpy as np


input_dir = './data_raw'

MAZ_OFFSET = 0

seg = 'seattle'
check_geography = seg == 'full'

output_dir = f'./data_{seg}'
if seg == 'downtown':
    taz_min = 339  # instead of 400 because need university
    taz_max = 630  # downtown seattle tazs
elif seg == 'seattle':
    taz_min = 0
    taz_max = 857  # seattle tazs
elif seg == 'full':
    taz_min = 0
    taz_max = 100000
else:
    raise RuntimeError(f"Unknown seg: {seg}")


def input_path(file_name):
    return os.path.join(input_dir, file_name)


def output_path(file_name):
    return os.path.join(output_dir, file_name)


def integerize_id_columns(df, table_name):
    columns = ['MAZ', 'OMAZ', 'DMAZ', 'TAZ', 'zone_id', 'household_id', 'HHID']
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


print(f"output_dir {output_dir} taz_min {taz_min} taz_max {taz_max}")

if check_geography:

    # ######## check for orphan_households not in any maz in land_use
    land_use = read_csv("land_use.csv")
    land_use = land_use[['MAZ', 'TAZ']]  # King County
    land_use = land_use.sort_values(['TAZ', 'MAZ'])

    households = read_csv("households.csv")
    orphan_households = households[~households.MAZ.isin(land_use.MAZ)]
    print(f"{len(orphan_households)} orphan_households")
    to_csv(orphan_households, "orphan_households.csv")

    # ######## check that land_use and maz and taz tables have same MAZs and TAZs

    # could just build maz and taz files, but want to make sure PSRC data is right

    land_use = read_csv("land_use.csv")
    land_use = land_use.sort_values('MAZ')
    maz = read_csv("maz.csv").sort_values('MAZ')

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

    land_use = land_use.sort_values('TAZ')
    taz = read_csv("taz.csv").sort_values('TAZ')

    # ### FATAL ###
    if not land_use.TAZ.isin(taz.TAZ).all():
        print(f"land_use.TAZ not in taz.TAZ\n{land_use.TAZ[~land_use.TAZ.isin(taz.MAZ)]}")
        raise RuntimeError(f"land_use.TAZ not in taz.TAZ")

    if not taz.TAZ.isin(land_use.TAZ).all():
        print(f"taz.TAZ not in land_use.TAZ\n{taz.TAZ[~taz.TAZ.isin(land_use.TAZ)]}")

    # #########s

#
# land_use
#
land_use = read_csv("land_use.csv")
land_use = land_use[(land_use["TAZ"] >= taz_min) & (land_use["TAZ"] <= taz_max)]
integerize_id_columns(land_use, 'land_use')
land_use = land_use.sort_values('MAZ')

# move MAZ and TAZ columns to front
land_use = land_use[['MAZ', 'TAZ'] + [c for c in land_use.columns if c not in ['MAZ', 'TAZ']]]
to_csv(land_use, "land_use.csv")

#
# maz
#
maz = read_csv("maz.csv").sort_values(['MAZ', 'TAZ'])
maz = maz[maz["MAZ"].isin(land_use.MAZ)]
integerize_id_columns(maz, 'maz')

assert (land_use.MAZ.isin(maz.MAZ).all())
assert (land_use.TAZ.isin(maz.TAZ).all())
assert (maz.TAZ.isin(land_use.TAZ).all())
to_csv(maz, "maz.csv")

#
# taz
#
taz = read_csv("taz.csv").sort_values(['TAZ'])
taz = taz[taz["TAZ"].isin(land_use.TAZ)]
integerize_id_columns(taz, 'taz')

assert (land_use.TAZ.isin(taz.TAZ).all())
to_csv(taz, "taz.csv")

# print(maz.shape)
# print(f"MAZ {len(maz.MAZ.unique())}")
# print(f"TAZ {len(maz.TAZ.unique())}")

#
# households
#
households = read_csv("households.csv")
households = households[households["MAZ"].isin(maz.MAZ)]
integerize_id_columns(households, 'households')

to_csv(households, "households.csv")

#
# persons
#
persons = read_csv("persons.csv")
persons = persons[persons["household_id"].isin(households.HHID)]
integerize_id_columns(persons, 'persons')

to_csv(persons, "persons.csv")

#
# maz_to_maz_walk and maz_to_maz_bike
#
for file_name in ["maz_to_maz_walk.csv", "maz_to_maz_bike.csv"]:
    m2m = read_csv(file_name)
    m2m = m2m[m2m.OMAZ.isin(maz.MAZ) & m2m.DMAZ.isin(maz.MAZ)]
    integerize_id_columns(m2m, file_name)
    to_csv(m2m, file_name)

#
# skims
#
omx_file_name = 'skims.omx'
omx_in = omx.open_file(input_path(omx_file_name))
print(f"omx_in shape {omx_in.shape()}")

assert not omx_in.listMappings()
taz = taz.sort_values('TAZ')
taz.index = taz.TAZ - 1
tazs_indexes = taz.index.tolist()  # index of TAZ in skim (zero-based, no mapping)
taz_labels = taz.TAZ.tolist()  # TAZ zone_ids in omx index order


omx_out = omx.open_file(output_path(omx_file_name), 'w')

# write mapping
omx_out.create_mapping('ZONE', taz_labels)

for mat_name in omx_in.list_matrices():

    # make sure we have a vanilla numpy array, not a CArray
    m = np.asanyarray(omx_in[mat_name])

    m = m[tazs_indexes, :][:, tazs_indexes]

    print(f"{mat_name} {m.shape}")

    omx_out[mat_name] = m


omx_in.close()
omx_out.close()
