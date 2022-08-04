# remove some of the asim-style columns added by marin_work_tour_mode_choice.py
# so data input files look 'realistic' - and that work is done instaed by 'import_tours' annotation expression files

import os

import openmatrix as omx
import pandas as pd

input_dir = "./data_3_marin"
output_dir = (
    "./data_3_marin/fix"  # don't overwrite - but these files shold replace 'oritinals'
)


def input_path(filenane):
    return os.path.join(input_dir, filenane)


def output_path(filenane):
    return os.path.join(output_dir, filenane)


# 0 - get county zones

mazs = pd.read_csv(input_path("maz_data_asim.csv"))
del mazs["zone_id"]
del mazs["county_id"]
mazs.to_csv(output_path("maz_data_asim.csv"), index=False)

tazs = mazs["TAZ"].unique()
tazs.sort()
assert ((tazs - 1) == range(len(tazs))).all()

# MAZ,TAZ
taps = pd.read_csv(input_path("maz_taz.csv"))
# nothing
taps.to_csv(output_path("maz_taz.csv"), index=False)

taps = pd.read_csv(input_path("tap_data.csv"))
# nothing
taps.to_csv(output_path("tap_data.csv"), index=False)


# 2 - nearby skims need headers

maz_tap_walk = pd.read_csv(input_path("maz_tap_walk.csv"))
maz_maz_walk = pd.read_csv(input_path("maz_maz_walk.csv"))
maz_maz_bike = pd.read_csv(input_path("maz_maz_bike.csv"))

del maz_tap_walk["TAP.1"]
del maz_maz_walk["DMAZ.1"]
del maz_maz_bike["DMAZ.1"]

maz_tap_walk.to_csv(output_path("maz_tap_walk.csv"), index=False)
maz_maz_walk.to_csv(output_path("maz_maz_walk.csv"), index=False)
maz_maz_bike.to_csv(output_path("maz_maz_bike.csv"), index=False)

# 3 - accessibility data

access = pd.read_csv(input_path("access.csv"))
del access["zone_id"]
access.to_csv(output_path("access.csv"), index=False)

# 4 - maz to tap drive data

taz_tap_drive = pd.read_csv(input_path("maz_taz_tap_drive.csv"))

taz_tap_drive.to_csv(output_path("maz_taz_tap_drive.csv"), index=False)

# 5 - households

households = pd.read_csv(input_path("households_asim.csv"))
del households["home_zone_id"]
del households["household_id"]

households.to_csv(output_path("households_asim.csv"), index=False)

# 6 - persons

persons = pd.read_csv(input_path("persons_asim.csv"))
del persons["person_id"]
del persons["household_id"]
del persons["is_university"]
persons.to_csv(output_path("persons_asim.csv"), index=False)

# 7 - tours file

work_tours = pd.read_csv(input_path("work_tours.csv"))
del work_tours["household_id"]
del work_tours["destination"]
del work_tours["start"]
del work_tours["end"]
del work_tours["tour_type"]
work_tours.to_csv(output_path("work_tours.csv"), index=False)
