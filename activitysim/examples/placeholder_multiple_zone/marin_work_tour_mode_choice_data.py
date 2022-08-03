# marin tvpb example data processing
# Ben Stabler, ben.stabler@rsginc.com, 09/17/20

import openmatrix as omx
import pandas as pd

# command to run the underdevelopment example
# python simulation.py -c configs_3_zone_marin -d data_3_marin -o output_3_marin

# data processing at c:\projects\activitysim\marin

# 1 - fix skim names, put time periods at end and make all names unique

time_periods = ["AM", "EA", "EV", "MD", "PM"]
for tp in time_periods:
    taz_file = omx.open_file("HWYSKM" + tp + "_taz.omx")
    taz_file_rename = omx.open_file("HWYSKM" + tp + "_taz_rename.omx", "w")
    for mat_name in taz_file.list_matrices():
        taz_file_rename[mat_name + "__" + tp] = taz_file[mat_name][:]
        print(mat_name + "__" + tp)
    taz_file.close()
    taz_file_rename.close()

for tp in time_periods:
    for skim_set in ["SET1", "SET2", "SET3"]:
        tap_file = omx.open_file("transit_skims_" + tp + "_" + skim_set + ".omx")
        tap_file_rename = omx.open_file(
            "transit_skims_" + tp + "_" + skim_set + "_rename.omx", "w"
        )
        for mat_name in tap_file.list_matrices():
            tap_file_rename[mat_name + "_" + skim_set + "__" + tp] = tap_file[mat_name][
                :
            ]
            print(mat_name + "_" + skim_set + "__" + tp)
        tap_file.close()
        tap_file_rename.close()

# 2 - nearby skims need headers

maz_tap_walk = pd.read_csv(
    "2015_test_2019_02_13_Part3/skims/ped_distance_maz_tap.txt", header=None
)
maz_maz_walk = pd.read_csv(
    "2015_test_2019_02_13_Part3/skims/ped_distance_maz_maz.txt", header=None
)
maz_maz_bike = pd.read_csv(
    "2015_test_2019_02_13_Part3/skims/bike_distance_maz_maz.txt", header=None
)

maz_tap_walk.columns = [
    "MAZ",
    "TAP",
    "TAP",
    "WALK_TRANSIT_GEN_COST",
    "WALK_TRANSIT_DIST",
]
maz_maz_walk.columns = ["OMAZ", "DMAZ", "DMAZ", "WALK_GEN_COST", "WALK_DIST"]
maz_maz_bike.columns = ["OMAZ", "DMAZ", "DMAZ", "BIKE_GEN_COST", "BIKE_DIST"]

maz_tap_walk["WALK_TRANSIT_DIST"] = maz_tap_walk["WALK_TRANSIT_DIST"] / 5280  # miles
maz_maz_walk["WALK_DIST"] = maz_maz_walk["WALK_DIST"] / 5280  # miles
maz_maz_bike["BIKE_DIST"] = maz_maz_bike["BIKE_DIST"] / 5280  # miles

maz_tap_walk[["MAZ", "TAP", "WALK_TRANSIT_DIST"]].to_csv(
    "maz_tap_walk.csv", index=False
)
maz_maz_walk[["OMAZ", "DMAZ", "WALK_DIST"]].to_csv("maz_maz_walk.csv", index=False)
maz_maz_bike[["OMAZ", "DMAZ", "BIKE_DIST"]].to_csv("maz_maz_bike.csv", index=False)

# 3 - maz data

mazs = pd.read_csv("2015_test_2019_02_13_Part2/landuse/maz_data_withDensity.csv")
pcost = pd.read_csv("2015_test_2019_02_13/ctramp_output/mgraParkingCost.csv")

mazs = pd.concat([mazs, pcost], axis=1)
mazs = mazs.fillna(0)

tazs = pd.read_csv("2015_test_2019_02_13_Part2/landuse/taz_data.csv")
tazs = tazs.set_index("TAZ", drop=False)

mazs["TERMINALTIME"] = tazs["TERMINALTIME"].loc[mazs["TAZ"]].tolist()

mazs["zone_id"] = mazs["MAZ"]
mazs["county_id"] = mazs["CountyID"]
mazs = mazs.set_index("zone_id", drop=False)

mazs.to_csv("maz_data_asim.csv", index=False)

# 4 - accessibility data

access = pd.read_csv("2015_test_2019_02_13/ctramp_output/accessibilities.csv")
access = access.drop([0])
access["zone_id"] = access["mgra"]
access = access.set_index("zone_id", drop=False)
access.to_csv("access.csv", index=False)

# 5 - maz to tap drive data

taz_tap_drive = pd.read_csv("2015_test_2019_02_13_Part3/skims/drive_maz_taz_tap.csv")

taz_tap_drive = taz_tap_drive.pivot_table(
    index=["FTAZ", "TTAP"], values=["DTIME", "DDIST", "WDIST"], fill_value=0
)

taz_tap_drive.columns = list(map("".join, taz_tap_drive.columns))
taz_tap_drive = taz_tap_drive.reset_index()
taz_tap_drive = taz_tap_drive.set_index("FTAZ")
taz_tap_drive["TAP"] = taz_tap_drive["TTAP"]

taz_tap_drive = pd.merge(
    mazs[["MAZ", "TAZ"]], taz_tap_drive, left_on=["TAZ"], right_on=["FTAZ"]
)
taz_tap_drive[["MAZ", "TAP", "DDIST", "DTIME", "WDIST"]].to_csv(
    "maz_taz_tap_drive.csv", index=False
)

# 6 - tours file, we just need work tours

itour = pd.read_csv("2015_test_2019_02_13/ctramp_output/indivTourData_3.csv")
work_tours = itour[itour["tour_purpose"] == "Work"]

work_tours["tour_id"] = range(1, len(work_tours) + 1)
work_tours["household_id"] = work_tours["hh_id"]
work_tours = work_tours.set_index("tour_id", drop=False)

work_tours["destination"] = work_tours["dest_mgra"]

work_tours["start"] = work_tours["start_period"]
work_tours["end"] = work_tours["end_period"]
work_tours["tour_type"] = "work"

work_tours.to_csv("work_tours.csv", index=False)

# 7 - households

households = pd.read_csv("2015_test_2019_02_13_Part2/popsyn/households.csv")
households["household_id"] = households["HHID"]
households["home_zone_id"] = households["MAZ"]
households = households.set_index("household_id", drop=False)

households.to_csv("households_asim.csv", index=False)

# 8 - persons

persons = pd.read_csv("2015_test_2019_02_13_Part2/popsyn/persons.csv")
persons["person_id"] = persons["PERID"]
persons["household_id"] = persons["HHID"]
persons = persons.set_index("person_id", drop=False)

persons_output = pd.read_csv("2015_test_2019_02_13/ctramp_output/personData_3.csv")
persons_output = persons_output.set_index("person_id", drop=False)
persons["type"] = persons_output["type"].loc[persons.index]
persons["value_of_time"] = persons_output["value_of_time"].loc[persons.index]
persons["is_university"] = persons["type"] == "University student"
persons["fp_choice"] = persons_output["fp_choice"]

persons.to_csv("persons_asim.csv", index=False)

# 9 - replace existing pipeline tables for restart for now

# run simple three zone example and get output pipeline and then replace tables before tour mode choice
pipeline = pd.io.pytables.HDFStore("pipeline.h5")
pipeline.keys()

pipeline["/accessibility/compute_accessibility"] = access  # index zone_id
pipeline["/households/joint_tour_frequency"] = households  # index household_id
pipeline["/persons/non_mandatory_tour_frequency"] = persons  # index person_id
pipeline["/land_use/initialize_landuse"] = mazs  # index zone_id
pipeline["/tours/non_mandatory_tour_scheduling"] = work_tours  # index tour_id

pipeline.close()
