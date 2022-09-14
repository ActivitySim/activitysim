#############################################################
# ActivitySim verification against TM1
# Ben Stabler, ben.stabler@rsginc.com, 02/22/19
# C:\projects\activitysim\verification>python compare_results.py
#############################################################

import openmatrix as omx
import pandas as pd

#############################################################
# INPUTS
#############################################################

pipeline_filename = "asim/pipeline.h5"
distance_matrix_filename = "asim/skims.omx"
asim_nmtf_alts_filename = "asim/non_mandatory_tour_frequency_alternatives.csv"

process_sp = True  # False skip work/sch shadow pricing comparisons, True do them
process_tm1 = True  # False only processes asim, True processes tm1 as well

asim_sp_work_filename = "asim/shadow_price_workplace_modeled_size_10.csv"
asim_sp_school_filename = "asim/shadow_price_school_modeled_size_10.csv"
asim_sp_school_no_sp_filename = "asim/shadow_price_school_modeled_size_1.csv"

tm1_access_filename = "tm1/accessibility.csv"
tm1_sp_filename = "tm1/ShadowPricing_9.csv"
tm1_work_filename = "tm1/wsLocResults_1.csv"
tm1_ao_filename = "tm1/aoResults.csv"
tm1_hh_filename = "tm1/householdData_1.csv"
tm1_cdap_filename = "tm1/cdapResults.csv"
tm1_per_filename = "tm1/personData_1.csv"
tm1_tour_filename = "tm1/indivTourData_1.csv"
tm1_jtour_filename = "tm1/jointTourData_1.csv"
tm1_trips_filename = "tm1/indivTripData_1.csv"
tm1_jtrips_filename = "tm1/jointTripData_1.csv"

#############################################################
# OUTPUT FILES FOR DEBUGGING
#############################################################

asim_zones_filename = "asim/asim_zones.csv"
asim_access_filename = "asim/asim_access.csv"
asim_per_filename = "asim/asim_per.csv"
asim_hh_filename = "asim/asim_hh.csv"
asim_tour_filename = "asim/asim_tours.csv"
asim_trips_filename = "asim/asim_trips.csv"

#############################################################
# COMMON LABELS
#############################################################

ptypes = [
    "",
    "Full-time worker",
    "Part-time worker",
    "University student",
    "Non-worker",
    "Retired",
    "Student of driving age",
    "Student of non-driving age",
    "Child too young for school",
]

mode_labels = [
    "",
    "DRIVEALONEFREE",
    "DRIVEALONEPAY",
    "SHARED2FREE",
    "SHARED2PAY",
    "SHARED3FREE",
    "SHARED3PAY",
    "WALK",
    "BIKE",
    "WALK_LOC",
    "WALK_LRF",
    "WALK_EXP",
    "WALK_HVY",
    "WALK_COM",
    "DRIVE_LOC",
    "DRIVE_LRF",
    "DRIVE_EXP",
    "DRIVE_HVY",
    "DRIVE_COM",
]

#############################################################
# DISTANCE SKIM
#############################################################

# read distance matrix (DIST)
distmat = omx.open_file(distance_matrix_filename)["DIST"][:]

#############################################################
# EXPORT TABLES
#############################################################

# write tables for verification
tazs = pd.read_hdf(pipeline_filename, "land_use/initialize_landuse")
tazs["zone"] = tazs.index
tazs.to_csv(asim_zones_filename, index=False)

access = pd.read_hdf(pipeline_filename, "accessibility/compute_accessibility")
access.to_csv(asim_access_filename, index=False)

hh = pd.read_hdf(pipeline_filename, "households/joint_tour_frequency")
hh["household_id"] = hh.index
hh.to_csv(asim_hh_filename, index=False)

per = pd.read_hdf(pipeline_filename, "persons/non_mandatory_tour_frequency")
per["person_id"] = per.index
per.to_csv(asim_per_filename, index=False)

tours = pd.read_hdf(pipeline_filename, "tours/stop_frequency")
tours["tour_id"] = tours.index
tours.to_csv(asim_tour_filename, index=False)

trips = pd.read_hdf(pipeline_filename, "trips/trip_mode_choice")
trips["trip_id"] = trips.index
trips.to_csv(asim_trips_filename, index=False)

#############################################################
# AGGREGATE
#############################################################

# accessibilities

if process_tm1:
    tm1_access = pd.read_csv(tm1_access_filename)
    tm1_access.to_csv("outputs/tm1_access.csv", na_rep=0)

asim_access = pd.read_csv(asim_access_filename)
asim_access.to_csv("outputs/asim_access.csv", na_rep=0)

#############################################################
# HOUSEHOLD AND PERSON
#############################################################

# work and school location

if process_sp:

    if process_tm1:
        tm1_markets = [
            "work_low",
            "work_med",
            "work_high",
            "work_high",
            "work_very high",
            "university",
            "school_high",
            "school_grade",
        ]
        tm1 = pd.read_csv(tm1_sp_filename)
        tm1 = tm1.groupby(tm1["zone"]).sum()
        tm1["zone"] = tm1.index
        tm1 = tm1.loc[tm1["zone"] > 0]
        ws_size = tm1[["zone"]]
        for i in range(len(tm1_markets)):
            ws_size[tm1_markets[i] + "_modeledDests"] = tm1[
                tm1_markets[i] + "_modeledDests"
            ]
        ws_size.to_csv("outputs/tm1_work_school_location.csv", na_rep=0)

    asim_markets = [
        "work_low",
        "work_med",
        "work_high",
        "work_high",
        "work_veryhigh",
        "university",
        "highschool",
        "gradeschool",
    ]
    asim = pd.read_csv(asim_sp_work_filename)
    asim_sch = pd.read_csv(asim_sp_school_filename)
    asim_sch_no_sp = pd.read_csv(asim_sp_school_no_sp_filename)
    asim_sch["gradeschool"] = asim_sch_no_sp[
        "gradeschool"
    ]  # grade school not shadow priced
    asim = asim.set_index("TAZ", drop=False)
    asim_sch = asim_sch.set_index("TAZ", drop=False)

    asim["gradeschool"] = asim_sch["gradeschool"].loc[asim["TAZ"]].tolist()
    asim["highschool"] = asim_sch["highschool"].loc[asim["TAZ"]].tolist()
    asim["university"] = asim_sch["university"].loc[asim["TAZ"]].tolist()

    ws_size = asim[["TAZ"]]
    for i in range(len(asim_markets)):
        ws_size[asim_markets[i] + "_asim"] = asim[asim_markets[i]]
    ws_size.to_csv("outputs/asim_work_school_location.csv", na_rep=0)

    # work county to county flows
    tazs = pd.read_csv(asim_zones_filename)
    counties = ["", "SF", "SM", "SC", "ALA", "CC", "SOL", "NAP", "SON", "MAR"]
    tazs["COUNTYNAME"] = pd.Series(counties)[tazs["county_id"].tolist()].tolist()
    tazs = tazs.set_index("zone", drop=False)

    if process_tm1:
        tm1_work = pd.read_csv(tm1_work_filename)
        tm1_work["HomeCounty"] = tazs["COUNTYNAME"].loc[tm1_work["HomeTAZ"]].tolist()
        tm1_work["WorkCounty"] = (
            tazs["COUNTYNAME"].loc[tm1_work["WorkLocation"]].tolist()
        )
        tm1_work_counties = tm1_work.groupby(["HomeCounty", "WorkCounty"]).count()[
            "HHID"
        ]
        tm1_work_counties = tm1_work_counties.reset_index()
        tm1_work_counties = tm1_work_counties.pivot(
            index="HomeCounty", columns="WorkCounty"
        )
        tm1_work_counties.to_csv("outputs/tm1_work_counties.csv", na_rep=0)

    asim_cdap = pd.read_csv(asim_per_filename)
    asim_cdap["HomeCounty"] = tazs["COUNTYNAME"].loc[asim_cdap["home_taz"]].tolist()
    asim_cdap["WorkCounty"] = (
        tazs["COUNTYNAME"].loc[asim_cdap["workplace_zone_id"]].tolist()
    )
    asim_work_counties = asim_cdap.groupby(["HomeCounty", "WorkCounty"]).count()[
        "household_id"
    ]
    asim_work_counties = asim_work_counties.reset_index()
    asim_work_counties = asim_work_counties.pivot(
        index="HomeCounty", columns="WorkCounty"
    )
    asim_work_counties.to_csv("outputs/asim_work_counties.csv", na_rep=0)

# auto ownership - count of hhs by num autos by taz

if process_tm1:
    tm1_ao = pd.read_csv(tm1_ao_filename)
    tm1_hh = pd.read_csv(tm1_hh_filename)
    tm1_ao = tm1_ao.set_index("HHID", drop=False)
    tm1_hh["ao"] = tm1_ao["AO"].loc[tm1_hh["hh_id"]].tolist()
    tm1_autos = tm1_hh.groupby(["taz", "ao"]).count()["hh_id"]
    tm1_autos = tm1_autos.reset_index()
    tm1_autos = tm1_autos.pivot(index="taz", columns="ao")
    tm1_autos.to_csv("outputs/tm1_autos.csv", na_rep=0)

asim_ao = pd.read_csv(asim_hh_filename)
asim_autos = asim_ao.groupby(["TAZ", "auto_ownership"]).count()["SERIALNO"]
asim_autos = asim_autos.reset_index()
asim_autos = asim_autos.pivot(index="TAZ", columns="auto_ownership")
asim_autos.to_csv("outputs/asim_autos.csv", na_rep=0)

# cdap - ptype count and ptype by M,N,H

if process_tm1:
    tm1_cdap = pd.read_csv(tm1_cdap_filename)
    tm1_cdap_sum = tm1_cdap.groupby(["PersonType", "ActivityString"]).count()["HHID"]
    tm1_cdap_sum = tm1_cdap_sum.reset_index()
    tm1_cdap_sum = tm1_cdap_sum.pivot(index="PersonType", columns="ActivityString")
    tm1_cdap_sum.to_csv("outputs/tm1_cdap.csv", na_rep=0)

asim_cdap = pd.read_csv(asim_per_filename)
asim_cdap_sum = asim_cdap.groupby(["ptype", "cdap_activity"]).count()["household_id"]
asim_cdap_sum = asim_cdap_sum.reset_index()
asim_cdap_sum = asim_cdap_sum.pivot(index="ptype", columns="cdap_activity")
asim_cdap_sum.to_csv("outputs/asim_cdap.csv", na_rep=0)

# free parking by ptype

if process_tm1:
    tm1_per = pd.read_csv(tm1_per_filename)
    tm1_per["fp_choice"] = tm1_per["fp_choice"] == 1  # 1=free, 2==pay
    tm1_work = pd.read_csv(tm1_work_filename)
    tm1_work = tm1_work.set_index("PersonID", drop=False)
    tm1_per["WorkLocation"] = (
        tm1_work["WorkLocation"].loc[tm1_per["person_id"]].tolist()
    )
    tm1_fp = tm1_per[tm1_per["WorkLocation"] > 0]
    tm1_fp = tm1_fp.groupby(["type", "fp_choice"]).count()["hh_id"]
    tm1_fp = tm1_fp.reset_index()
    tm1_fp = tm1_fp.pivot(index="type", columns="fp_choice")
    tm1_fp.to_csv("outputs/tm1_fp.csv", na_rep=0)

asim_cdap["ptypename"] = pd.Series(ptypes)[asim_cdap["ptype"].tolist()].tolist()
asim_fp = asim_cdap.groupby(["ptypename", "free_parking_at_work"]).count()[
    "household_id"
]
asim_fp = asim_fp.reset_index()
asim_fp = asim_fp.pivot(index="ptypename", columns="free_parking_at_work")
asim_fp.to_csv("outputs/asim_fp.csv", na_rep=0)

# value of time

if process_tm1:
    tm1_per = pd.read_csv(tm1_per_filename)
    tm1_per["vot_bin"] = pd.cut(tm1_per["value_of_time"], range(51))
    tm1_per.groupby(["vot_bin"]).count()["hh_id"].to_csv(
        "outputs/tm1_vot.csv", na_rep=0
    )

asim_per = pd.read_csv(asim_per_filename)
asim_per["vot_bin"] = pd.cut(asim_per["value_of_time"], range(51))
asim_per.groupby(["vot_bin"]).count()["household_id"].to_csv(
    "outputs/asim_vot.csv", na_rep=0
)

#############################################################
# TOUR
#############################################################

# indiv mandatory tour freq

tm1_imf_codes = ["", "0", "work1", "work2", "school1", "school2", "work_and_school"]

if process_tm1:
    tm1_per = pd.read_csv(tm1_per_filename)
    tm1_hh = pd.read_csv(tm1_hh_filename)
    tm1_hh = tm1_hh.set_index("hh_id", drop=False)
    tm1_per["hhsize"] = tm1_hh["size"].loc[tm1_per["hh_id"]].tolist()
    # indexing starts at 1
    tm1_per["imf_choice_name"] = pd.Series(tm1_imf_codes)[
        (tm1_per["imf_choice"] + 1).tolist()
    ].tolist()
    tm1_imf = tm1_per.groupby(["type", "imf_choice_name"]).count()["hh_id"]
    tm1_imf = tm1_imf.reset_index()
    tm1_imf = tm1_imf.pivot(index="type", columns="imf_choice_name")

    tm1_imf.to_csv("outputs/tm1_imtf.csv", na_rep=0)

asim_ao = asim_ao.set_index("household_id", drop=False)
asim_cdap["hhsize"] = asim_ao["hhsize"].loc[asim_cdap["household_id"]].tolist()
asim_cdap["ptypename"] = pd.Series(ptypes)[asim_cdap["ptype"].tolist()].tolist()
asim_imf = pd.read_csv(asim_per_filename)
asim_imf["ptypename"] = pd.Series(ptypes)[asim_imf["ptype"].tolist()].tolist()
asim_imf["mandatory_tour_frequency"] = pd.Categorical(
    asim_imf["mandatory_tour_frequency"], categories=tm1_imf_codes
)
asim_imf["mandatory_tour_frequency"][
    asim_imf["mandatory_tour_frequency"].isnull()
] = "0"
asim_imf = asim_imf.groupby(["ptypename", "mandatory_tour_frequency"]).count()[
    "household_id"
]
asim_imf = asim_imf.reset_index()
asim_imf = asim_imf.pivot(index="ptypename", columns="mandatory_tour_frequency")
asim_imf.to_csv("outputs/asim_imtf.csv", na_rep=0)

# indiv mand tour departure and duration

if process_tm1:
    tm1_tours = pd.read_csv(tm1_tour_filename)
    tm1_tours = tm1_tours[tm1_tours["tour_category"] == "MANDATORY"]
    tm1_tours["tour_purpose"][tm1_tours["tour_purpose"].str.contains("work")] = "work"
    tm1_tours["tour_purpose"][tm1_tours["tour_purpose"].str.contains("s")] = "school"
    tm1_mtdd = tm1_tours.groupby(["start_hour", "end_hour", "tour_purpose"]).count()[
        "hh_id"
    ]
    tm1_mtdd = tm1_mtdd.reset_index()

    tm1_mtdd_sch = tm1_mtdd[tm1_mtdd["tour_purpose"] == "school"][
        ["start_hour", "end_hour", "hh_id"]
    ].pivot(index="start_hour", columns="end_hour")
    tm1_mtdd_work = tm1_mtdd[tm1_mtdd["tour_purpose"] == "work"][
        ["start_hour", "end_hour", "hh_id"]
    ].pivot(index="start_hour", columns="end_hour")
    tm1_mtdd_sch.to_csv("outputs/tm1_mtdd_school.csv", na_rep=0)
    tm1_mtdd_work.to_csv("outputs/tm1_mtdd_work.csv", na_rep=0)

asim_tours = pd.read_csv(asim_tour_filename)
asim_tours_man = asim_tours[asim_tours["tour_category"] == "mandatory"]
asim_mtdd = asim_tours_man.groupby(["start", "end", "tour_type"]).count()[
    "household_id"
]
asim_mtdd = asim_mtdd.reset_index()

asim_mtdd_sch = asim_mtdd[asim_mtdd["tour_type"] == "school"][
    ["start", "end", "household_id"]
].pivot(index="start", columns="end")
asim_mtdd_work = asim_mtdd[asim_mtdd["tour_type"] == "work"][
    ["start", "end", "household_id"]
].pivot(index="start", columns="end")

asim_mtdd_sch.to_csv("outputs/asim_mtdd_school.csv", na_rep=0)
asim_mtdd_work.to_csv("outputs/asim_mtdd_work.csv", na_rep=0)

# joint tour frequency

jtf_labels = [
    "",
    "0_tours",
    "1_Shop",
    "1_Main",
    "1_Eat",
    "1_Visit",
    "1_Disc",
    "2_SS",
    "2_SM",
    "2_SE",
    "2_SV",
    "2_SD",
    "2_MM",
    "2_ME",
    "2_MV",
    "2_MD",
    "2_EE",
    "2_EV",
    "2_ED",
    "2_VV",
    "2_VD",
    "2_DD",
]

if process_tm1:
    tm1_jtf = tm1_hh
    tm1_jtf = tm1_jtf[tm1_jtf["jtf_choice"] > 0]
    tm1_jtf["jtf_choice_label"] = pd.Series(jtf_labels)[
        tm1_jtf["jtf_choice"].tolist()
    ].tolist()
    tm1_jtf.groupby("jtf_choice_label").count()["hh_id"].to_csv(
        "outputs/tm1_jtf.csv", na_rep=0
    )

asim_jtf = pd.read_csv(asim_hh_filename)
asim_jtf = asim_jtf[asim_jtf["joint_tour_frequency"] != ""]
asim_jtf.groupby("joint_tour_frequency").count()["household_id"].to_csv(
    "outputs/asim_jtf.csv", na_rep=0
)

# joint tour comp

if process_tm1:
    tm1_jtours = pd.read_csv(tm1_jtour_filename)
    comp_labels = ["", "adult", "children", "mixed"]
    tm1_jtours["tour_composition_labels"] = pd.Series(comp_labels)[
        tm1_jtours["tour_composition"].tolist()
    ].tolist()
    tm1_jtour_comp = tm1_jtours.groupby(
        ["tour_purpose", "tour_composition_labels"]
    ).count()["hh_id"]
    tm1_jtour_comp = tm1_jtour_comp.reset_index()
    tm1_jtour_comp = tm1_jtour_comp.pivot(
        index="tour_purpose", columns="tour_composition_labels"
    )
    tm1_jtour_comp.to_csv("outputs/tm1_jtour_comp.csv", na_rep=0)

asim_jtours = pd.read_csv(asim_tour_filename)
asim_jtours = asim_jtours[asim_jtours["tour_category"] == "joint"]
asim_jtour_comp = asim_jtours.groupby(["tour_type", "composition"]).count()[
    "household_id"
]
asim_jtour_comp = asim_jtour_comp.reset_index()
asim_jtour_comp = asim_jtour_comp.pivot(index="tour_type", columns="composition")
asim_jtour_comp.to_csv("outputs/asim_jtour_comp.csv", na_rep=0)

# joint tour destination

if process_tm1:
    tm1_jtours["distance"] = distmat[
        tm1_jtours["orig_taz"] - 1, tm1_jtours["dest_taz"] - 1
    ]
    tm1_jtours["dist_bin"] = pd.cut(tm1_jtours["distance"], range(51))
    tm1_jtours.groupby(["dist_bin"]).count()["hh_id"].to_csv(
        "outputs/tm1_jtour_dist.csv", na_rep=0
    )

asim_jtours["distance"] = distmat[
    asim_jtours["origin"].astype(int) - 1, asim_jtours["destination"].astype(int) - 1
]
asim_jtours["dist_bin"] = pd.cut(asim_jtours["distance"], range(51))
asim_jtours.groupby(["dist_bin"]).count()["household_id"].to_csv(
    "outputs/asim_jtour_dist.csv", na_rep=0
)

# joint tour tdd

if process_tm1:
    tm1_jtours_tdd = tm1_jtours.groupby(["start_hour", "end_hour"]).count()["hh_id"]
    tm1_jtours_tdd = tm1_jtours_tdd.reset_index()
    tm1_jtours_tdd = tm1_jtours_tdd.pivot(index="start_hour", columns="end_hour")
    tm1_jtours_tdd.to_csv("outputs/tm1_jtours_tdd.csv", na_rep=0)

asim_jtours_tdd = asim_jtours.groupby(["start", "end"]).count()["household_id"]
asim_jtours_tdd = asim_jtours_tdd.reset_index()
asim_jtours_tdd = asim_jtours_tdd.pivot(index="start", columns="end")
asim_jtours_tdd.to_csv("outputs/asim_jtours_tdd.csv", na_rep=0)

# non-mand tour freq

alts = pd.read_csv(asim_nmtf_alts_filename)
alts["ID"] = range(len(alts))

if process_tm1:
    tm1_per = pd.read_csv(tm1_per_filename)
    # 0 doesn't participate in choice model therefore 0 tours, and -1 to align with asim
    tm1_per["inmf_choice"][tm1_per["inmf_choice"] == 0] = 1
    tm1_per["inmf_choice"] = tm1_per["inmf_choice"] - 1
    tm1_nmtf_sum = tm1_per.groupby(["inmf_choice"]).count()["hh_id"]
    tm1_alts = pd.concat([alts, tm1_nmtf_sum], axis=1)
    tm1_alts.to_csv("outputs/tm1_nmtf.csv", na_rep=0)

asim_per_nmtf = pd.read_csv(asim_per_filename)
asim_per_nmtf["ptypename"] = pd.Series(ptypes)[asim_per_nmtf["ptype"].tolist()].tolist()
asim_nmtf_sum = asim_per_nmtf.groupby(["non_mandatory_tour_frequency"]).count()[
    "household_id"
]
asim_alts = pd.concat([alts, asim_nmtf_sum], axis=1)
asim_alts.to_csv("outputs/asim_nmtf.csv", na_rep=0)

# non_mandatory_tour_destination

if process_tm1:
    tm1_tours = pd.read_csv(tm1_tour_filename)
    tm1_tours["distance"] = distmat[
        tm1_tours["orig_taz"] - 1, tm1_tours["dest_taz"] - 1
    ]
    tm1_tours["dist_bin"] = pd.cut(tm1_tours["distance"], range(51))
    tm1_tours_nm = tm1_tours[tm1_tours["tour_category"] == "INDIVIDUAL_NON_MANDATORY"]
    tm1_tours_nm.groupby(["dist_bin"]).count()["hh_id"].to_csv(
        "outputs/tm1_nmtd_dist.csv", na_rep=0
    )

asim_nm_tours = pd.read_csv(asim_tour_filename)
asim_nm_tours = asim_nm_tours[asim_nm_tours["tour_category"] == "non_mandatory"]
asim_nm_tours["distance"] = distmat[
    asim_nm_tours["origin"].astype(int) - 1,
    asim_nm_tours["destination"].astype(int) - 1,
]
asim_nm_tours["dist_bin"] = pd.cut(asim_nm_tours["distance"], range(51))
asim_nm_tours.groupby(["dist_bin"]).count()["household_id"].to_csv(
    "outputs/asim_nmtd_dist.csv", na_rep=0
)

# non_mandatory_tour_scheduling

if process_tm1:
    tm1_nmtours_tdd = tm1_tours_nm.groupby(["start_hour", "end_hour"]).count()["hh_id"]
    tm1_nmtours_tdd = tm1_nmtours_tdd.reset_index()
    tm1_nmtours_tdd = tm1_nmtours_tdd.pivot(index="start_hour", columns="end_hour")
    tm1_nmtours_tdd.to_csv("outputs/tm1_nmtours_tdd.csv", na_rep=0)

asim_nmtours_tdd = asim_nm_tours.groupby(["start", "end"]).count()["household_id"]
asim_nmtours_tdd = asim_nmtours_tdd.reset_index()
asim_nmtours_tdd = asim_nmtours_tdd.pivot(index="start", columns="end")
asim_nmtours_tdd.to_csv("outputs/asim_nmtours_tdd.csv", na_rep=0)

# tour mode choice

if process_tm1:
    tm1_tours = pd.read_csv(tm1_tour_filename)
    tm1_jtours = pd.read_csv(tm1_jtour_filename)
    tm1_tours["tour_mode_labels"] = pd.Series(mode_labels)[
        tm1_tours["tour_mode"].tolist()
    ].tolist()
    tm1_tours["tour_mode_labels"] = pd.Categorical(
        tm1_tours["tour_mode_labels"], categories=mode_labels
    )
    tm1_jtours["tour_mode_labels"] = pd.Series(mode_labels)[
        tm1_jtours["tour_mode"].tolist()
    ].tolist()
    tm1_jtours["tour_mode_labels"] = pd.Categorical(
        tm1_jtours["tour_mode_labels"], categories=mode_labels
    )
    tm1_nmn_tour_mode = tm1_tours.groupby(
        ["tour_mode_labels", "tour_category"]
    ).count()["hh_id"]
    tm1_nmn_tour_mode = tm1_nmn_tour_mode.reset_index()
    tm1_nmn_tour_mode = tm1_nmn_tour_mode.pivot(
        index="tour_mode_labels", columns="tour_category"
    )

    tm1_jtour_mode = tm1_jtours.groupby(["tour_mode_labels", "tour_category"]).count()[
        "hh_id"
    ]
    tm1_jtour_mode = tm1_jtour_mode.reset_index()
    tm1_jtour_mode = tm1_jtour_mode.pivot(
        index="tour_mode_labels", columns="tour_category"
    )

    tm1_tour_mode = pd.concat([tm1_nmn_tour_mode, tm1_jtour_mode], axis=1)
    tm1_tour_mode.columns = ["atwork", "non_mandatory", "mandatory", "joint"]
    tm1_tour_mode = tm1_tour_mode[["atwork", "joint", "mandatory", "non_mandatory"]]
    tm1_tour_mode.to_csv("outputs/tm1_tour_mode.csv", na_rep=0)

asim_tours = pd.read_csv(asim_tour_filename)
asim_tours["tour_mode"] = pd.Categorical(
    asim_tours["tour_mode"], categories=mode_labels
)
asim_tour_mode = asim_tours.groupby(["tour_mode", "tour_category"]).count()[
    "household_id"
]
asim_tour_mode = asim_tour_mode.reset_index()
asim_tour_mode = asim_tour_mode.pivot(index="tour_mode", columns="tour_category")
asim_tour_mode.to_csv("outputs/asim_tour_mode.csv", na_rep=0)

# atwork_subtour_frequency

if process_tm1:
    tm1_work_tours = tm1_tours[tm1_tours["tour_purpose"].str.startswith("work")]
    tm1_atwork_freq_strs = [
        "",
        "no_subtours",
        "eat",
        "business1",
        "maint",
        "business2",
        "eat_business",
    ]
    tm1_work_tours["atWork_freq_str"] = pd.Series(tm1_atwork_freq_strs)[
        tm1_work_tours["atWork_freq"].tolist()
    ].tolist()
    tm1_work_tours.groupby(["atWork_freq_str"]).count()["hh_id"].to_csv(
        "outputs/tm1_atwork_tf.csv", na_rep=0
    )

asim_work_tours = asim_tours[asim_tours["primary_purpose"] == "work"]
asim_work_tours.groupby(["atwork_subtour_frequency"]).count()["household_id"].to_csv(
    "outputs/asim_atwork_tf.csv", na_rep=0
)

# atwork_subtour_destination

if process_tm1:
    tm1_tours = pd.read_csv(tm1_tour_filename)
    tm1_tours["distance"] = distmat[
        tm1_tours["orig_taz"] - 1, tm1_tours["dest_taz"] - 1
    ]
    tm1_tours_atw = tm1_tours[tm1_tours["tour_category"] == "AT_WORK"]
    tm1_tours_atw["dist_bin"] = pd.cut(tm1_tours_atw["distance"], range(51))
    tm1_tours_atw.groupby(["dist_bin"]).count()["hh_id"].to_csv(
        "outputs/tm1_atwork_dist.csv", na_rep=0
    )

asim_atw_tours = pd.read_csv(asim_tour_filename)
asim_atw_tours = asim_atw_tours[asim_atw_tours["tour_category"] == "atwork"]
asim_atw_tours["distance"] = distmat[
    asim_atw_tours["origin"].astype(int) - 1,
    asim_atw_tours["destination"].astype(int) - 1,
]
asim_atw_tours["dist_bin"] = pd.cut(asim_atw_tours["distance"], range(51))
asim_atw_tours.groupby(["dist_bin"]).count()["household_id"].to_csv(
    "outputs/asim_atwork_dist.csv", na_rep=0
)

# atwork_subtour_scheduling

if process_tm1:
    tm1_tours_atw_tdd = tm1_tours_atw.groupby(["start_hour", "end_hour"]).count()[
        "hh_id"
    ]
    tm1_tours_atw_tdd = tm1_tours_atw_tdd.reset_index()
    tm1_tours_atw_tdd = tm1_tours_atw_tdd.pivot(index="start_hour", columns="end_hour")
    tm1_tours_atw_tdd.to_csv("outputs/tm1_atwork_tours_tdd.csv", na_rep=0)

asim_atw_tours_tdd = asim_atw_tours.groupby(["start", "end"]).count()["household_id"]
asim_atw_tours_tdd = asim_atw_tours_tdd.reset_index()
asim_atw_tours_tdd = asim_atw_tours_tdd.pivot(index="start", columns="end")
asim_atw_tours_tdd.to_csv("outputs/asim_atwork_tours_tdd.csv", na_rep=0)

# atwork_subtour_mode_choice - see tour mode above

# tour stop frequency

if process_tm1:
    tm1_tours = pd.read_csv(tm1_tour_filename)
    tm1_jtours = pd.read_csv(tm1_jtour_filename)

    tm1_tours["tour_purpose_simple"] = tm1_tours["tour_purpose"]
    tm1_tours["tour_purpose_simple"] = tm1_tours["tour_purpose_simple"].str.replace(
        "atwork_", ""
    )
    tm1_tours["tour_purpose_simple"][
        tm1_tours["tour_purpose_simple"].str.contains("work_")
    ] = "work"
    tm1_tours["tour_purpose_simple"][
        tm1_tours["tour_purpose_simple"].str.contains("school_")
    ] = "school"
    tm1_tours["tour_purpose_simple"][
        tm1_tours["tour_purpose_simple"].str.contains("university")
    ] = "school"
    tm1_tours["tour_purpose_simple"][
        tm1_tours["tour_purpose_simple"].str.contains("escort_")
    ] = "escort"
    tm1_tours_atw = tm1_tours[tm1_tours["tour_category"] == "AT_WORK"]
    tm1_tours_nmn = tm1_tours[tm1_tours["tour_category"] != "AT_WORK"]

    tm1_tours_nmn["tsf"] = (
        tm1_tours_nmn["num_ob_stops"].astype(str)
        + "-"
        + tm1_tours_nmn["num_ib_stops"].astype(str)
    )
    tm1_stop_freq = tm1_tours_nmn.groupby(["tsf", "tour_purpose_simple"]).count()[
        "hh_id"
    ]
    tm1_stop_freq = tm1_stop_freq.reset_index()
    tm1_stop_freq = tm1_stop_freq.pivot(index="tsf", columns="tour_purpose_simple")

    tm1_jtours["tsf"] = (
        tm1_jtours["num_ob_stops"].astype(str)
        + "-"
        + tm1_jtours["num_ib_stops"].astype(str)
    )
    tm1_tours_atw["tsf"] = (
        tm1_tours_atw["num_ob_stops"].astype(str)
        + "-"
        + tm1_tours_atw["num_ib_stops"].astype(str)
    )

    tm1_stop_freq_joint = tm1_jtours.groupby(["tsf"]).count()["hh_id"]
    tm1_stop_freq_atwork = tm1_tours_atw.groupby(["tsf"]).count()["hh_id"]
    tm1_stop_freq = pd.concat(
        [tm1_stop_freq, tm1_stop_freq_joint, tm1_stop_freq_atwork], axis=1
    )
    tm1_stop_freq.to_csv("outputs/tm1_stop_freq.csv", na_rep=0)

asim_tours = pd.read_csv(asim_tour_filename)
asim_nmn_tours = asim_tours[
    (asim_tours["tour_category"] == "mandatory")
    | (asim_tours["tour_category"] == "non_mandatory")
]
asim_joint_tours = asim_tours[asim_tours["tour_category"] == "joint"]
asim_atw_tours = asim_tours[asim_tours["tour_category"] == "atwork"]
asim_stop_freq = asim_nmn_tours.groupby(["stop_frequency", "tour_type"]).count()[
    "household_id"
]
asim_stop_freq = asim_stop_freq.reset_index()
asim_stop_freq = asim_stop_freq.pivot(index="stop_frequency", columns="tour_type")

asim_stop_freq_joint = asim_joint_tours.groupby(["stop_frequency"]).count()[
    "household_id"
]
asim_stop_freq_atwork = asim_atw_tours.groupby(["stop_frequency"]).count()[
    "household_id"
]
asim_stop_freq = pd.concat(
    [asim_stop_freq, asim_stop_freq_joint, asim_stop_freq_atwork], axis=1
)
asim_stop_freq.to_csv("outputs/asim_stop_freq.csv", na_rep=0)

#############################################################
# TRIP
#############################################################

# trip purpose

if process_tm1:
    tm1_trips = pd.read_csv(tm1_trips_filename)
    tm1_jtrips = pd.read_csv(tm1_jtrips_filename)
    tm1_trips["orig_purpose"][tm1_trips["orig_purpose"] == "university"] = "univ"
    tm1_trips["orig_purpose"] = pd.Categorical(tm1_trips["orig_purpose"])
    tm1_jtrips["orig_purpose"] = pd.Categorical(
        tm1_jtrips["orig_purpose"], categories=tm1_trips["orig_purpose"].cat.categories
    )
    tm1_trip_purp = tm1_trips.groupby(["orig_purpose", "tour_category"]).count()[
        "hh_id"
    ]
    tm1_trip_purp = tm1_trip_purp.reset_index()
    tm1_trip_purp = tm1_trip_purp.pivot(index="orig_purpose", columns="tour_category")
    tm1_jtrip_purp = tm1_jtrips.groupby(["orig_purpose"]).count()["hh_id"]
    tm1_trip_purp = pd.concat([tm1_trip_purp, tm1_jtrip_purp], axis=1)
    tm1_trip_purp.columns = ["atwork", "non_mandatory", "mandatory", "joint"]
    tm1_trip_purp = tm1_trip_purp[["atwork", "joint", "mandatory", "non_mandatory"]]
    tm1_trip_purp.to_csv("outputs/tm1_trip_purp.csv", na_rep=0)

asim_trips = pd.read_csv(asim_trips_filename)
asim_tours = pd.read_csv(asim_tour_filename)
asim_tours = asim_tours.set_index("tour_id", drop=False)
asim_trips["tour_category"] = (
    asim_tours["tour_category"].loc[asim_trips["tour_id"]].tolist()
)
asim_trip_purp = asim_trips.groupby(["purpose", "tour_category"]).count()[
    "household_id"
]
asim_trip_purp = asim_trip_purp.reset_index()
asim_trip_purp = asim_trip_purp.pivot(index="purpose", columns="tour_category")
asim_trip_purp.to_csv("outputs/asim_trip_purp.csv", na_rep=0)

# trip destination

if process_tm1:
    tm1_trips["distance"] = distmat[
        tm1_trips["orig_taz"] - 1, tm1_trips["dest_taz"] - 1
    ]
    tm1_jtrips["distance"] = distmat[
        tm1_jtrips["orig_taz"] - 1, tm1_jtrips["dest_taz"] - 1
    ]
    tm1_trips["dist_bin"] = pd.cut(tm1_trips["distance"], range(51))
    tm1_jtrips["dist_bin"] = pd.cut(tm1_jtrips["distance"], range(51))
    tm1_trips_dist = pd.concat(
        [
            tm1_trips.groupby(["dist_bin"]).count()["hh_id"]
            + tm1_jtrips.groupby(["dist_bin"]).count()["hh_id"]
        ],
        axis=1,
    )
    tm1_trips_dist.to_csv("outputs/tm1_trips_dist.csv", na_rep=0)

asim_trips["distance"] = distmat[
    asim_trips["origin"] - 1, asim_trips["destination"] - 1
]
asim_trips["dist_bin"] = pd.cut(asim_trips["distance"], range(51))
asim_trips.groupby(["dist_bin"]).count()["household_id"].to_csv(
    "outputs/asim_trips_dist.csv", na_rep=0
)

# trip scheduling

if process_tm1:
    tm1_trips_tdd = (
        tm1_trips.groupby(["depart_hour"]).count()["hh_id"]
        + tm1_jtrips.groupby(["depart_hour"]).count()["hh_id"]
    )
    tm1_trips_tdd.to_csv("outputs/tm1_trips_depart.csv", na_rep=0)

asim_trips_tdd = asim_trips.groupby(["depart"]).count()["household_id"]
asim_trips_tdd.to_csv("outputs/asim_trips_depart.csv", na_rep=0)

# trip mode share by tour purpose

if process_tm1:
    tm1_trips["trip_mode_str"] = pd.Series(mode_labels)[
        tm1_trips["trip_mode"].tolist()
    ].tolist()
    tm1_trips["trip_mode_str"] = pd.Categorical(
        tm1_trips["trip_mode_str"], categories=mode_labels
    )
    tm1_jtrips["trip_mode_str"] = pd.Series(mode_labels)[
        tm1_jtrips["trip_mode"].tolist()
    ].tolist()
    tm1_jtrips["trip_mode_str"] = pd.Categorical(
        tm1_jtrips["trip_mode_str"], categories=mode_labels
    )

    tm1_trip_mode = tm1_trips.groupby(["trip_mode_str", "tour_category"]).count()[
        "hh_id"
    ]
    tm1_trip_mode = tm1_trip_mode.reset_index()
    tm1_trip_mode = tm1_trip_mode.pivot(index="trip_mode_str", columns="tour_category")

    tm1_jtrip_mode = tm1_jtrips.groupby(["trip_mode_str"]).count()["hh_id"]
    tm1_trip_mode = pd.concat([tm1_trip_mode, tm1_jtrip_mode], axis=1)
    tm1_trip_mode.columns = ["atwork", "non_mandatory", "mandatory", "joint"]
    tm1_trip_mode = tm1_trip_mode[["atwork", "joint", "mandatory", "non_mandatory"]]
    tm1_trip_mode.to_csv("outputs/tm1_trip_mode.csv", na_rep=0)

asim_trips["trip_mode"] = pd.Categorical(
    asim_trips["trip_mode"], categories=mode_labels
)
asim_trip_mode = asim_trips.groupby(["trip_mode", "tour_category"]).count()[
    "household_id"
]
asim_trip_mode = asim_trip_mode.reset_index()
asim_trip_mode = asim_trip_mode.pivot(index="trip_mode", columns="tour_category")
asim_trip_mode.to_csv("outputs/asim_trip_mode.csv", na_rep=0)
