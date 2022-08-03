# ActivitySim
# See full license in LICENSE.txt.

import logging
import os
import sys

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
logger.addHandler(ch)


inputs = {
    "households": "final_households.csv",
    "persons": "final_persons.csv",
    "tours": "final_tours.csv",
    "joint_tour_participants": "final_joint_tour_participants.csv",
    "trips": "final_trips.csv",
}

surveys = {
    "households": "survey_households.csv",
    "persons": "survey_persons.csv",
    "tours": "survey_tours.csv",
    "joint_tour_participants": "survey_joint_tour_participants.csv",
    "trips": "survey_trips.csv",
}


args = sys.argv[1:]
assert len(args) == 1, "usage extract_survey_data.py <data_dir>"

data_dir = args[0]

input_dir = os.path.join(data_dir, "survey_data/")
output_dir = os.path.join(data_dir, "survey_data/")

configs_dir = os.path.dirname("../example/configs/")

households = pd.read_csv(os.path.join(input_dir, inputs["households"]))
persons = pd.read_csv(os.path.join(input_dir, inputs["persons"]))
tours = pd.read_csv(os.path.join(input_dir, inputs["tours"]))
joint_tour_participants = pd.read_csv(
    os.path.join(input_dir, inputs["joint_tour_participants"])
)
trips = pd.read_csv(os.path.join(input_dir, inputs["trips"]))

households = households[
    [
        "household_id",
        "home_zone_id",
        "income",
        "hhsize",
        "HHT",
        "auto_ownership",
        "num_workers",
    ]
]
persons = persons[
    [
        "person_id",
        "household_id",
        "age",
        "PNUM",
        "sex",
        "pemploy",
        "pstudent",
        "ptype",
        "school_zone_id",
        "workplace_zone_id",
        "free_parking_at_work",
    ]
]
tours = tours[
    [
        "tour_id",
        "person_id",
        "household_id",
        "tour_type",
        "tour_category",
        "destination",
        "origin",
        "start",
        "end",
        "tour_mode",
        "parent_tour_id",
    ]
]
joint_tour_participants = joint_tour_participants[
    ["participant_id", "tour_id", "household_id", "person_id", "participant_num"]
]

OPTIONAL_TRIP_COLUMNS = []
# FIXME trip order is ambiguous if two trips have same start time?
# OPTIONAL_TRIP_COLUMNS = ['trip_num']

trips = trips[
    [
        "trip_id",
        "person_id",
        "household_id",
        "tour_id",
        "outbound",
        "purpose",
        "destination",
        "origin",
        "depart",
        "trip_mode",
    ]
    + OPTIONAL_TRIP_COLUMNS
]

households.to_csv(os.path.join(output_dir, surveys["households"]), index=False)
persons.to_csv(os.path.join(output_dir, surveys["persons"]), index=False)
tours.to_csv(os.path.join(output_dir, surveys["tours"]), index=False)
joint_tour_participants.to_csv(
    os.path.join(output_dir, surveys["joint_tour_participants"]), index=False
)
trips.to_csv(os.path.join(output_dir, surveys["trips"]), index=False)
