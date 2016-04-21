import orca
from activitysim import defaults
import pandas as pd
import numpy as np
import os


orca.run(["school_location_simulate"])
orca.run(["workplace_location_simulate"])
print orca.get_table("persons").distance_to_work.describe()
orca.run(["auto_ownership_simulate"])
orca.run(["cdap_simulate"])
orca.run(['mandatory_tour_frequency'])
orca.get_table("mandatory_tours").tour_type.value_counts()
orca.run(["mandatory_scheduling"])
orca.run(['non_mandatory_tour_frequency'])
orca.get_table("non_mandatory_tours").tour_type.value_counts()
orca.run(["destination_choice"])
orca.run(["non_mandatory_scheduling"])

# FIXME - jwd - choose more felicitous name or do this elsewhere?
orca.run(["patch_mandatory_tour_destination"])

orca.run(['tour_mode_choice_simulate'])
orca.run(['trip_mode_choice_simulate'])
