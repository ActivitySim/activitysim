import orca
from activitysim import defaults
import pandas as pd
import numpy as np
import os

#orca.add_injectable("store", pd.HDFStore(
#        os.path.join("..", "activitysim", "defaults", "test", "test.h5"), "r"))
#orca.add_injectable("nonmotskm_matrix", np.ones((1454, 1454)))

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
orca.run(['mode_choice_simulate'])
