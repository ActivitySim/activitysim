import urbansim.sim.simulation as sim
from activitysim import defaults
import pandas as pd
import numpy as np
import os

sim.run(["school_location_simulate"])
sim.run(["workplace_location_simulate"])
print sim.get_table("persons").distance_to_work.describe()
sim.run(["auto_ownership_simulate"])
# sim.run(["cdap_simulate"])
sim.run(['mandatory_tour_frequency'])
sim.get_table("mandatory_tours").tour_type.value_counts()
sim.run(["mandatory_scheduling"])
sim.run(['non_mandatory_tour_frequency'])
sim.get_table("non_mandatory_tours").tour_type.value_counts()
sim.run(["destination_choice"])
sim.run(["non_mandatory_scheduling"])
sim.run(['mode_choice_simulate'])
