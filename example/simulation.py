import orca
from activitysim import defaults
from activitysim import tracing
import pandas as pd
import numpy as np
import os

from activitysim.tracing import print_elapsed_time


def run_model(model_name):
    t0 = print_elapsed_time()
    orca.run([model_name])
    t0 = print_elapsed_time(model_name, t0)
    
orca.add_injectable("output_dir", 'output')
tracing.config_logger()

t0 = print_elapsed_time()

run_model("compute_accessibility")
run_model("school_location_simulate")
run_model("workplace_location_simulate")
print orca.get_table("persons").distance_to_work.describe()
run_model("auto_ownership_simulate")
run_model("cdap_simulate")
run_model('mandatory_tour_frequency')
orca.get_table("mandatory_tours").tour_type.value_counts()
run_model("mandatory_scheduling")
run_model('non_mandatory_tour_frequency')
orca.get_table("non_mandatory_tours").tour_type.value_counts()
run_model("destination_choice")
run_model("non_mandatory_scheduling")

# FIXME - jwd - choose more felicitous name or do this elsewhere?
run_model("patch_mandatory_tour_destination")

run_model('tour_mode_choice_simulate')
run_model('trip_mode_choice_simulate')

t0 = print_elapsed_time("all models", t0)
