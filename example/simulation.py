import orca
from activitysim import defaults
from activitysim import tracing
import pandas as pd
import numpy as np
import os

from activitysim.tracing import print_elapsed_time

from activitysim import pipeline
import extensions


def set_random_seed():
    np.random.seed(0)

# uncomment the line below to set random seed so that run results are reproducible
# orca.add_injectable("set_random_seed", set_random_seed)

tracing.config_logger()

t0 = print_elapsed_time()

_MODELS = [
    'compute_accessibility',
    'school_location_simulate',
    'workplace_location_simulate',
    'auto_ownership_simulate',
    'cdap_simulate',
    'mandatory_tour_frequency',
    'mandatory_scheduling',
    'non_mandatory_tour_frequency',
    'destination_choice',
    'non_mandatory_scheduling',
    'tour_mode_choice_simulate',
    'trip_mode_choice_simulate'
]


resume_after = 'mandatory_scheduling'
resume_after = None

pipeline.run(resume_after=resume_after)

print pipeline.get_table("persons").distance_to_work.describe()
print pipeline.get_table("mandatory_tours").tour_type.value_counts()
print pipeline.get_table("non_mandatory_tours").tour_type.value_counts()

# write final households table to a CSV file to review results
hh_outfile_name = os.path.join(orca.get_injectable("output_dir"), "final_households_table.csv")
pipeline.get_table('households').to_csv(hh_outfile_name)

# write initial households table to a CSV file to review results
hh_outfile_name = os.path.join(orca.get_injectable("output_dir"), "initia_households_table.csv")
pipeline.get_table('households', checkpoint_name = 'init').to_csv(hh_outfile_name)


# write final households table to a CSV file to review results
hh_outfile_name = os.path.join(orca.get_injectable("output_dir"), "final_persons_table.csv")
pipeline.get_table('persons').to_csv(hh_outfile_name)


# write final households table to a CSV file to review results
hh_outfile_name = os.path.join(orca.get_injectable("output_dir"), "final_mandatory_tours.csv")
pipeline.get_table('mandatory_tours').to_csv(hh_outfile_name)

# write final households table to a CSV file to review results
hh_outfile_name = os.path.join(orca.get_injectable("output_dir"), "final_non_mandatory_tours.csv")
pipeline.get_table('non_mandatory_tours').to_csv(hh_outfile_name)

# write final households table to a CSV file to review results
hh_outfile_name = os.path.join(orca.get_injectable("output_dir"), "final_tours.csv")
orca.get_table('tours').to_frame().to_csv(hh_outfile_name)

pipeline.close()

t0 = print_elapsed_time("all models", t0)
