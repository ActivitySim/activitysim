import orca
from activitysim import abm
from activitysim.core import tracing
import pandas as pd
import numpy as np
import os

from activitysim.core.tracing import print_elapsed_time

from activitysim.core import pipeline
import extensions


# comment out the line below to default base seed to 0 random seed
# so that run results are reproducible
# pipeline.set_rn_generator_base_seed(seed=None)


tracing.config_logger()

t0 = print_elapsed_time()

_MODELS = [
    'compute_accessibility',
    'school_location_sample',
    'school_location_logsums',
    'school_location_simulate',
    'workplace_location_sample',
    'workplace_location_logsums',
    'workplace_location_simulate',
    'auto_ownership_simulate',
    'cdap_simulate',
    'mandatory_tour_frequency',
    'mandatory_scheduling',
    'non_mandatory_tour_frequency',
    'destination_choice',
    'non_mandatory_scheduling',
    'tour_mode_choice_simulate',
    'create_simple_trips',
    'trip_mode_choice_simulate'
]


# If you provide a resume_after argument to pipeline.run
# the pipeline manager will attempt to load checkpointed tables from the checkpoint store
# and resume pipeline processing on the next submodel step after the specified checkpoint
resume_after = None
# resume_after = 'mandatory_scheduling'

pipeline.run(models=_MODELS, resume_after=resume_after)

print "\n#### run completed"

# retrieve the state of a checkpointed table after a specific model was run
df = pipeline.get_table(table_name="persons", checkpoint_name="school_location_simulate")
print "\npersons table columns after school_location_simulate:", df.columns.values

# get_table without checkpoint_name returns the latest version of the table
df = pipeline.get_table("tours")
print "\ntour_type value counts\n", df.tour_type.value_counts()

# get_table for a computed (non-checkpointed, internal, orca) table
# return the most recent value of a (non-checkpointed, internal) computed table
df = pipeline.get_table("persons_merged")
df = df[['household_id', 'age', 'auPkTotal', 'roundtrip_auto_time_to_work']]
print "\npersons_merged selected columns\n", df.head(20)

# write final versions of all checkpointed dataframes to CSV files to review results
for table_name in pipeline.checkpointed_tables():
    file_name = "final_%s_table.csv" % table_name
    file_path = os.path.join(orca.get_injectable("output_dir"), file_name)
    pipeline.get_table(table_name).to_csv(file_path)

# tables will no longer be available after pipeline is closed
pipeline.close()

# write checkpoints (this can be called whether of not pipeline is open)
file_path = os.path.join(orca.get_injectable("output_dir"), "checkpoints.csv")
pipeline.get_checkpoints().to_csv(file_path)

t0 = print_elapsed_time("all models", t0)
