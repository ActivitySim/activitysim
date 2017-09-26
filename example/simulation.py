import orca
from activitysim import abm
from activitysim.core import tracing
import pandas as pd
import numpy as np
import os

from activitysim.core.tracing import print_elapsed_time
from activitysim.core.config import handle_standard_args
from activitysim.core.config import setting

from activitysim.core import pipeline
import extensions

handle_standard_args()

# comment out the line below to default base seed to 0 random seed
# so that run results are reproducible
# pipeline.set_rn_generator_base_seed(seed=None)

tracing.config_logger()

t0 = print_elapsed_time()

MODELS = setting('models')


# If you provide a resume_after argument to pipeline.run
# the pipeline manager will attempt to load checkpointed tables from the checkpoint store
# and resume pipeline processing on the next submodel step after the specified checkpoint
resume_after = setting('resume_after', None)

if resume_after:
    print "resume_after", resume_after

pipeline.run(models=MODELS, resume_after=resume_after)

print "\n#### run completed"

# write final versions of all checkpointed dataframes to CSV files to review results
for table_name in pipeline.checkpointed_tables():
    file_name = "final_%s_table.csv" % table_name
    file_path = os.path.join(orca.get_injectable("output_dir"), file_name)
    pipeline.get_table(table_name).to_csv(file_path)

# write checkpoints
file_path = os.path.join(orca.get_injectable("output_dir"), "checkpoints.csv")
pipeline.get_checkpoints().to_csv(file_path)

# tables will no longer be available after pipeline is closed
pipeline.close_pipeline()

t0 = print_elapsed_time("all models", t0)
