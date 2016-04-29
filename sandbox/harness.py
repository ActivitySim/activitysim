import os
import time

import orca
from activitysim import defaults
import pandas as pd
import numpy as np
import openmatrix as omx
import yaml

def memory_usage_psutil():
    # return the memory usage in MB
    import psutil
    process = psutil.Process(os.getpid())
    mem = process.get_memory_info()[0] / float(2 ** 20)
    return mem


def set_random_seed():
    np.random.seed(0)

# set the max households for all tests (this is to limit memory use on travis)
HOUSEHOLDS_SAMPLE_SIZE = 50

@orca.injectable(cache=False)
def settings(configs_dir):
    with open(os.path.join(configs_dir, "configs", "settings.yaml")) as f:
        obj = yaml.load(f)
        obj['households_sample_size'] = HOUSEHOLDS_SAMPLE_SIZE
        return obj


orca.add_injectable("set_random_seed", set_random_seed)


orca.run(["school_location_simulate"])
orca.run(["workplace_location_simulate"])
#print orca.get_table("persons").distance_to_work.describe()
orca.run(["auto_ownership_simulate"])
#orca.run(["cdap_simulate"])
orca.run(['mandatory_tour_frequency'])
#orca.get_table("mandatory_tours").tour_type.value_counts()
orca.run(["mandatory_scheduling"])
orca.run(['non_mandatory_tour_frequency'])
#orca.get_table("non_mandatory_tours").tour_type.value_counts()
orca.run(["destination_choice"])
orca.run(["non_mandatory_scheduling"])

#print "mem pre-mode_choice_simulate:", memory_usage_psutil()
orca.run(['mode_choice_simulate'])
#print "mem post-mode_choice_simulate:", memory_usage_psutil()

# tmp.close()
# os.remove(tmp_name)


    # with orca.eval_variable('output_store') as output_store:
    #     # for troubleshooting, write table with benefits for each row in manifest
    #     output_store['aggregate_trips'] = results

