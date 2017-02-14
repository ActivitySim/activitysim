import orca
from activitysim import defaults
from activitysim import tracing
from activitysim import activitysim as asim
import pandas as pd
import numpy as np
import os

from activitysim.tracing import print_elapsed_time


# you will want to configure this with the locations of the canonical datasets
DATA_REPO = os.path.join(os.path.dirname(__file__), '..', '..', 'activitysim-data')

@orca.injectable()
def data_dir():
    return os.path.join(DATA_REPO, 'multi_zone')


def set_random_seed():
    np.random.seed(0)


def run_model(model_name):
    t0 = print_elapsed_time()
    orca.run([model_name])
    t0 = print_elapsed_time(model_name, t0)


import extensions

# uncomment the line below to set random seed so that run results are reproducible
# orca.add_injectable("set_random_seed", set_random_seed)

tracing.config_logger()

t0 = print_elapsed_time()


t0 = print_elapsed_time()

taz_skim_stack = orca.get_injectable('taz_skim_stack')
t0 = print_elapsed_time("load taz_skim_stack", t0)

tap_skim_stack = orca.get_injectable('tap_skim_stack')
t0 = print_elapsed_time("load tap_skim_stack", t0)

network_los = orca.get_injectable('network_los')
t0 = print_elapsed_time("load network_los", t0)


random_taz = asim.random_rows(network_los.taz, 10).TAZ

print "random_taz", random_taz

t0 = print_elapsed_time("all models", t0)
