import orca
from activitysim import defaults
from activitysim import tracing
import pandas as pd
import numpy as np
import os

from activitysim.tracing import print_elapsed_time


def set_random_seed():
    np.random.seed(0)


def run_model(model_name):
    t0 = print_elapsed_time()
    orca.run([model_name])
    t0 = print_elapsed_time(model_name, t0)


# uncomment the line below to set random seed so that run results are reproducible
# orca.add_injectable("set_random_seed", set_random_seed)

tracing.config_logger()

t0 = print_elapsed_time()


# force load of skims
skim_stack = orca.get_injectable('skim_stack')


t0 = print_elapsed_time("all models", t0)
