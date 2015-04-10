import urbansim.sim.simulation as sim
import warnings
import os
import yaml
import pandas as pd
import numpy as np


warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)
pd.options.mode.chained_assignment = None


@sim.injectable()
def set_random_seed():
    pass


@sim.injectable()
def configs_dir():
    return '.'


@sim.injectable()
def data_dir():
    return '.'


@sim.injectable()
def settings(configs_dir):
    with open(os.path.join(configs_dir, "configs", "settings.yaml")) as f:
        return yaml.load(f)


@sim.injectable(cache=True)
def store(data_dir, settings):
    return pd.HDFStore(os.path.join(data_dir, "data", settings["store"]),
                       mode='r')
