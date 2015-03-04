import urbansim.sim.simulation as sim
import warnings
import os
import yaml
import pandas as pd
import numpy as np

"""
Definition of terms:

CDAP = coordinated daily activity pattern
TDD = tour departure and duration
"""


warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)
pd.options.mode.chained_assignment = None


@sim.injectable(cache=True)
def settings():
    with open(os.path.join("configs", "settings.yaml")) as f:
        settings = yaml.load(f)
        # monkey patch on the settings object since it's pretty global
        # but will also be available as injectable
        sim.settings = settings
        return settings


@sim.injectable(cache=True)
def store(settings):
    return pd.HDFStore(
        os.path.join("data", settings["store"]),
        mode='r')


# these are the alternatives for the workplace choice
@sim.table()
def zones():
    # I grant this is a weird idiom but it helps to name the index
    return pd.DataFrame({"TAZ": np.arange(1454)+1}).set_index("TAZ")
