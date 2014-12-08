import numpy as np
import pandas as pd
import os
import uuid
import yaml
from urbansim.utils import misc
import urbansim.sim.simulation as sim

import warnings

warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)
pd.options.mode.chained_assignment = None


@sim.injectable('settings', cache=True)
def settings():
    with open(os.path.join(misc.configs_dir(), "settings.yaml")) as f:
        settings = yaml.load(f)
        # monkey patch on the settings object since it's pretty global
        # but will also be available as injectable
        sim.settings = settings
        return settings


@sim.injectable('run_number')
def run_number():
    return misc.get_run_number()


@sim.injectable('uuid', cache=True)
def uuid_hex():
    return uuid.uuid4().hex


@sim.injectable('store', cache=True)
def hdfstore(settings):
    return pd.HDFStore(
        os.path.join(misc.data_dir(), settings["store"]),
        mode='r')


@sim.injectable("scenario")
def scenario(settings):
    return settings["scenario"]


@sim.table("land_use_data", cache=True)
def land_use_data(store):
    return store["land_use/taz_data"]


@sim.table("accessibility", cache=True)
def land_use(store):
    return store["skims/accessibility"]
