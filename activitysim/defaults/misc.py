# ActivitySim
# See full license in LICENSE.txt.

import os
import warnings

import numpy as np
import orca
import pandas as pd
import yaml


warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)
pd.options.mode.chained_assignment = None


@orca.injectable()
def set_random_seed():
    pass


@orca.injectable()
def configs_dir():
    return '.'


@orca.injectable()
def data_dir():
    return '.'


@orca.injectable()
def settings(configs_dir):
    with open(os.path.join(configs_dir, "configs", "settings.yaml")) as f:
        return yaml.load(f)


@orca.injectable(cache=True)
def store(data_dir, settings):
    return pd.HDFStore(os.path.join(data_dir, "data", settings["store"]),
                       mode='r')


@orca.injectable(cache=True)
def preload_3d_skims(settings):
    return bool(settings.get('preload_3d_skims', False))


@orca.injectable(cache=True)
def cache_lazy_loaded_skims(settings):
    return bool(settings.get('cache_lazy_loaded_skims', False))


@orca.injectable(cache=True)
def cache_skim_key_values(settings, preload_3d_skims, cache_lazy_loaded_skims):
    if preload_3d_skims or cache_lazy_loaded_skims:
        return settings['time_periods']['labels']
    else:
        return None
