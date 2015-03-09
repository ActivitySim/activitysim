# ActivitySim
# Copyright (C) 2014-2015 Synthicity, LLC
# See full license in LICENSE.txt.

import numpy as np
import pandas as pd
import os
import uuid
import yaml
from urbansim.utils import misc
import urbansim.sim.simulation as sim
from .. import activitysim as asim

import warnings

warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)
pd.options.mode.chained_assignment = None


@sim.injectable(cache=True)
def settings():
    with open(os.path.join(misc.configs_dir(), "settings.yaml")) as f:
        settings = yaml.load(f)
        # monkey patch on the settings object since it's pretty global
        # but will also be available as injectable
        sim.settings = settings
        return settings


@sim.injectable()
def run_number():
    return misc.get_run_number()


@sim.injectable(cache=True)
def uuid():
    return uuid.uuid4().hex


@sim.injectable(cache=True)
def store(settings):
    return pd.HDFStore(
        os.path.join(misc.data_dir(), settings["store"]),
        mode='r')


@sim.injectable()
def scenario(settings):
    return settings["scenario"]


@sim.table(cache=True)
def land_use(store):
    return store["land_use/taz_data"]


@sim.table(cache=True)
def accessibility(store):
    df = store["skims/accessibility"]
    df.columns = [c.upper() for c in df.columns]
    return df


@sim.table(cache=True)
def households(store, settings):

    if "households_sample_size" in settings:
        return asim.random_rows(store["households"],
                                settings["households_sample_size"])

    return store["households"]


@sim.table(cache=True)
def persons(store, settings, households):
    df = store["persons"]

    if "households_sample_size" in settings:
        # keep all persons in the sampled households
        df = df[df.household_id.isin(households.index)]

    return df


sim.broadcast('households', 'persons', cast_index=True, onto_on='household_id')
sim.broadcast('land_use', 'households', cast_index=True, onto_on='TAZ')
sim.broadcast('accessibility', 'households', cast_index=True, onto_on='TAZ')
