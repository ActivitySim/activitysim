# ActivitySim
# Copyright (C) 2014-2015 Synthicity, LLC
# See full license in LICENSE.txt.

import pandas as pd
import numpy as np
import urbansim.sim.simulation as sim
from activitysim.defaults import datasources


@sim.column("households")
def income_in_thousands(households):
    return households.income / 1000


@sim.column("households")
def income_segment(households):
    return pd.cut(households.income_in_thousands,
                  bins=[-np.inf, 30, 60, 100, np.inf],
                  labels=[1, 2, 3, 4])


@sim.column("households")
def drivers(households, persons):
    # we assume that everyone 16 and older is a potential driver
    return persons.local.query("16 <= age").\
        groupby("household_id").size().\
        reindex(households.index).fillna(0)


@sim.column("households")
def num_young_children(households, persons):
    return persons.local.query("age <= 4").\
        groupby("household_id").size().\
        reindex(households.index).fillna(0)


@sim.column("households")
def num_children(households, persons):
    return persons.local.query("5 <= age <= 15").\
        groupby("household_id").size().\
        reindex(households.index).fillna(0)


@sim.column("households")
def num_adolescents(households, persons):
    return persons.local.query("16 <= age <= 17").\
        groupby("household_id").size().\
        reindex(households.index).fillna(0)


@sim.column("households")
def num_college_age(households, persons):
    return persons.local.query("18 <= age <= 24").\
        groupby("household_id").size().\
        reindex(households.index).fillna(0)


@sim.column("households")
def num_young_adults(households, persons):
    return persons.local.query("25 <= age <= 34").\
        groupby("household_id").size().\
        reindex(households.index).fillna(0)


@sim.column("land_use")
def household_density(land_use):
    return land_use.total_households / land_use.total_acres


@sim.column("land_use")
def employment_density(land_use):
    return land_use.total_employment / land_use.total_acres


@sim.column("land_use")
def density_index(land_use):
    return (land_use.household_density * land_use.employment_density) / \
        (land_use.household_density + land_use.employment_density)


@sim.column("land_use")
def county_name(land_use, settings):
    assert "county_map" in settings
    inv_map = {v: k for k, v in settings["county_map"].items()}
    return land_use.county_id.map(inv_map)
