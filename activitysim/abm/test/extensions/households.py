import numpy as np
import pandas as pd

from activitysim.core.util import reindex
from activitysim.core import inject


@inject.column("households")
def income_in_thousands(households):
    return households.income / 1000


@inject.column("households")
def income_segment(households):
    return pd.cut(households.income_in_thousands,
                  bins=[-np.inf, 30, 60, 100, np.inf],
                  labels=[1, 2, 3, 4]).astype(int)


@inject.column("households")
def non_workers(households, persons):
    return persons.household_id.value_counts() - households.workers


@inject.column("households")
def drivers(households, persons):
    # we assume that everyone 16 and older is a potential driver
    return persons.local.query("16 <= age").\
        groupby("household_id").size().\
        reindex(households.index).fillna(0)


@inject.column("households")
def num_young_children(households, persons):
    return persons.local.query("age <= 4").\
        groupby("household_id").size().\
        reindex(households.index).fillna(0)


@inject.column("households")
def num_children(households, persons):
    return persons.local.query("5 <= age <= 15").\
        groupby("household_id").size().\
        reindex(households.index).fillna(0)


@inject.column("households")
def num_adolescents(households, persons):
    return persons.local.query("16 <= age <= 17").\
        groupby("household_id").size().\
        reindex(households.index).fillna(0)


@inject.column("households")
def num_college_age(households, persons):
    return persons.local.query("18 <= age <= 24").\
        groupby("household_id").size().\
        reindex(households.index).fillna(0)


@inject.column("households")
def num_young_adults(households, persons):
    return persons.local.query("25 <= age <= 34").\
        groupby("household_id").size().\
        reindex(households.index).fillna(0)


# just a rename / alias
@inject.column("households")
def home_taz(households):
    return households.TAZ


# map household type ids to strings
@inject.column("households")
def household_type(households, settings):
    return households.HHT.map(settings["household_type_map"])


@inject.column("households")
def non_family(households):
    return households.household_type.isin(["nonfamily_male_alone",
                                           "nonfamily_male_notalone",
                                           "nonfamily_female_alone",
                                           "nonfamily_female_notalone"])


# can't just invert these unfortunately because there's a null household type
@inject.column("households")
def family(households):
    return households.household_type.isin(["family_married",
                                           "family_male",
                                           "family_female"])


# FIXME - not sure why we would need this since it is added by auto_ownership model
# @inject.column('households')
# def auto_ownership(households):
#     return pd.Series(0, households.index)


@inject.column('households')
def hhsize(households):
    return households.PERSONS


@inject.column('households')
def home_is_urban(households, land_use, settings):
    s = reindex(land_use.area_type, households.home_taz)
    return s < settings['urban_threshold']


@inject.column('households')
def home_is_rural(households, land_use, settings):
    s = reindex(land_use.area_type, households.home_taz)
    return s > settings['rural_threshold']
