# ActivitySim
# See full license in LICENSE.txt.

import numpy as np
import orca
import pandas as pd

from activitysim import activitysim as asim
from activitysim.util import reindex

# not actually used, but helpful for dumping/documenting the contents of store
# @orca.table(cache=True)
# def households_internal(store, settings):
#     df = store["households"]
#     return df


@orca.table(cache=True)
def households(set_random_seed, store, settings):

    if "households_sample_size" in settings:
        return asim.random_rows(store["households"],
                                settings["households_sample_size"])

    return store["households"]


# this is a placeholder table for columns that get computed after the
# auto ownership model
@orca.table()
def households_autoown(households):
    return pd.DataFrame(index=households.index)


# this is a common merge so might as well define it once here and use it
@orca.table()
def households_merged(households, land_use, accessibility):
    return orca.merge_tables(households.name, tables=[
        households, land_use, accessibility])


orca.broadcast('households', 'persons', cast_index=True, onto_on='household_id')


@orca.column("households")
def income_in_thousands(households):
    return households.income / 1000


@orca.column("households")
def income_segment(households):
    return pd.cut(households.income_in_thousands,
                  bins=[-np.inf, 30, 60, 100, np.inf],
                  labels=[1, 2, 3, 4])


@orca.column("households")
def non_workers(households, persons):
    return persons.household_id.value_counts() - households.workers


@orca.column("households")
def drivers(households, persons):
    # we assume that everyone 16 and older is a potential driver
    return persons.local.query("16 <= age").\
        groupby("household_id").size().\
        reindex(households.index).fillna(0)


@orca.column("households")
def num_young_children(households, persons):
    return persons.local.query("age <= 4").\
        groupby("household_id").size().\
        reindex(households.index).fillna(0)


@orca.column("households")
def num_children(households, persons):
    return persons.local.query("5 <= age <= 15").\
        groupby("household_id").size().\
        reindex(households.index).fillna(0)


@orca.column("households")
def num_adolescents(households, persons):
    return persons.local.query("16 <= age <= 17").\
        groupby("household_id").size().\
        reindex(households.index).fillna(0)


@orca.column("households")
def num_college_age(households, persons):
    return persons.local.query("18 <= age <= 24").\
        groupby("household_id").size().\
        reindex(households.index).fillna(0)


@orca.column("households")
def num_young_adults(households, persons):
    return persons.local.query("25 <= age <= 34").\
        groupby("household_id").size().\
        reindex(households.index).fillna(0)


# just a rename / alias
@orca.column("households")
def home_taz(households):
    return households.TAZ


# map household type ids to strings
@orca.column("households")
def household_type(households, settings):
    return households.HHT.map(settings["household_type_map"])


@orca.column("households")
def non_family(households):
    return households.household_type.isin(["nonfamily_male_alone",
                                           "nonfamily_male_notalone",
                                           "nonfamily_female_alone",
                                           "nonfamily_female_notalone"])


# can't just invert these unfortunately because there's a null household type
@orca.column("households")
def family(households):
    return households.household_type.isin(["family_married",
                                           "family_male",
                                           "family_female"])


@orca.column("households")
def num_under16_not_at_school(persons, households):
    return persons.under16_not_at_school.groupby(persons.household_id).size().\
        reindex(households.index).fillna(0)


@orca.column('households')
def auto_ownership(households):
    return pd.Series(0, households.index)


@orca.column('households')
def hhsize(households):
    return households.PERSONS


@orca.column('households_autoown')
def no_cars(households):
    return (households.auto_ownership == 0)


@orca.column('households')
def home_is_urban(households, land_use, settings):
    s = reindex(land_use.area_type, households.home_taz)
    return s < settings['urban_threshold']


@orca.column('households')
def home_is_rural(households, land_use, settings):
    s = reindex(land_use.area_type, households.home_taz)
    return s > settings['rural_threshold']


@orca.column('households_autoown')
def car_sufficiency(households, persons):
    return households.auto_ownership - persons.household_id.value_counts()


@orca.column('households')
def work_tour_auto_time_savings(households):
    # TODO fix this variable from auto ownership model
    return pd.Series(0, households.index)
