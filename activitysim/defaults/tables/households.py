import pandas as pd
import numpy as np
import urbansim.sim.simulation as sim
import urbansim.utils.misc as usim_misc
from activitysim import activitysim as asim


@sim.table(cache=True)
def households(set_random_seed, store, settings):

    if "households_sample_size" in settings:
        return asim.random_rows(store["households"],
                                settings["households_sample_size"])

    return store["households"]


# this is a placeholder table for columns that get computed after the
# auto ownership model
@sim.table()
def households_autoown(households):
    return pd.DataFrame(index=households.index)


# this is a common merge so might as well define it once here and use it
@sim.table()
def households_merged(households, land_use, accessibility):
    return sim.merge_tables(households.name, tables=[households,
                                                     land_use,
                                                     accessibility])


sim.broadcast('households', 'persons', cast_index=True, onto_on='household_id')


@sim.column("households")
def income_in_thousands(households):
    return households.income / 1000


@sim.column("households")
def income_segment(households):
    return pd.cut(households.income_in_thousands,
                  bins=[-np.inf, 30, 60, 100, np.inf],
                  labels=[1, 2, 3, 4])


@sim.column("households")
def non_workers(households, persons):
    return persons.household_id.value_counts() - households.workers


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


# just a rename / alias
@sim.column("households")
def home_taz(households):
    return households.TAZ


# map household type ids to strings
@sim.column("households")
def household_type(households, settings):
    return households.HHT.map(settings["household_type_map"])


@sim.column("households")
def non_family(households):
    return households.household_type.isin(["nonfamily_male_alone",
                                           "nonfamily_male_notalone",
                                           "nonfamily_female_alone",
                                           "nonfamily_female_notalone"])


# can't just invert these unfortunately because there's a null household type
@sim.column("households")
def family(households):
    return households.household_type.isin(["family_married",
                                           "family_male",
                                           "family_female"])


@sim.column("households")
def num_under16_not_at_school(persons, households):
    return persons.under16_not_at_school.groupby(persons.household_id).size().\
        reindex(households.index).fillna(0)


@sim.column('households')
def hhsize(households):
    return households.PERSONS


@sim.column('households_autoown')
def no_cars(households):
    return (households.auto_ownership == 0)


@sim.column('households')
def home_is_urban(households, land_use, settings):
    s = usim_misc.reindex(land_use.area_type, households.home_taz)
    return s < settings['urban_threshold']


@sim.column('households')
def home_is_rural(households, land_use, settings):
    s = usim_misc.reindex(land_use.area_type, households.home_taz)
    return s > settings['rural_threshold']


@sim.column('households_autoown')
def car_sufficiency(households, persons):
    return households.auto_ownership - persons.household_id.value_counts()


@sim.column('households')
def work_tour_auto_time_savings(households):
    # TODO fix this variable from auto ownership model
    return pd.Series(0, households.index)
