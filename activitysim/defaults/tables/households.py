import pandas as pd
import numpy as np
import urbansim.sim.simulation as sim
import urbansim.utils.misc as usim_misc
from activitysim import activitysim as asim


@sim.table(cache=True)
def households(store, settings):

    if "households_sample_size" in settings:
        return asim.random_rows(store["households"],
                                settings["households_sample_size"])

    return store["households"]


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


@sim.column("households")
def auto_ownership(households):
    # FIXME this is really because we ask for ALL columns in the persons data
    # FIXME frame - urbansim actually only asks for the columns that are used by
    # FIXME the model specs in play at that time
    return pd.Series(0, households.index)


@sim.column('households')
def hhsize(households):
    return households.PERSONS


@sim.column('households')
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


@sim.column('households')
def car_sufficiency(households, persons):
    return households.auto_ownership - persons.household_id.value_counts()


@sim.column('households')
def work_tour_auto_time_savings(households):
    # TODO fix this variable from auto ownership model
    return pd.Series(0, households.index)


# this is an idiom to grab the person of the specified type and check to see if
# there is 1 or more of that kind of person in each household
def presence_of(ptype, persons, households, at_home=False):
    if at_home:
        # if at_home, they need to be of given type AND at home
        s = persons.household_id[(persons.ptype_cat == ptype) &
                                 (persons.cdap_activity == "H")]
    else:
        s = persons.household_id[persons.ptype_cat == ptype]

    return (s.value_counts() > 0).reindex(households.index).fillna(False)


# FIXME this is in non-mandatory tour generation - and should really be from
# FIXME the perspective of the current chooser - which it's not right now
@sim.column('households')
def has_non_worker(persons, households):
    return presence_of("nonwork", persons, households)


# FIXME this is in non-mandatory tour generation - and should really be from
# FIXME the perspective of the current chooser - which it's not right now
@sim.column('households')
def has_retiree(persons, households):
    return presence_of("retired", persons, households)


# FIXME this is in non-mandatory tour generation - and should really be from
# FIXME the perspective of the current chooser - which it's not right now
@sim.column('households')
def has_preschool_kid(persons, households):
    return presence_of("preschool", persons, households)


# FIXME this is in non-mandatory tour generation - and should really be from
# FIXME the perspective of the current chooser - which it's not right now
@sim.column('households')
def has_preschool_kid_at_home(persons, households):
    return presence_of("preschool", persons, households, at_home=True)


# FIXME this is in non-mandatory tour generation - and should really be from
# FIXME the perspective of the current chooser - which it's not right now
@sim.column('households')
def has_driving_kid(persons, households):
    return presence_of("driving", persons, households)


# FIXME this is in non-mandatory tour generation - and should really be from
# FIXME the perspective of the current chooser - which it's not right now
@sim.column('households')
def has_school_kid(persons, households):
    return presence_of("school", persons, households)


# FIXME this is in non-mandatory tour generation - and should really be from
# FIXME the perspective of the current chooser - which it's not right now
@sim.column('households')
def has_school_kid_at_home(persons, households):
    return presence_of("school", persons, households, at_home=True)


# FIXME this is in non-mandatory tour generation - and should really be from
# FIXME the perspective of the current chooser - which it's not right now
@sim.column('households')
def has_full_time(persons, households):
    return presence_of("full", persons, households)


# FIXME this is in non-mandatory tour generation - and should really be from
# FIXME the perspective of the current chooser - which it's not right now
@sim.column('households')
def has_part_time(persons, households):
    return presence_of("part", persons, households)


# FIXME this is in non-mandatory tour generation - and should really be from
# FIXME the perspective of the current chooser - which it's not right now
@sim.column('households')
def has_university(persons, households):
    return presence_of("university", persons, households)
