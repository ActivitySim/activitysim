import numpy as np
import pandas as pd

from activitysim.core.util import other_than, reindex
from activitysim.core import inject
from constants import *


@inject.column("persons")
def age_16_to_19(persons):
    c = persons.to_frame(["age"]).eval("16 <= age <= 19")
    return c


@inject.column("persons")
def age_16_p(persons):
    return persons.to_frame(["age"]).eval("16 <= age")


@inject.column("persons")
def adult(persons):
    return persons.to_frame(["age"]).eval("18 <= age")


# FIXME - these are my "placeholder" for joint trip generation
# number of joint shopping tours
@inject.column("persons")
def num_shop_j(persons):
    return pd.Series(0, persons.index)


# FIXME - these are my "placeholder" for joint trip generation
# number of joint shopping tours
@inject.column("persons")
def num_main_j(persons):
    return pd.Series(0, persons.index)


# FIXME - these are my "placeholder" for joint trip generation
# number of joint shopping tours
@inject.column("persons")
def num_eat_j(persons):
    return pd.Series(0, persons.index)


# FIXME - these are my "placeholder" for joint trip generation
# number of joint shopping tours
@inject.column("persons")
def num_visi_j(persons):
    return pd.Series(0, persons.index)


# FIXME - these are my "placeholder" for joint trip generation
# number of joint shopping tours
@inject.column("persons")
def num_disc_j(persons):
    return pd.Series(0, persons.index)


@inject.column("persons")
def num_joint_tours(persons):
    return persons.num_shop_j + persons.num_main_j + persons.num_eat_j +\
        persons.num_visi_j + persons.num_disc_j


@inject.column("persons")
def male(persons):
    return persons.sex == 1


@inject.column("persons")
def female(persons):
    return persons.sex == 2


# this is an idiom to grab the person of the specified type and check to see if
# there is 1 or more of that kind of person in each household other than the person themself
def presence_of(ptype, persons):
    bools = persons.ptype_cat == ptype
    return other_than(persons.household_id, bools)


@inject.column('persons')
def has_non_worker(persons):
    return presence_of("nonwork", persons)


@inject.column('persons')
def has_retiree(persons):
    return presence_of("retired", persons)


@inject.column('persons')
def has_preschool_kid(persons):
    return presence_of("preschool", persons)


@inject.column('persons')
def has_driving_kid(persons):
    return presence_of("driving", persons)


@inject.column('persons')
def has_school_kid(persons):
    return presence_of("school", persons)


@inject.column('persons')
def has_full_time(persons):
    return presence_of("full", persons)


@inject.column('persons')
def has_part_time(persons):
    return presence_of("part", persons)


@inject.column('persons')
def has_university(persons):
    return presence_of("university", persons)


# convert person type categories to string descriptors
@inject.column("persons")
def ptype_cat(persons, settings):
    return persons.ptype.map(settings["person_type_map"])


# borrowing these definitions from the original code
@inject.column("persons")
def student_is_employed(persons):
    return (persons.ptype_cat.isin(['university', 'driving']) &
            persons.pemploy.isin([PEMPLOY_FULL, PEMPLOY_PART]))


@inject.column("persons")
def nonstudent_to_school(persons):
    return (persons.ptype_cat.isin(['full', 'part', 'nonwork', 'retired']) &
            persons.pstudent.isin([PSTUDENT_GRADE_OR_HIGH, PSTUDENT_UNIVERSITY]))


@inject.column("persons")
def is_worker(persons):
    return persons.pemploy.isin([PEMPLOY_FULL, PEMPLOY_PART])


@inject.column("persons")
def is_student(persons):
    return persons.pstudent.isin([PSTUDENT_GRADE_OR_HIGH, PSTUDENT_UNIVERSITY])


@inject.column("persons")
def is_gradeschool(persons, settings):
    return (persons.pstudent == PSTUDENT_GRADE_OR_HIGH) & \
           (persons.age <= settings['grade_school_max_age'])


@inject.column("persons")
def is_highschool(persons, settings):
    return (persons.pstudent == PSTUDENT_GRADE_OR_HIGH) & \
           (persons.age > settings['grade_school_max_age'])


@inject.column("persons")
def is_university(persons):
    return persons.pstudent == PSTUDENT_UNIVERSITY


@inject.column("persons")
def home_taz(households, persons):
    return reindex(households.home_taz, persons.household_id)


# FIXME now totally sure what this is but it's used in non mandatory tour
# FIXME generation and probably has to do with remaining unscheduled time
@inject.column('persons')
def max_window(persons):
    return pd.Series(0, persons.index)
