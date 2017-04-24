import numpy as np
import pandas as pd

import orca

from activitysim.core.util import other_than, reindex


@orca.column("persons")
def age_16_to_19(persons):
    c = persons.to_frame(["age"]).eval("16 <= age <= 19")
    return c


@orca.column("persons")
def age_16_p(persons):
    return persons.to_frame(["age"]).eval("16 <= age")


@orca.column("persons")
def adult(persons):
    return persons.to_frame(["age"]).eval("18 <= age")


# FIXME - these are my "placeholder" for joint trip generation
# number of joint shopping tours
@orca.column("persons")
def num_shop_j(persons):
    return pd.Series(0, persons.index)


# FIXME - these are my "placeholder" for joint trip generation
# number of joint shopping tours
@orca.column("persons")
def num_main_j(persons):
    return pd.Series(0, persons.index)


# FIXME - these are my "placeholder" for joint trip generation
# number of joint shopping tours
@orca.column("persons")
def num_eat_j(persons):
    return pd.Series(0, persons.index)


# FIXME - these are my "placeholder" for joint trip generation
# number of joint shopping tours
@orca.column("persons")
def num_visi_j(persons):
    return pd.Series(0, persons.index)


# FIXME - these are my "placeholder" for joint trip generation
# number of joint shopping tours
@orca.column("persons")
def num_disc_j(persons):
    return pd.Series(0, persons.index)


@orca.column("persons")
def num_joint_tours(persons):
    return persons.num_shop_j + persons.num_main_j + persons.num_eat_j +\
        persons.num_visi_j + persons.num_disc_j


@orca.column("persons")
def male(persons):
    return persons.sex == 1


@orca.column("persons")
def female(persons):
    return persons.sex == 2


# this is an idiom to grab the person of the specified type and check to see if
# there is 1 or more of that kind of person in each household
def presence_of(ptype, persons):
    bools = persons.ptype_cat == ptype
    return other_than(persons.household_id, bools)


@orca.column('persons')
def has_non_worker(persons):
    return presence_of("nonwork", persons)


@orca.column('persons')
def has_retiree(persons):
    return presence_of("retired", persons)


@orca.column('persons')
def has_preschool_kid(persons):
    return presence_of("preschool", persons)


@orca.column('persons')
def has_driving_kid(persons):
    return presence_of("driving", persons)


@orca.column('persons')
def has_school_kid(persons):
    return presence_of("school", persons)


@orca.column('persons')
def has_full_time(persons):
    return presence_of("full", persons)


@orca.column('persons')
def has_part_time(persons):
    return presence_of("part", persons)


@orca.column('persons')
def has_university(persons):
    return presence_of("university", persons)


# convert employment categories to string descriptors
@orca.column("persons")
def employed_cat(persons, settings):
    return persons.pemploy.map(settings["employment_map"])


# convert student categories to string descriptors
@orca.column("persons")
def student_cat(persons, settings):
    return persons.pstudent.map(settings["student_map"])


# convert person type categories to string descriptors
@orca.column("persons")
def ptype_cat(persons, settings):
    return persons.ptype.map(settings["person_type_map"])


# borrowing these definitions from the original code
@orca.column("persons")
def student_is_employed(persons):
    return (persons.ptype_cat.isin(['university', 'driving']) &
            persons.employed_cat.isin(['full', 'part']))


@orca.column("persons")
def nonstudent_to_school(persons):
    return (persons.ptype_cat.isin(['full', 'part', 'nonwork', 'retired']) &
            persons.student_cat.isin(['grade_or_high', 'college']))


@orca.column("persons")
def is_worker(persons):
    return persons.employed_cat.isin(['full', 'part'])


@orca.column("persons")
def is_student(persons):
    return persons.student_cat.isin(['grade_or_high', 'college'])


@orca.column("persons")
def is_gradeschool(persons, settings):
    return (persons.student_cat == "grade_or_high") & \
           (persons.age <= settings['grade_school_max_age'])


@orca.column("persons")
def is_highschool(persons, settings):
    return (persons.student_cat == "grade_or_high") & \
           (persons.age > settings['grade_school_max_age'])


@orca.column("persons")
def is_university(persons):
    return persons.student_cat == "university"


@orca.column("persons")
def home_taz(households, persons):
    return reindex(households.home_taz, persons.household_id)


# FIXME now totally sure what this is but it's used in non mandatory tour
# FIXME generation and probably has to do with remaining unscheduled time
@orca.column('persons')
def max_window(persons):
    return pd.Series(0, persons.index)
