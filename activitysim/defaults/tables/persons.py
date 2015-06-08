import numpy as np
import orca
import pandas as pd

from activitysim.activitysim import other_than
from activitysim.util import reindex


# this caches things so you don't have to read in the file from disk again
@orca.table(cache=True)
def persons_internal(store, settings, households):
    df = store["persons"]

    if "households_sample_size" in settings:
        # keep all persons in the sampled households
        df = df[df.household_id.isin(households.index)]

    return df


# this caches all the columns that are computed on the persons table
@orca.table(cache=True)
def persons(persons_internal):
    return persons_internal.to_frame()


# this is the placeholder for all the columns to update after the
# school location choice model
@orca.table()
def persons_school(persons):
    return pd.DataFrame(index=persons.index)


# this is the placeholder for all the columns to update after the
# workplace location choice model
@orca.table()
def persons_workplace(persons):
    return pd.DataFrame(index=persons.index)


# this is the placeholder for all the columns to update after the
# non-mandatory tour frequency model
@orca.table()
def persons_nmtf(persons):
    return pd.DataFrame(index=persons.index)


# another common merge for persons
@orca.table()
def persons_merged(persons, households, land_use, accessibility):
    return orca.merge_tables(persons.name, tables=[
        persons, households, land_use, accessibility])


@orca.column("persons")
def age_16_to_19(persons):
    return persons.to_frame(["age"]).eval("16 <= age <= 19")


@orca.column("persons")
def age_16_p(persons):
    return persons.to_frame(["age"]).eval("16 <= age")


@orca.column("persons")
def adult(persons):
    return persons.to_frame(["age"]).eval("18 <= age")


@orca.column("persons", cache=True)
def cdap_activity(set_random_seed, persons):
    # return a default until it gets filled in by the model
    return pd.Series(np.random.randint(3, size=len(persons)),
                     index=persons.index).map({0: 'Mandatory',
                                               1: 'NonMandatory',
                                               2: 'Home'})


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


@orca.column("persons_nmtf")
def num_escort_tours(persons, non_mandatory_tours):
    nmt = non_mandatory_tours.to_frame()
    return nmt[nmt.tour_type == "escort"].groupby("person_id").size()\
        .reindex(persons.index).fillna(0)


@orca.column("persons_nmtf")
def num_non_escort_tours(persons, non_mandatory_tours):
    nmt = non_mandatory_tours.to_frame()
    return nmt[nmt.tour_type != "escort"].groupby("person_id").size()\
        .reindex(persons.index).fillna(0)


# count the number of mandatory tours for each person
@orca.column("persons")
def num_mand(persons):
    if "mandatory_tour_frequency" not in persons.columns:
        return pd.Series(0, index=persons.index)

    s = persons.mandatory_tour_frequency.map({
        "work1": 1,
        "work2": 2,
        "school1": 1,
        "school2": 2,
        "work_and_school": 2
    }, na_action='ignore')
    return s.fillna(0)


@orca.column("persons")
def work_and_school_and_worker(persons):
    if "mandatory_tour_frequency" not in persons.columns:
        return pd.Series(0, index=persons.index)

    s = (persons.mandatory_tour_frequency == "work_and_school").\
        reindex(persons.index).fillna(False)

    return s & persons.is_worker


@orca.column("persons")
def work_and_school_and_student(persons):
    if "mandatory_tour_frequency" not in persons.columns:
        return pd.Series(0, index=persons.index)

    s = (persons.mandatory_tour_frequency == "work_and_school").\
        reindex(persons.index).fillna(False)

    return s & persons.is_student


# FIXME now totally sure what this is but it's used in non mandatory tour
# FIXME generation and probably has to do with remaining unscheduled time
@orca.column('persons')
def max_window(persons):
    return pd.Series(0, persons.index)


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
def under16_not_at_school(persons):
    return (persons.ptype_cat.isin(["school", "preschool"]) &
            persons.cdap_activity.isin(["N", "H"]))


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
def workplace_taz(persons):
    return pd.Series(1, persons.index)


@orca.column("persons")
def home_taz(households, persons):
    return reindex(households.home_taz, persons.household_id)


# this use the distance skims to compute the raw distance to work from home
@orca.column("persons_workplace")
def school_taz(persons):
    return pd.Series(1, persons.index)


# this use the distance skims to compute the raw distance to work from home
@orca.column("persons")
def distance_to_work(persons, distance_skim):
    return pd.Series(distance_skim.get(persons.home_taz,
                                       persons.workplace_taz),
                     index=persons.index)


# same deal but to school
@orca.column("persons_school")
def distance_to_school(persons, distance_skim):
    return pd.Series(distance_skim.get(persons.home_taz,
                                       persons.school_taz),
                     index=persons.index)


# similar but this adds the am peak travel time to the pm peak travel time in
# the opposite direction (by car)
@orca.column("persons_workplace")
def roundtrip_auto_time_to_work(persons, sovam_skim, sovpm_skim):
    return pd.Series(sovam_skim.get(persons.home_taz,
                                    persons.workplace_taz) +
                     sovpm_skim.get(persons.workplace_taz,
                                    persons.home_taz),
                     index=persons.index)


# this adds the am peak travel time to the md peak travel time in
# the opposite direction (by car), assuming students leave school earlier
@orca.column("persons_school")
def roundtrip_auto_time_to_school(persons, sovam_skim, sovmd_skim):
    return pd.Series(sovam_skim.get(persons.home_taz,
                                    persons.school_taz) +
                     sovmd_skim.get(persons.school_taz,
                                    persons.home_taz),
                     index=persons.index)


@orca.column('persons_workplace')
def workplace_in_cbd(persons, land_use, settings):
    s = reindex(land_use.area_type, persons.workplace_taz)
    return s < settings['cbd_threshold']


# this is an idiom to grab the person of the specified type and check to see if
# there is 1 or more of that kind of person in each household
def presence_of(ptype, persons, at_home=False):
    if at_home:
        # if at_home, they need to be of given type AND at home
        bools = (persons.ptype_cat == ptype) & (persons.cdap_activity == "H")
    else:
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
def has_preschool_kid_at_home(persons):
    return presence_of("preschool", persons, at_home=True)


@orca.column('persons')
def has_driving_kid(persons):
    return presence_of("driving", persons)


@orca.column('persons')
def has_school_kid(persons):
    return presence_of("school", persons)


@orca.column('persons')
def has_school_kid_at_home(persons):
    return presence_of("school", persons, at_home=True)


@orca.column('persons')
def has_full_time(persons):
    return presence_of("full", persons)


@orca.column('persons')
def has_part_time(persons):
    return presence_of("part", persons)


@orca.column('persons')
def has_university(persons):
    return presence_of("university", persons)
