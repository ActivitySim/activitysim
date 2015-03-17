import pandas as pd
import numpy as np
import urbansim.sim.simulation as sim
import urbansim.utils.misc as usim_misc


@sim.table(cache=True)
def persons(store, settings, households):
    df = store["persons"]

    if "households_sample_size" in settings:
        # keep all persons in the sampled households
        df = df[df.household_id.isin(households.index)]

    return df


# another common merge for persons
@sim.table()
def persons_merged(persons, households, land_use, accessibility):
    return sim.merge_tables(persons.name, tables=[persons,
                                                  households,
                                                  land_use,
                                                  accessibility])


@sim.column("persons")
def age_16_to_19(persons):
    return persons.to_frame(["age"]).eval("16 <= age <= 19")


@sim.column("persons")
def age_16_p(persons):
    return persons.to_frame(["age"]).eval("16 <= age")


# FIXME - this is my "placeholder" for the CDAP model ;)
@sim.column("persons")
def cdap_activity(persons):
    return pd.Series(np.random.randint(3, size=len(persons)),
                     index=persons.index).map({0: 'M', 1: 'N', 2: 'H'})


# FIXME - these are my "placeholder" for joint trip generation
# number of joint shopping tours
@sim.column("persons")
def num_shop_j(persons):
    return pd.Series(0, persons.index)


# FIXME - these are my "placeholder" for joint trip generation
# number of joint shopping tours
@sim.column("persons")
def num_main_j(persons):
    return pd.Series(0, persons.index)


# FIXME - these are my "placeholder" for joint trip generation
# number of joint shopping tours
@sim.column("persons")
def num_eat_j(persons):
    return pd.Series(0, persons.index)


# FIXME - these are my "placeholder" for joint trip generation
# number of joint shopping tours
@sim.column("persons")
def num_visi_j(persons):
    return pd.Series(0, persons.index)


# FIXME - these are my "placeholder" for joint trip generation
# number of joint shopping tours
@sim.column("persons")
def num_disc_j(persons):
    return pd.Series(0, persons.index)


@sim.column("persons")
def num_joint_tours(persons):
    return persons.num_shop_j + persons.num_main_j + persons.num_eat_j +\
        persons.num_visi_j + persons.num_disc_j


@sim.column("persons")
def male(persons):
    return persons.sex == 1


@sim.column("persons")
def female(persons):
    return persons.sex == 1


# count the number of mandatory tours for each person
@sim.column("persons")
def num_mand(persons):
    # FIXME this is really because we ask for ALL columns in the persons data
    # FIXME frame - urbansim actually only asks for the columns that are used by
    # FIXME the model specs in play at that time
    if "mandatory_tour_frequency" not in persons.columns:
        return pd.Series(0, index=persons.index)

    s = persons.mandatory_tour_frequency.map({
        "work1": 1,
        "work2": 2,
        "school1": 1,
        "school2": 2,
        "work_and_school": 2
    })
    return s


# FIXME now totally sure what this is but it's used in non mandatory tour
# FIXME generation and probably has to do with remaining unscheduled time
@sim.column('persons')
def max_window(persons):
    return pd.Series(0, persons.index)


# convert employment categories to string descriptors
@sim.column("persons")
def employed_cat(persons, settings):
    return persons.pemploy.map(settings["employment_map"])


# convert student categories to string descriptors
@sim.column("persons")
def student_cat(persons, settings):
    return persons.pstudent.map(settings["student_map"])


# convert person type categories to string descriptors
@sim.column("persons")
def ptype_cat(persons, settings):
    return persons.ptype.map(settings["person_type_map"])


# borrowing these definitions from the original code
@sim.column("persons")
def student_is_employed(persons):
    return (persons.ptype_cat.isin(['university', 'driving']) &
            persons.employed_cat.isin(['full', 'part']))


@sim.column("persons")
def nonstudent_to_school(persons):
    return (persons.ptype_cat.isin(['full', 'part', 'nonwork', 'retired']) &
            persons.student_cat.isin(['high', 'college']))


@sim.column("persons")
def under16_not_at_school(persons):
    return (persons.ptype_cat.isin(["school", "preschool"]) &
            persons.cdap_activity.isin(["N", "H"]))


@sim.column("persons")
def is_worker(persons):
    return persons.employed_cat.isin(['full', 'part'])


@sim.column("persons")
def is_student(persons):
    return persons.student_cat.isin(['high', 'college'])


@sim.column("persons")
def workplace_taz(persons):
    # FIXME this is really because we ask for ALL columns in the persons data
    # FIXME frame - urbansim actually only asks for the columns that are used by
    # FIXME the model specs in play at that time
    return pd.Series(1, persons.index)


@sim.column("persons")
def home_taz(households, persons):
    return usim_misc.reindex(households.home_taz,
                             persons.household_id)


@sim.column("persons")
def school_taz(persons):
    # FIXME need to fix this after getting school lcm working
    return persons.workplace_taz


# this use the distance skims to compute the raw distance to work from home
@sim.column("persons")
def distance_to_work(persons, distance_skim):
    return pd.Series(distance_skim.get(persons.home_taz,
                                       persons.workplace_taz),
                     index=persons.index)


# same deal but to school
@sim.column("persons")
def distance_to_school(persons, distance_skim):
    return pd.Series(distance_skim.get(persons.home_taz,
                                       persons.school_taz),
                     index=persons.index)


# similar but this adds the am peak travel time to the pm peak travel time in
# the opposite direction (by car)
@sim.column("persons")
def roundtrip_auto_time_to_work(persons, sovam_skim, sovpm_skim):
    return pd.Series(sovam_skim.get(persons.home_taz,
                                    persons.workplace_taz) +
                     sovpm_skim.get(persons.workplace_taz,
                                    persons.home_taz),
                     index=persons.index)


# this adds the am peak travel time to the md peak travel time in
# the opposite direction (by car), assuming students leave school earlier
@sim.column("persons")
def roundtrip_auto_time_to_school(persons, sovam_skim, sovmd_skim):
    return pd.Series(sovam_skim.get(persons.home_taz,
                                    persons.school_taz) +
                     sovmd_skim.get(persons.school_taz,
                                    persons.home_taz),
                     index=persons.index)


@sim.column('persons')
def workplace_in_cbd(persons, land_use, settings):
    s = usim_misc.reindex(land_use.area_type, persons.workplace_taz)
    return s < settings['cbd_threshold']
