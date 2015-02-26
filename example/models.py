import urbansim.sim.simulation as sim
import urbansim.utils.misc as usim_misc
import os
from activitysim import activitysim as asim
import openmatrix as omx
from activitysim import skim
import numpy as np
import pandas as pd


# this is the max number of cars allowable in the auto ownership model
MAX_NUM_CARS = 5


"""
This part of this file is currently creating small tables to serve as
alternatives in the various models
"""


@sim.table()
def auto_alts():
    return asim.identity_matrix(["cars%d" % i for i in range(MAX_NUM_CARS)])


@sim.table()
def mandatory_tour_frequency_alts():
    return asim.identity_matrix(["work1", "work2", "school1", "school2",
                                 "work_and_school"])


# these are the alternatives for the workplace choice
@sim.table()
def zones():
    # I grant this is a weird idiom but it helps to name the index
    return pd.DataFrame({"TAZ": np.arange(1454)+1}).set_index("TAZ")


@sim.table()
def non_mandatory_tour_frequency_alts():
    f = os.path.join("configs",
                     "non_mandatory_tour_frequency_alternatives.csv")
    df = pd.read_csv(f)
    df["tot_tours"] = df.sum(axis=1)
    return df


"""
Read in the omx files and create the skim objects
"""


@sim.injectable()
def nonmotskm_omx():
    return omx.openFile(os.path.join('data', "nonmotskm.omx"))


@sim.injectable()
def distance_skim(nonmotskm_omx):
    return skim.Skim(nonmotskm_omx['DIST'], offset=-1)


@sim.injectable()
def sovam_skim(nonmotskm_omx):
    # FIXME use the right omx file
    return skim.Skim(nonmotskm_omx['DIST'], offset=-1)


@sim.injectable()
def sovmd_skim(nonmotskm_omx):
    # FIXME use the right omx file
    return skim.Skim(nonmotskm_omx['DIST'], offset=-1)


@sim.injectable()
def sovpm_skim(nonmotskm_omx):
    # FIXME use the right omx file
    return skim.Skim(nonmotskm_omx['DIST'], offset=-1)


"""
Read in the spec files and reformat as necessary
"""


@sim.injectable()
def auto_ownership_spec():
    f = os.path.join('configs', "auto_ownership.csv")
    # FIXME should read in all variables and comment out ones not used
    return asim.read_model_spec(f).head(4*26)


@sim.injectable()
def workplace_location_spec():
    f = os.path.join('configs', "workplace_location.csv")
    # FIXME should read in all variables and comment out ones not used
    return asim.read_model_spec(f).head(15)


@sim.injectable()
def mandatory_tour_frequency_spec():
    f = os.path.join('configs', "mandatory_tour_frequency.csv")
    return asim.read_model_spec(f)


@sim.injectable()
def non_mandatory_tour_frequency_spec():
    f = os.path.join('configs', "non_mandatory_tour_frequency_ftw.csv")
    return asim.read_model_spec(f)


@sim.table()
def workplace_size_spec():
    f = os.path.join('configs', 'workplace_location_size_terms.csv')
    return pd.read_csv(f)


"""
This is a special submodel for the workplace location choice
"""


@sim.table()
def workplace_size_terms(land_use, workplace_size_spec):
    """
    This method takes the land use data and multiplies various columns of the
    land use data by coefficients from the workplace_size_spec table in order
    to yield a size term (a linear combination of land use variables) with
    specified coefficients for different segments (like low, med, and high
    income)
    """
    land_use = land_use.to_frame()
    df = workplace_size_spec.to_frame().query("purpose == 'work'")
    df = df.drop("purpose", axis=1).set_index("segment")
    new_df = {}
    for index, row in df.iterrows():
        missing = row[~row.index.isin(land_use.columns)]
        if len(missing) > 0:
            print "WARNING: missing columns in land use\n", missing.index
        row = row[row.index.isin(land_use.columns)]
        sparse = land_use[list(row.index)]
        new_df["size_"+index] = np.dot(sparse.as_matrix(), row.values)
    new_df = pd.DataFrame(new_df, index=land_use.index)
    return new_df


"""
Auto ownership is a standard model which predicts how many cars a household
with given characteristics owns
"""


@sim.model()
def auto_ownership_simulate(households,
                            auto_alts,
                            auto_ownership_spec,
                            land_use,
                            accessibility):

    choosers = sim.merge_tables(households.name, tables=[households,
                                                         land_use,
                                                         accessibility])
    alternatives = auto_alts.to_frame()

    choices, model_design = \
        asim.simple_simulate(choosers, alternatives, auto_ownership_spec,
                             mult_by_alt_col=True)

    # map these back to integers
    choices = choices.map(dict([("cars%d" % i, i)
                                for i in range(MAX_NUM_CARS)]))

    print "Choices:\n", choices.value_counts()
    sim.add_column("households", "auto_ownership", choices)

    return model_design


"""
The workplace location model predicts the zones in which various people will
work.  Interestingly there's not really any supply side to this model - we
assume there are workplaces for the people to work.
"""


# FIXME there are three school models that go along with this one which have
# FIXME not been implemented yet
@sim.model()
def workplace_location_simulate(persons,
                                households,
                                zones,
                                workplace_location_spec,
                                distance_skim,
                                workplace_size_terms):

    choosers = sim.merge_tables(persons.name, tables=[persons, households])
    alternatives = zones.to_frame().join(workplace_size_terms.to_frame())

    skims = {
        "distance": distance_skim
    }

    choices, model_design = \
        asim.simple_simulate(choosers,
                             alternatives,
                             workplace_location_spec,
                             skims,
                             skim_join_name="TAZ",
                             mult_by_alt_col=False,
                             sample_size=50)

    print "Describe of choices:\n", choices.describe()
    sim.add_column("persons", "workplace_taz", choices)

    return model_design


"""
This model predicts the frequency of making mandatory trips (see the
alternatives above) - these trips include work and school in some combination.
"""


@sim.model()
def mandatory_tour_frequency(persons,
                             households,
                             land_use,
                             mandatory_tour_frequency_alts,
                             mandatory_tour_frequency_spec):

    choosers = sim.merge_tables(persons.name, tables=[persons,
                                                      households,
                                                      land_use])

    # filter based on results of CDAP
    choosers = choosers[choosers.cdap_activity == 'M']
    print "%d persons run for mandatory tour model" % len(choosers)

    choices, model_design = \
        asim.simple_simulate(choosers,
                             mandatory_tour_frequency_alts.to_frame(),
                             mandatory_tour_frequency_spec,
                             mult_by_alt_col=True)

    print "Choices:\n", choices.value_counts()
    sim.add_column("persons", "mandatory_tour_frequency", choices)

    return model_design


"""
This model predicts the frequency of making non-mandatory trips (
alternatives for this model come from a seaparate csv file which is
configured by the user) - these trips include escort, shopping, othmaint,
othdiscr, eatout, and social trips in various combination.
"""


# FIXME there are 8 different person types, all of which have a different
# FIXME specification for this model - at this time, only the full time
# FIXME worker person type is being implemented
@sim.model()
def non_mandatory_tour_frequency(persons,
                                 households,
                                 land_use,
                                 accessibility,
                                 non_mandatory_tour_frequency_alts,
                                 non_mandatory_tour_frequency_spec):

    print non_mandatory_tour_frequency_spec.tail()

    choosers = sim.merge_tables(persons.name, tables=[persons,
                                                      households,
                                                      land_use,
                                                      accessibility])

    # filter based on results of CDAP
    choosers = choosers[choosers.cdap_activity.isin(['M', 'N'])]
    print "%d persons run for non-mandatory tour model" % len(choosers)

    choices, model_design = \
        asim.simple_simulate(choosers,
                             non_mandatory_tour_frequency_alts.to_frame(),
                             non_mandatory_tour_frequency_spec,
                             mult_by_alt_col=False)

    print "Choices:\n", choices.value_counts()
    # this is adding the INDEX of the alternative that is chosen - when
    # we use the results of this choice we will need both these indexes AND
    # the alternatives themselves
    sim.add_column("persons", "non_mandatory_tour_frequency", choices)

    return model_design


"""
This section contains computed columns on each table.
"""

"""
for the land use table
"""


@sim.column("land_use")
def total_households(land_use):
    return land_use.local.TOTHH


@sim.column("land_use")
def total_employment(land_use):
    return land_use.local.TOTEMP


@sim.column("land_use")
def total_acres(land_use):
    return land_use.local.TOTACRE


@sim.column("land_use")
def county_id(land_use):
    return land_use.local.COUNTY


"""
for households
"""


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
def no_cars(households):
    return (households.auto_ownership == 0)


@sim.column('households')
def home_is_urban(households, land_use, settings):
    s = usim_misc.reindex(land_use.area_type, households.home_taz)
    return s < settings['urban_threshold']


@sim.column('households')
def car_sufficiency(households, persons):
    return households.auto_ownership - persons.household_id.value_counts()


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


"""
for the persons table
"""


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
