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
    return pd.read_csv(f)


@sim.column("non_mandatory_tour_frequency_alts")
def tot_tours(non_mandatory_tour_frequency_alts):
    # this assumes that the alt dataframe is only counts of trip types
    return non_mandatory_tour_frequency_alts.local.sum(axis=1)


@sim.table()
def tour_departure_and_duration_alts():
    # right now this file just contains the start and end hour
    f = os.path.join("configs",
                     "tour_departure_and_duration_alternatives.csv")
    return pd.read_csv(f)


# used to have duration in the actual alternative csv file,
# but this is probably better as a computed column
@sim.column("tour_departure_and_duration_alts")
def duration(tour_departure_and_duration_alts):
    return tour_departure_and_duration_alts.end - \
        tour_departure_and_duration_alts.start


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
    f = os.path.join('configs', "non_mandatory_tour_frequency.csv")
    # this is a spec in already stacked format
    # it also has multiple segments in different columns in the spec
    return asim.read_model_spec(f, stack=False)


@sim.table()
def destination_choice_size_terms():
    f = os.path.join('configs', 'destination_choice_size_terms.csv')
    return pd.read_csv(f)


@sim.table()
def destination_choice_spec():
    f = os.path.join('configs', 'destination_choice_alternatives_sample.csv')
    return asim.read_model_spec(f, stack=False).head(5)


@sim.table()
def tour_departure_and_duration_spec():
    f = os.path.join('configs', 'tour_departure_and_duration.csv')
    return asim.read_model_spec(f, stack=False)


"""
This is a special submodel for the workplace location choice
"""


@sim.table()
def workplace_size_terms(land_use, destination_choice_size_terms):
    """
    This method takes the land use data and multiplies various columns of the
    land use data by coefficients from the workplace_size_spec table in order
    to yield a size term (a linear combination of land use variables) with
    specified coefficients for different segments (like low, med, and high
    income)
    """
    land_use = land_use.to_frame()
    df = destination_choice_size_terms.to_frame().query("purpose == 'work'")
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


@sim.model()
def non_mandatory_tour_frequency(persons,
                                 households,
                                 land_use,
                                 accessibility,
                                 non_mandatory_tour_frequency_alts,
                                 non_mandatory_tour_frequency_spec):

    choosers = sim.merge_tables(persons.name, tables=[persons,
                                                      households,
                                                      land_use,
                                                      accessibility])

    # filter based on results of CDAP
    choosers = choosers[choosers.cdap_activity.isin(['M', 'N'])]
    print "%d persons run for non-mandatory tour model" % len(choosers)

    choices_list = []
    # segment by person type and pick the right spec for each person type
    for name, segment in choosers.groupby('ptype_cat'):

        print "Running segment '%s' of size %d" % (name, len(segment))

        choices, _ = \
            asim.simple_simulate(segment,
                                 non_mandatory_tour_frequency_alts.to_frame(),
                                 # notice that we pick the column for the
                                 # segment for each segment we run
                                 non_mandatory_tour_frequency_spec[name],
                                 mult_by_alt_col=False)
        choices_list.append(choices)

    choices = pd.concat(choices_list)

    print "Choices:\n", choices.value_counts()
    # this is adding the INDEX of the alternative that is chosen - when
    # we use the results of this choice we will need both these indexes AND
    # the alternatives themselves
    sim.add_column("persons", "non_mandatory_tour_frequency", choices)


"""
We have now generated mandatory and non-mandatory tours, but they are
attributes of the person table - this function creates a "tours" table which
has one row per tour that has been generated (and the person id it is
associated with)
"""


@sim.table()
def non_mandatory_tours(persons,
                        non_mandatory_tour_frequency_alts):

    # get the actual alternatives for each person - have to go back to the
    # non_mandatory_tour_frequency_alts dataframe to get this - the choice
    # above just stored the index values for the chosen alts
    tours = non_mandatory_tour_frequency_alts.local.\
        loc[persons.non_mandatory_tour_frequency]

    # assign person ids to the index
    tours.index = persons.index[~persons.non_mandatory_tour_frequency.isnull()]

    # reformat with the columns given below
    tours = tours.stack().reset_index()
    tours.columns = ["person_id", "trip_type", "num_tours"]

    # now do a repeat and a take, so if you have two trips of given type you
    # now have two rows, and zero trips yields zero rows
    tours = tours.take(np.repeat(tours.index.values, tours.num_tours.values))

    # make index unique and drop num_tours since we don't need it anymore
    tours = tours.reset_index(drop=True).drop("num_tours", axis=1)

    """
    Pretty basic at this point - trip table looks like this so far
              person_id trip_type
    0          4419    escort
    1          4419    escort
    2          4419  othmaint
    3          4419    eatout
    4          4419    social
    5         10001    escort
    6         10001    escort
    """
    return tours


sim.broadcast('persons', 'non_mandatory_tours',
              cast_index=True, onto_on='person_id')


"""
This does the same as the above but for mandatory tours.  Ending format is
the same as in the comment above except trip types are "work" and "school"
"""

@sim.table()
def mandatory_tours(persons):

    persons = persons.to_frame(columns=["mandatory_tour_frequency",
                                        "is_worker"])
    persons = persons[~persons.mandatory_tour_frequency.isnull()]

    tours = []
    # this is probably easier to do in non-vectorized fashion (at least for now)
    for key, row in persons.iterrows():

        mtour = row.mandatory_tour_frequency
        is_worker = row.is_worker

        # 1 work trip
        if mtour == "work1":
            tours += [(key, "work", 1)]
        # 2 work trips
        elif mtour == "work2":
            tours += [(key, "work", 1), (key, "work", 2)]
        # 1 school trip
        elif mtour == "school1":
            tours += [(key, "school", 1)]
        # 2 school trips
        elif mtour == "school2":
            tours += [(key, "school", 1), (key, "school", 2)]
        # 1 work and 1 school trip
        elif mtour == "work_and_school":
            if is_worker:
                # is worker, work trip goes first
                tours += [(key, "work", 1), (key, "school", 2)]
            else:
                # is student, work trip goes second
                tours += [(key, "school", 1), (key, "work", 2)]
        else:
            assert 0

    return pd.DataFrame(tours, columns=["person_id", "trip_type", "tour_num"])


sim.broadcast('persons', 'mandatory_tours',
              cast_index=True, onto_on='person_id')


"""
Given the tour generation from the above, each tour needs to have a
destination, so in this case tours are the choosers (with the associated
person that's making the tour)
"""


@sim.model()
def destination_choice(non_mandatory_tours,
                       persons,
                       households,
                       land_use,
                       zones,
                       distance_skim,
                       destination_choice_spec):

    tours = non_mandatory_tours

    # FIXME these models don't have size terms at the moment

    # FIXME these models don't use stratified sampling

    # FIXME is the distance to the second trip based on the choice of the
    # FIXME first trip - ouch!

    # choosers are tours - in a sense tours are choosing their destination
    choosers = sim.merge_tables(tours.name, tables=[tours,
                                                    persons,
                                                    households])

    skims = {
        "distance": distance_skim
    }

    choices_list = []
    # segment by trip type and pick the right spec for each person type
    for name, segment in choosers.groupby('trip_type'):

        # FIXME - there are two options here escort with kids and without
        if name == "escort":
            continue

        print "Running segment '%s' of size %d" % (name, len(segment))

        choices, _ = \
            asim.simple_simulate(choosers,
                                 zones.to_frame(),
                                 destination_choice_spec[name],
                                 skims,
                                 skim_join_name="TAZ",
                                 mult_by_alt_col=False,
                                 sample_size=50)

        choices_list.append(choices)

    choices = pd.concat(choices_list)

    print "Choices:\n", choices.describe()
    # every trip now has a destination which is the index from the
    # alternatives table - in this case it's the destination taz
    sim.add_column("non_mandatory_tours", "destination", choices)


"""
This model predicts the departure time and duration of each activity
"""


@sim.model()
def mandatory_tour_departure_and_duration(mandatory_tours,
                                          persons,
                                          households,
                                          land_use,
                                          tour_departure_and_duration_alts,
                                          tour_departure_and_duration_spec):

    choosers = sim.merge_tables(mandatory_tours.name, tables=[mandatory_tours,
                                                              persons,
                                                              households,
                                                              land_use])

    print "Running %d mandatory tour scheduling choices" % len(choosers)

    # assert there's only a first or second mandatory tour - that's a basic
    # assumption of this model formulation right now
    assert choosers.tour_num.isin([1, 2]).value_counts()[True] == len(choosers)

    first_tours = choosers[choosers.tour_num == 1]
    second_tours = choosers[choosers.tour_num == 2]

    spec = tour_departure_and_duration_spec.work.head(27)
    alts = tour_departure_and_duration_alts.to_frame()

    print choosers.mandatory_tour_frequency.value_counts()
    print spec

    # this is a bit odd to python - we can't run through in for loops for
    # performance reasons - we first have to do a pass for the first tours and
    # then for the second tours - this is mainly because the second tours are
    # dependent on the first tours' scheduling

    print "Running %d mandatory first tour choices" % len(first_tours)

    alts["end_of_previous_tour"] = -1

    # FIXME - a note to remember that this also needs the mode choice logsum
    alts["mode_choice_logsum"] = 0

    first_choices, _ = \
        asim.simple_simulate(first_tours, alts, spec, mult_by_alt_col=False)

    print "Running %d mandatory second tour choices" % len(second_tours)

    # FIXME need to set end_of_previous_tour to the ends computed above
    second_choices, _ = \
        asim.simple_simulate(second_tours, alts, spec, mult_by_alt_col=False)

    choices = pd.concat([first_choices, second_choices])

    # as with non-mandatory tour generation, this stores the INDEX of
    # the alternative in the tour_departure and_duration_alts dataframe -
    # to actually use it we'll have ot go back and grab the start and end times
    print "Choices:\n", choices.describe()

    sim.add_column("persons", "mandatory_tour_departure_and_duration", choices)


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
def home_is_rural(households, land_use, settings):
    s = usim_misc.reindex(land_use.area_type, households.home_taz)
    return s > settings['rural_threshold']


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


# FIXME this is in non-mandatory tour generation - and should really be from
# FIXME the perspective of the current chooser - which it's not right now
@sim.column('households')
def has_university(persons, households):
    return presence_of("university", persons, households)


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

