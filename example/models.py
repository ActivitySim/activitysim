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


"""
Read in the omx files and create the skim objects
"""


@sim.injectable()
def nonmotskm_omx():
    return omx.openFile('data/nonmotskm.omx')


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

    choices, model_design = \
        asim.simple_simulate(choosers,
                             mandatory_tour_frequency_alts.to_frame(),
                             mandatory_tour_frequency_spec,
                             mult_by_alt_col=True)

    print "Choices:\n", choices.value_counts()
    sim.add_column("persons", "mandatory_tour_frequency", choices)

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


"""
for the persons table
"""


# FIXME - this is my "placeholder" for the CDAP model ;)
@sim.column("persons")
def cdap_activity(persons):
    return pd.Series(np.random.randint(3, size=len(persons)),
                     index=persons.index).map({0: 'M', 1: 'N', 2: 'H'})


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
