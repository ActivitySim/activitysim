import urbansim.sim.simulation as sim
import urbansim.utils.misc as usim_misc
import os
from activitysim import activitysim as asim
import openmatrix as omx
from activitysim import skim
import numpy as np
import pandas as pd


@sim.table()
def auto_alts():
    return asim.identity_matrix(["cars%d" % i for i in range(5)])


@sim.table()
def zones():
    # I grant this is a weird idiom but it helps to name the index
    return pd.DataFrame({"TAZ": np.arange(1454)+1}).set_index("TAZ")


@sim.table()
def mandatory_tour_frequency_alts():
    return asim.identity_matrix(["work1", "work2", "school1", "school2",
                                 "work_and_school"])


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


@sim.injectable()
def auto_ownership_spec():
    f = os.path.join('configs', "auto_ownership.csv")
    return asim.read_model_spec(f).head(4*26)


@sim.injectable()
def workplace_location_spec():
    f = os.path.join('configs', "workplace_location.csv")
    return asim.read_model_spec(f).head(15)


@sim.injectable()
def mandatory_tour_frequency_spec():
    f = os.path.join('configs', "mandatory_tour_frequency.csv")
    return asim.read_model_spec(f).head(87)


@sim.table()
def workplace_size_spec():
    f = os.path.join('configs', 'workplace_location_size_terms.csv')
    return pd.read_csv(f)


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

    print "Choices:\n", choices.value_counts()
    sim.add_column("households", "auto_ownership", choices)

    return model_design


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


@sim.model()
def mandatory_tour_frequency(persons,
                             mandatory_tour_frequency_alts,
                             mandatory_tour_frequency_spec):

    choosers = persons.to_frame()
    print mandatory_tour_frequency_spec

    choices, model_design = \
        asim.simple_simulate(choosers,
                             mandatory_tour_frequency_alts.to_frame(),
                             mandatory_tour_frequency_spec,
                             mult_by_alt_col=True)

    print "Choices:\n", choices.value_counts()
    sim.add_column("persons", "mandatory_tour_frequency", choices)

    return model_design


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


# just a rename / alias
@sim.column("households")
def home_taz(households):
    return households.TAZ


# for now, this really is just a dictionary lookup
@sim.column("persons")
def employed_cat(persons, settings):
    return persons.pemploy.map(settings["employment_map"])


@sim.column("persons")
def student_cat(persons, settings):
    return persons.pstudent.map(settings["student_map"])


@sim.column("persons")
def student_is_employed(persons):
    ec = persons.employed_cat
    sc = persons.student_cat
    return ((ec == 'full') | (ec == 'part')) & \
           ((sc == 'high') | (sc == 'college'))


@sim.column("persons")
def home_taz(households, persons):
    return usim_misc.reindex(households.home_taz,
                             persons.household_id)


@sim.column("persons")
def distance_to_work(persons, distance_skim):
    return pd.Series(distance_skim.get(persons.home_taz,
                                       persons.workplace_taz),
                     index=persons.index)


@sim.column("persons")
def workplace_taz(persons):
    return pd.Series(1, persons.index)


@sim.column("persons")
def school_taz(persons):
    # FIXME need to fix this after getting school lcm working
    return persons.workplace_taz


@sim.column("persons")
def distance_to_school(persons, distance_skim):
    return pd.Series(distance_skim.get(persons.home_taz,
                                       persons.school_taz),
                     index=persons.index)


@sim.column("persons")
def roundtrip_auto_time_to_work(persons, sovam_skim, sovpm_skim):
    return pd.Series(sovam_skim.get(persons.home_taz,
                                    persons.workplace_taz) +
                     sovpm_skim.get(persons.workplace_taz,
                                    persons.home_taz),
                     index=persons.index)


@sim.column("persons")
def roundtrip_auto_time_to_school(persons, sovam_skim, sovmd_skim):
    return pd.Series(sovam_skim.get(persons.home_taz,
                                    persons.school_taz) +
                     sovmd_skim.get(persons.school_taz,
                                    persons.home_taz),
                     index=persons.index)