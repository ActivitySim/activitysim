import urbansim.sim.simulation as sim
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


@sim.injectable()
def nonmotskm_omx():
    return omx.openFile('data/nonmotskm.omx')


@sim.injectable()
def distance_matrix(nonmotskm_omx):
    return skim.Skim(nonmotskm_omx['DIST'], offset=-1)


@sim.injectable()
def auto_ownership_spec():
    f = os.path.join('configs', "auto_ownership_coeffs.csv")
    return asim.read_model_spec(f).head(4*26)


@sim.injectable()
def workplace_location_spec():
    f = os.path.join('configs', "workplace_location.csv")
    return asim.read_model_spec(f).head(7)


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
                                distance_matrix):

    choosers = sim.merge_tables(persons.name, tables=[persons, households])
    alternatives = zones.to_frame()

    skims = {
        "distance": distance_matrix
    }

    choices, model_design = \
        asim.simple_simulate(choosers,
                             alternatives,
                             workplace_location_spec,
                             skims,
                             skim_join_name="TAZ",
                             mult_by_alt_col=False,
                             sample_size=50)

    print "Describe of hoices:\n", choices.describe()
    sim.add_column("persons", "workplace_taz", choices)

    return model_design