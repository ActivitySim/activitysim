import os
import pandas as pd
import urbansim.sim.simulation as sim
from activitysim import activitysim as asim
from .util.vectorize_tour_scheduling import vectorize_tour_scheduling


"""
This model predicts the departure time and duration of each activity for
non-mandatory tours
"""


@sim.table()
def tdd_non_mandatory_spec(configs_dir):
    f = os.path.join(configs_dir, 'configs',
                     'tour_departure_and_duration_nonmandatory.csv')
    return asim.read_model_spec(f).fillna(0)


@sim.model()
def non_mandatory_scheduling(set_random_seed,
                             non_mandatory_tours_merged,
                             tdd_alts,
                             tdd_non_mandatory_spec):

    tours = non_mandatory_tours_merged.to_frame()

    print "Running %d non-mandatory tour scheduling choices" % len(tours)

    # FIXME we're not even halfway down the specfile
    spec = tdd_non_mandatory_spec.to_frame().head(4)[['Coefficient']]
    alts = tdd_alts.to_frame()

    choices = vectorize_tour_scheduling(tours, alts, spec)

    print "Choices:\n", choices.describe()

    sim.add_column("non_mandatory_tours",
                   "tour_departure_and_duration",
                   choices)
