import os
import pandas as pd
import urbansim.sim.simulation as sim
from activitysim import activitysim as asim
from non_mandatory_scheduling import vectorize_tour_schedules

"""
This model predicts the departure time and duration of each activity for
mandatory tours
"""


@sim.table()
def tdd_alts(configs_dir):
    # right now this file just contains the start and end hour
    f = os.path.join(configs_dir, "configs",
                     "tour_departure_and_duration_alternatives.csv")
    return pd.read_csv(f)


# used to have duration in the actual alternative csv file,
# but this is probably better as a computed column like this
@sim.column("tdd_alts")
def duration(tdd_alts):
    return tdd_alts.end - tdd_alts.start


@sim.table()
def tdd_mandatory_spec(configs_dir):
    f = os.path.join(configs_dir, 'configs',
                     'tour_departure_and_duration_mandatory.csv')
    return asim.read_model_spec(f).fillna(0)


@sim.model()
def mandatory_scheduling(set_random_seed,
                         mandatory_tours_merged,
                         tdd_alts,
                         tdd_mandatory_spec):

    tours = mandatory_tours_merged.to_frame()

    print "Running %d mandatory tour scheduling choices" % len(tours)

    spec = tdd_mandatory_spec.to_frame()
    alts = tdd_alts.to_frame()

    choices = vectorize_tour_schedules(tours, alts, spec)

    print "Choices:\n", choices.describe()

    sim.add_column("mandatory_tours",
                   "tour_departure_and_duration",
                   choices)
