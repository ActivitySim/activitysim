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
def tdd_alts():
    # right now this file just contains the start and end hour
    f = os.path.join("configs",
                     "tour_departure_and_duration_alternatives.csv")
    return pd.read_csv(f)


# used to have duration in the actual alternative csv file,
# but this is probably better as a computed column like this
@sim.column("tdd_alts")
def duration(tdd_alts):
    return tdd_alts.end - tdd_alts.start


@sim.table()
def tdd_mandatory_spec():
    f = os.path.join('configs', 'tour_departure_and_duration_mandatory.csv')
    return asim.read_model_spec(f, stack=False)


@sim.model()
def tour_departure_and_duration_mandatory(mandatory_tours_merged,
                                          tdd_alts,
                                          tdd_mandatory_spec):

    tours = mandatory_tours_merged

    print "Running %d mandatory tour scheduling choices" % len(tours)

    spec = tdd_mandatory_spec.to_frame().head(27)
    alts = tdd_alts.to_frame()

    # FIXME the "windowing" variables are not currently implemented

    choices = vectorize_tour_schedules(tours, alts, spec)

    print "Choices:\n", choices.describe()

    sim.add_column("mandatory_tours", "mandatory_tdd", choices)
