import os
import pandas as pd
import urbansim.sim.simulation as sim
from activitysim import activitysim as asim


"""
This model predicts the departure time and duration of each activity for
non-mandatory tours
"""


@sim.table()
def tdd_non_mandatory_spec():
    f = os.path.join('configs', 'tour_departure_and_duration_nonmandatory.csv')
    return asim.read_model_spec(f, stack=False)


# FIXME - move to activitysim
def vectorize_tour_schedules(tours, alts, spec):

    max_num_trips = tours.groupby('person_id').size().max()

    # because this is Python, we have to vectorize everything by doing the
    # "nth" trip for each person in a for loop (in other words, because each
    # trip is dependent on the time windows left by the previous decision) -
    # hopefully this will work out ok!

    choices = []

    for i in range(max_num_trips):

        nth_tours = tours.groupby('person_id').nth(i)

        print "Running %d non-mandatory #%d tour choices" % \
              (len(nth_tours), i+1)

        # FIXME below two lines are placeholders

        alts["mode_choice_logsum"] = 0
        nth_tours["end_of_previous_tour"] = -1

        nth_choices, _ = \
            asim.simple_simulate(nth_tours, alts, spec, mult_by_alt_col=False)

        choices.append(nth_choices)

    # return the concatenated choices
    return pd.concat(choices)


@sim.model()
def tour_departure_and_duration_non_mandatory(non_mandatory_tours_merged,
                                              tdd_alts,
                                              tdd_non_mandatory_spec):

    tours = non_mandatory_tours_merged

    print "Running %d non-mandatory tour scheduling choices" % len(tours)

    spec = tdd_non_mandatory_spec.Coefficient.head(4)
    alts = tdd_alts.to_frame()

    choices = vectorize_tour_schedules(tours, alts, spec)

    print "Choices:\n", choices.describe()

    sim.add_column("non_mandatory_tours", "non_mandatory_tdd", choices)
