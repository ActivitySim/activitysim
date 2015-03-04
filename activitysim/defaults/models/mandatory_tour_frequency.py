import os
import pandas as pd
import urbansim.sim.simulation as sim
from activitysim import activitysim as asim

"""
This model predicts the frequency of making mandatory trips (see the
alternatives above) - these trips include work and school in some combination.
"""


@sim.table()
def mandatory_tour_frequency_alts():
    return asim.identity_matrix(["work1", "work2", "school1", "school2",
                                 "work_and_school"])


@sim.injectable()
def mandatory_tour_frequency_spec():
    f = os.path.join('configs', "mandatory_tour_frequency.csv")
    return asim.read_model_spec(f)


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

    return pd.DataFrame(tours, columns=["person_id", "tour_type", "tour_num"])


sim.broadcast('persons', 'mandatory_tours',
              cast_index=True, onto_on='person_id')
