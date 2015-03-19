import os
import pandas as pd
import urbansim.sim.simulation as sim
from activitysim import activitysim as asim

"""
This model predicts the frequency of making mandatory trips (see the
alternatives above) - these trips include work and school in some combination.
"""


@sim.injectable()
def mandatory_tour_frequency_spec(configs_dir):
    f = os.path.join(configs_dir, 'configs', "mandatory_tour_frequency.csv")
    return asim.read_model_spec(f).fillna(0)


@sim.model()
def mandatory_tour_frequency(set_random_seed,
                             persons_merged,
                             mandatory_tour_frequency_spec):

    choosers = persons_merged.to_frame()
    # filter based on results of CDAP
    choosers = choosers[choosers.cdap_activity == 'M']
    print "%d persons run for mandatory tour model" % len(choosers)

    choices = asim.simple_simulate(choosers, mandatory_tour_frequency_spec)

    # convert indexes to alternative names
    choices = pd.Series(
        mandatory_tour_frequency_spec.columns[choices.values],
        index=choices.index)

    print "Choices:\n", choices.value_counts()
    sim.add_column("persons", "mandatory_tour_frequency", choices)


"""
This reprocesses the choice of index of the mandatory tour frequency
alternatives into an actual dataframe of tours.  Ending format is
the same as got non_mandatory_tours except trip types are "work" and "school"
"""


# TODO this needs a simple input / output unit test
@sim.table()
def mandatory_tours(persons):

    persons = persons.to_frame(columns=["mandatory_tour_frequency",
                                        "is_worker"])
    persons = persons[~persons.mandatory_tour_frequency.isnull()]

    tours = []
    # this is probably easier to do in non-vectorized fashion like this
    for key, row in persons.iterrows():

        mtour = row.mandatory_tour_frequency
        is_worker = row.is_worker

        # this logic came from the CTRAMP model - I copied it as best as I
        # could from the previous code - basically we need to know which
        # tours are the first tour and which are subsequent, and work /
        # school depends on the status of the person (is_worker variable)

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

    """
    Pretty basic at this point - trip table looks like this so far
           person_id tour_type tour_num
    0          4419    work     1
    1          4419    school   2
    4          4650    school   1
    5         10001    school   1
    6         10001    work     2
    """

    return pd.DataFrame(tours, columns=["person_id", "tour_type", "tour_num"])


# broadcast mandatory_tours on to persons using the person_id foreign key
sim.broadcast('persons', 'mandatory_tours',
              cast_index=True, onto_on='person_id')
sim.broadcast('persons_merged', 'mandatory_tours',
              cast_index=True, onto_on='person_id')
