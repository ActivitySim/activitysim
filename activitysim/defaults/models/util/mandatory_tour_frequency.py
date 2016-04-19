# ActivitySim
# See full license in LICENSE.txt.

import pandas as pd


def process_mandatory_tours(persons):
    """
    This method processes the mandatory_tour_frequency column that comes out of
    the model of the same name and turns into a DataFrame that represents the
    mandatory tours that were generated

    Parameters
    ----------
    persons : DataFrame
        Persons is a DataFrame which has a column call
        mandatory_tour_frequency (which came out of the mandatory tour
        frequency model) and a column is_worker which indicates the person's
        worker status.  The only valid values of the mandatory_tour_frequency
        column to take are "work1", "work2", "school1", "school2" and
        "work_and_school"

    Returns
    -------
    tours : DataFrame
        An example of a tours DataFrame is supplied as a comment in the
        source code - it has an index which is a tour identifier, a person_id
        column, a tour_type column which is "work" or "school" and a tour_num
        column which is set to 1 or 2 depending whether it is the first or
        second mandatory tour made by the person.  The logic for whether the
        work or school tour comes first given a "work_and_school" choice
        depends on the is_worker column and was copied from the original
        implementation.
    """

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
