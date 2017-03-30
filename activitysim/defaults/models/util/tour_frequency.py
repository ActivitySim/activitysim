# ActivitySim
# See full license in LICENSE.txt.

import itertools
import numpy as np
import pandas as pd


def canonical_tours():
    """
        create labels for every the possible tour by combining tour_type/tour_num.

    Returns
    -------
        list of canonical tour labels in alphabetical order
    """
    # the problem is we don't know what the possible tour_types and their max tour_nums are

    # FIXME - should get this from alts table
    # alts = orca.get_table('non_mandatory_tour_frequency_alts').local
    # non_mandatory_tour_flavors = {c : alts[alts].max() for c in alts.columns.names}
    non_mandatory_tour_flavors = {'escort': 2, 'shopping': 1, 'othmaint': 1, 'othdiscr': 1,
                                  'eatout': 1, 'social': 1}

    # this logic is hardwired in process_mandatory_tours()
    mandatory_tour_flavors = {'work': 2, 'school': 2}

    tour_flavors = dict(non_mandatory_tour_flavors, **mandatory_tour_flavors)

    sub_channels = [tour_type + str(tour_num)
                    for tour_type, max_count in tour_flavors.iteritems()
                    for tour_num in range(1, max_count + 1)]

    sub_channels.sort()
    return sub_channels


def set_tour_index(tours):
    """

    Parameters
    ----------
    tours : DataFrame
        Tours dataframe to reindex.
        The new index values are stable based on the person_id, tour_type, and tour_num.
        The existing index is ignored and replaced.

        Having a stable (predictable) index value
        It also simplifies attaching random number streams to tours that are stable
        (even across simulations)

    Returns
    -------

    """

    possible_tours = canonical_tours()
    possible_tours_count = len(possible_tours)

    # concat tour_type + tour_num
    tours['tour_id'] = tours.tour_type + tours.tour_num.map(str)

    # map recognized strings to ints
    tours.tour_id = tours.tour_id.replace(to_replace=possible_tours,
                                          value=range(possible_tours_count))
    # convert to numeric - shouldn't be any NaNs - this will raise error if there are
    tours.tour_id = pd.to_numeric(tours.tour_id, errors='coerce').astype(int)

    tours.tour_id = (tours.person_id * possible_tours_count) + tours.tour_id

    if len(tours.tour_id) > len(tours.tour_id.unique()):
        print "\ntours.tour_id not unique\n", tours

    tours.set_index('tour_id', inplace=True, verify_integrity=True)


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
        work_taz = row.workplace_taz
        school_taz = row.school_taz

        # this logic came from the CTRAMP model - I copied it as best as I
        # could from the previous code - basically we need to know which
        # tours are the first tour and which are subsequent, and work /
        # school depends on the status of the person (is_worker variable)

        # 1 work trip
        if mtour == "work1":
            tours += [(key, "work", 1, work_taz)]
        # 2 work trips
        elif mtour == "work2":
            tours += [(key, "work", 1, work_taz), (key, "work", 2, work_taz)]
        # 1 school trip
        elif mtour == "school1":
            tours += [(key, "school", 1, school_taz)]
        # 2 school trips
        elif mtour == "school2":
            tours += [(key, "school", 1, school_taz), (key, "school", 2, school_taz)]
        # 1 work and 1 school trip
        elif mtour == "work_and_school":
            if is_worker:
                # is worker, work trip goes first
                tours += [(key, "work", 1, work_taz), (key, "school", 2, school_taz)]
            else:
                # is student, work trip goes second
                tours += [(key, "school", 1, school_taz), (key, "work", 2, work_taz)]
        else:
            assert 0

    """
    Pretty basic at this point - trip table looks like this so far
           person_id tour_type tour_num destination
    tour_id
    0          4419    work     1       <work_taz>
    1          4419    school   2       <school_taz>
    4          4650    school   1       <school_taz>
    5         10001    school   1       <school_taz>
    6         10001    work     2       <work_taz>
    """

    df = pd.DataFrame(tours, columns=["person_id", "tour_type", "tour_num", "destination"])

    set_tour_index(df)

    return df


def process_non_mandatory_tours(non_mandatory_tour_frequency,
                                non_mandatory_tour_frequency_alts):
    """
    This method processes the non_mandatory_tour_frequency column that comes
    out of the model of the same name and turns into a DataFrame that
    represents the non mandatory tours that were generated

    Parameters
    ----------
    non_mandatory_tour_frequency: Series
        A series which has person id as the index and the chosen alternative
        index as the value
    non_mandatory_tour_frequency_alts: DataFrame
        A DataFrame which has as a unique index which relates to the values
        in the series above typically includes columns which are named for trip
        purposes with values which are counts for that trip purpose.  Example
        trip purposes include escort, shopping, othmaint, othdiscr, eatout,
        social, etc.  A row would be an alternative which might be to take
        one shopping trip and zero trips of other purposes, etc.

    Returns
    -------
    tours : DataFrame
        An example of a tours DataFrame is supplied as a comment in the
        source code - it has an index which is a unique tour identifier,
        a person_id column, and a tour type column which comes from the
        column names of the alternatives DataFrame supplied above.
    """

    nmtf = non_mandatory_tour_frequency

    # get the actual alternatives for each person - have to go back to the
    # non_mandatory_tour_frequency_alts dataframe to get this - the choice
    # above just stored the index values for the chosen alts
    tours = non_mandatory_tour_frequency_alts.loc[nmtf]

    # assign person ids to the index
    tours.index = nmtf.index

    # reformat with the columns given below
    tours = tours.stack().reset_index()
    tours.columns = ["person_id", "tour_type", "tour_count"]

    # map non-zero tour_counts to a list of ranges [1,2,1] -> [[0], [0, 1], [0]]
    tour_nums = map(range, tours.tour_count[tours.tour_count > 0].values)
    # flatten (more baroque but faster than np.hstack)
    tour_nums = np.array(list(itertools.chain.from_iterable(tour_nums))) + 1

    # now do a repeat and a take, so if you have two trips of given type you
    # now have two rows, and zero trips yields zero rows
    tours = tours.take(np.repeat(tours.index.values, tours.tour_count.values))

    tours['tour_num'] = tour_nums

    # make index unique
    set_tour_index(tours)

    """
    Pretty basic at this point - trip table looks like this so far
           person_id tour_type tour_num
    tour_id
    0          4419    escort   1
    1          4419    escort   2
    2          4419  othmaint   1
    3          4419    eatout   1
    4          4419    social   1
    5         10001    escort   1
    6         10001    escort   2
    """
    return tours
