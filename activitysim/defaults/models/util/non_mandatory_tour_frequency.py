# ActivitySim
# Copyright (C) 2014-2015 Synthicity, LLC
# Copyright (C) 2015 Autodesk
# See full license in LICENSE.txt.

import numpy as np


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
    tours.columns = ["person_id", "tour_type", "num_tours"]

    # now do a repeat and a take, so if you have two trips of given type you
    # now have two rows, and zero trips yields zero rows
    tours = tours.take(np.repeat(tours.index.values, tours.num_tours.values))

    # make index unique and drop num_tours since we don't need it anymore
    tours = tours.reset_index(drop=True).drop("num_tours", axis=1)

    """
    Pretty basic at this point - trip table looks like this so far
           person_id tour_type
    0          4419    escort
    1          4419    escort
    2          4419  othmaint
    3          4419    eatout
    4          4419    social
    5         10001    escort
    6         10001    escort
    """
    return tours
