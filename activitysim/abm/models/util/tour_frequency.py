# ActivitySim
# See full license in LICENSE.txt.

import itertools
import numpy as np
import pandas as pd


TOUR_CATEGORIES = ['mandatory', 'non_mandatory', 'subtour']


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

    # FIXME - this logic is hardwired in process_mandatory_tours()
    mandatory_tour_flavors = {'work': 2, 'school': 2}

    # FIXME - should get this from alts table
    atwork_subtour_flavors = {'eat': 1, 'business': 2, 'maint': 1}

    tour_flavors = dict(non_mandatory_tour_flavors)
    tour_flavors.update(mandatory_tour_flavors)
    tour_flavors.update(atwork_subtour_flavors)

    sub_channels = [tour_type + str(tour_num)
                    for tour_type, max_count in tour_flavors.iteritems()
                    for tour_num in range(1, max_count + 1)]

    sub_channels.sort()
    return sub_channels


def set_tour_index(tours, tour_num_col):
    """

    Parameters
    ----------
    tours : DataFrame
        Tours dataframe to reindex.
        The new index values are stable based on the person_id, tour_type, and tour_num.
        The existing index is ignored and replaced.

        This gives us a stable (predictable) tour_id
        It also simplifies attaching random number streams to tours that are stable
        (even across simulations)
    """

    possible_tours = canonical_tours()
    possible_tours_count = len(possible_tours)

    assert tour_num_col in tours.columns

    tours['tour_id'] = tours.tour_type + tours[tour_num_col] .map(str)

    # map recognized strings to ints
    tours.tour_id = tours.tour_id.replace(to_replace=possible_tours,
                                          value=range(possible_tours_count))
    # convert to numeric - shouldn't be any NaNs - this will raise error if there are
    tours.tour_id = pd.to_numeric(tours.tour_id, errors='coerce').astype(int)

    tours.tour_id = (tours.person_id * possible_tours_count) + tours.tour_id

    if len(tours.tour_id) > len(tours.tour_id.unique()):
        print "\ntours.tour_id not unique\n", tours

    tours.set_index('tour_id', inplace=True, verify_integrity=True)


def process_tours(tour_frequency, tour_frequency_alts, tour_category, parent_col='person_id'):
    """
    This method processes the tour_frequency column that comes
    out of the model of the same name and turns into a DataFrame that
    represents the tours that were generated

    Parameters
    ----------
    tour_frequency: Series
        A series which has person id as the index and the chosen alternative
        index as the value
    tour_frequency_alts: DataFrame
        A DataFrame which has as a unique index which relates to the values
        in the series above typically includes columns which are named for trip
        purposes with values which are counts for that trip purpose.  Example
        trip purposes include escort, shopping, othmaint, othdiscr, eatout,
        social, etc.  A row would be an alternative which might be to take
        one shopping trip and zero trips of other purposes, etc.
    tour_category : str
        one of 'mandatory', 'non_mandatory' or 'subtour'

    Returns
    -------
    tours : DataFrame
        An example of a tours DataFrame is supplied as a comment in the
        source code - it has an index which is a unique tour identifier,
        a person_id column, and a tour type column which comes from the
        column names of the alternatives DataFrame supplied above.
    """

    # get the actual alternatives for each person - have to go back to the
    # non_mandatory_tour_frequency_alts dataframe to get this - the choice
    # above just stored the index values for the chosen alts
    tours = tour_frequency_alts.loc[tour_frequency]

    # assign person ids to the index
    tours.index = tour_frequency.index

    """
               alt1       alt2     alt3
    PERID
    2588676       2         0         0
    2588677       1         1         0
    """

    # reformat with the columns given below
    tours = tours.stack().reset_index()
    tours.columns = [parent_col, "tour_type", "tour_type_count"]

    """
        <parent_col> tour_type  tour_type_count
    0     2588676    alt1           2
    1     2588676    alt2           0
    2     2588676    alt3           0
    3     2588676    alt1           1
    4     2588677    alt2           1
    5     2588677    alt3           0

    parent_col is the index from non_mandatory_tour_frequency
    tour_type is the column name from non_mandatory_tour_frequency_alts
    tour_type_count is the count value of the tour's chosen alt's tour_type from alts table
    """

    # now do a repeat and a take, so if you have two trips of given type you
    # now have two rows, and zero trips yields zero rows
    tours = tours.take(np.repeat(tours.index.values, tours.tour_type_count.values))

    grouped = tours.groupby([parent_col, 'tour_type'])
    tours['tour_type_num'] = grouped.cumcount() + 1
    tours['tour_type_count'] = tours['tour_type_num'] + grouped.cumcount(ascending=False)

    grouped = tours.groupby(parent_col)
    tours['tour_num'] = grouped.cumcount() + 1
    tours['tour_count'] = tours['tour_num'] + grouped.cumcount(ascending=False)

    """
        <parent_col> tour_type  tour_type_num  tour_type_count tour_num  tour_count
    0     2588676       alt1           1           2               1         2
    0     2588676       alt1           2           2               2         2
    0     2588676       alt1           1           1               1         2
    0     2588676       alt2           1           1               2         2
    """

    # set these here to ensure consistency across different tour categories
    assert tour_category in ['mandatory', 'non_mandatory', 'subtour']
    tours['mandatory'] = (tour_category == 'mandatory')
    tours['non_mandatory'] = (tour_category == 'non_mandatory')
    tours['tour_category'] = tour_category

    return tours


def process_mandatory_tours(persons, mandatory_tour_frequency_alts):
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
        depends on the is_worker column: work tours first for workers, second for non-workers
    """

    assert not persons.mandatory_tour_frequency.isnull().any()

    tours = process_tours(persons.mandatory_tour_frequency.dropna(),
                          mandatory_tour_frequency_alts,
                          tour_category='mandatory')

    tours_merged = pd.merge(tours[['person_id', 'tour_type']],
                            persons,
                            left_on='person_id', right_index=True)

    # by default work tours are first for work_and_school tours
    # swap tour_nums for non-workers so school tour is 1 and work is 2
    work_and_school_and_student = \
        (tours_merged.mandatory_tour_frequency == 'work_and_school') & ~tours_merged.is_worker

    tours.tour_num = tours.tour_num.where(~work_and_school_and_student, 3 - tours.tour_num)

    # work tours destination is workplace_taz, school tours destination is school_taz
    tours['destination'] = \
        tours_merged.workplace_taz.where((tours_merged.tour_type == 'work'),
                                         tours_merged.school_taz)

    # assign stable (predictable) tour_id
    set_tour_index(tours, 'tour_num')

    """
               person_id tour_type  tour_type_count  tour_type_num  tour_num  tour_count
    tour_id
    12413245      827549    school                2              1         2           2
    12413244      827549    school                2              2         1           2
    12413264      827550      work                2              1         2           2
    ...
               mandatory  non_mandatory tour_category  destination

                    True          False     mandatory          102
                    True          False     mandatory          102
                    True          False     mandatory            9
    """
    return tours


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
    tours = process_tours(non_mandatory_tour_frequency,
                          non_mandatory_tour_frequency_alts,
                          tour_category='non_mandatory')

    # assign stable (predictable) tour_id
    set_tour_index(tours, 'tour_type_num')

    """
               person_id tour_type  tour_type_count  tour_type_num  tour_num   tour_count
    tour_id
    17008286     1133885  shopping                1              1         1            3
    17008283     1133885  othmaint                1              1         2            3
    17008282     1133885  othdiscr                1              1         3            3
    ...
               mandatory  non_mandatory  tour_category

                   False           True  non_mandatory
                   False           True  non_mandatory
                   False           True  non_mandatory
    """

    return tours


def process_atwork_subtours(work_tours, atwork_subtour_frequency_alts):

    """
    This method processes the atwork_subtour_frequency column that comes
    out of the model of the same name and turns into a DataFrame that
    represents the subtours tours that were generated

    Parameters
    ----------
    work_tours: DataFrame
        A series which has parent work tour tour_id as the index and
        columns with person_id and atwork_subtour_frequency.
    atwork_subtour_frequency_alts: DataFrame
        A DataFrame which has as a unique index with atwork_subtour_frequency values
        and frequency counts for the subtours to be generated for that choice

    Returns
    -------
    tours : DataFrame
        An example of a tours DataFrame is supplied as a comment in the
        source code - it has an index which is a unique tour identifier,
        a person_id column, and a tour type column which comes from the
        column names of the alternatives DataFrame supplied above.
    """

    # print atwork_subtour_frequency_alts
    """
                  eat  business  maint
    alt
    no_subtours     0         0      0
    eat             1         0      0
    business1       0         1      0
    maint           0         0      1
    business2       0         2      0
    eat_business    1         1      0
    """

    parent_col = 'parent_tour_id'
    tours = process_tours(work_tours.atwork_subtour_frequency.dropna(),
                          atwork_subtour_frequency_alts,
                          tour_category='subtour',
                          parent_col=parent_col)

    # print tours
    """
               parent_tour_id tour_type  tour_type_count  tour_type_num  tour_num  tour_count
    tour_id
    77147972         77147984       eat                1              1         1           2
    77401056         77147984     maint                1              1         2           2
    80893007         80893019       eat                1              1         1           1

              mandatory  non_mandatory tour_category
                  False          False       subtour
                  False          False       subtour
                  False          False       subtour
    """

    # merge person_id from parent work_tours
    work_tours = work_tours[["person_id"]]
    tours = pd.merge(tours, work_tours, left_on=parent_col, right_index=True)

    # assign stable (predictable) tour_id
    set_tour_index(tours, 'tour_type_num')

    """
               person_id tour_type  tour_type_count  tour_type_num  tour_num  tour_count
    tour_id
    77147972     5143198       eat                1              1         1           2
    77401056     5143198     maint                1              1         2           2
    80893007     5392867       eat                1              1         1           1

              mandatory  non_mandatory tour_category   parent_tour_id

                  False          False       subtour         77147984
                  False          False       subtour         77147984
                  False          False       subtour         80893019
    """

    return tours
