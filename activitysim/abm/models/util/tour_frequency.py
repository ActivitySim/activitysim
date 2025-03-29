# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from activitysim.abm.models.util.canonical_ids import set_tour_index
from activitysim.core import workflow
from activitysim.core.configuration.base import PreprocessorSettings, PydanticReadable
from activitysim.core.configuration.logit import LogitComponentSettings
from activitysim.core.util import reindex

logger = logging.getLogger(__name__)


def create_tours(tour_counts, tour_category, parent_col="person_id"):
    """
    This method processes the tour_frequency column that comes
    out of the model of the same name and turns into a DataFrame that
    represents the tours that were generated

    Parameters
    ----------
    tour_counts: DataFrame
        table specifying how many tours of each type to create
        one row per person (or parent_tour for atwork subtours)
        one (int) column per tour_type, with number of tours to create
    tour_category : str
        one of 'mandatory', 'non_mandatory', 'atwork', or 'joint'

    Returns
    -------
    tours : pandas.DataFrame
        An example of a tours DataFrame is supplied as a comment in the
        source code - it has an index which is a unique tour identifier,
        a person_id column, and a tour type column which comes from the
        column names of the alternatives DataFrame supplied above.

        tours.tour_type       - tour type (e.g. school, work, shopping, eat)
        tours.tour_type_num   - if there are two 'school' type tours, they will be numbered 1 and 2
        tours.tour_type_count - number of tours of tour_type parent has (parent's max tour_type_num)
        tours.tour_num        - index of tour (of any type) for parent
        tours.tour_count      - number of tours of any type) for parent (parent's max tour_num)
        tours.tour_category   - one of 'mandatory', 'non_mandatory', 'atwork', or 'joint'
    """

    # FIXME - document requirement to ensure adjacent tour_type_nums in tour_num order

    """
               alt1       alt2     alt3
    <parent_col>
    2588676       2         0         0
    2588677       1         1         0
    """

    # reformat with the columns given below
    tours = tour_counts.stack().reset_index()
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

    grouped = tours.groupby([parent_col, "tour_type"])
    tours["tour_type_num"] = grouped.cumcount() + 1
    tours["tour_type_count"] = tours["tour_type_num"] + grouped.cumcount(
        ascending=False
    )

    grouped = tours.groupby(parent_col)
    tours["tour_num"] = grouped.cumcount() + 1
    tours["tour_count"] = tours["tour_num"] + grouped.cumcount(ascending=False)

    """
        <parent_col> tour_type  tour_type_num  tour_type_count tour_num  tour_count
    0     2588676       alt1           1           2               1         4
    0     2588676       alt1           2           2               2         4
    0     2588676       alt2           1           1               3         4
    0     2588676       alt3           1           1               4         4
    """

    # set these here to ensure consistency across different tour categories

    # do not enforce this here, other categories are possible
    # assert tour_category in ["mandatory", "non_mandatory", "atwork", "joint"]
    tours["tour_category"] = tour_category
    cat_tour_category = pd.api.types.CategoricalDtype(
        ["mandatory", "joint", "non_mandatory", "atwork"], ordered=False
    )
    tours["tour_category"] = tours["tour_category"].astype(cat_tour_category)

    # for joint tours, the correct number will be filled in after participation step
    tours["number_of_participants"] = 1

    # index is arbitrary but don't want any duplicates in index
    tours.reset_index(drop=True, inplace=True)

    # downcast
    tours["tour_count"] = tours["tour_count"].astype("int8")
    tours["tour_num"] = tours["tour_num"].astype("int8")
    tours["tour_type_num"] = tours["tour_type_num"].astype("int8")
    tours["tour_type_count"] = tours["tour_type_count"].astype("int8")
    tours["number_of_participants"] = tours["number_of_participants"].astype("int8")

    return tours


def process_tours(
    tour_frequency, tour_frequency_alts, tour_category, parent_col="person_id"
):
    """
    This method processes the tour_frequency column that comes
    out of the model of the same name and turns into a DataFrame that
    represents the tours that were generated

    Parameters
    ----------
    tour_frequency: Series
        A series which has <parent_col> as the index and the chosen alternative
        index as the value
    tour_frequency_alts: DataFrame
        A DataFrame which has as a unique index which relates to the values
        in the series above typically includes columns which are named for trip
        purposes with values which are counts for that trip purpose.  Example
        trip purposes include escort, shopping, othmaint, othdiscr, eatout,
        social, etc.  A row would be an alternative which might be to take
        one shopping trip and zero trips of other purposes, etc.
    tour_category : str
        one of 'mandatory', 'non_mandatory', 'atwork', or 'joint'
    parent_col: str
        the name of the index (parent_tour_id for atwork subtours, otherwise person_id)

    Returns
    -------
    tours : pandas.DataFrame
        An example of a tours DataFrame is supplied as a comment in the
        source code - it has an index which is a unique tour identifier,
        a person_id column, and a tour type column which comes from the
        column names of the alternatives DataFrame supplied above.

        tours.tour_type       - tour type (e.g. school, work, shopping, eat)
        tours.tour_type_num   - if there are two 'school' type tours, they will be numbered 1 and 2
        tours.tour_type_count - number of tours of tour_type parent has (parent's max tour_type_num)
        tours.tour_num        - index of tour (of any type) for parent
        tours.tour_count      - number of tours of any type) for parent (parent's max tour_num)
        tours.tour_category   - one of 'mandatory', 'non_mandatory', 'atwork', or 'joint'
    """

    # FIXME - document requirement to ensure adjacent tour_type_nums in tour_num order

    # get the actual alternatives for each person - have to go back to the
    # non_mandatory_tour_frequency_alts dataframe to get this - the choice
    # above just stored the index values for the chosen alts
    tour_counts = tour_frequency_alts.loc[tour_frequency]

    # assign person ids to the index
    tour_counts.index = tour_frequency.index

    """
               alt1       alt2     alt3
    <parent_col>
    2588676       2         0         0
    2588677       1         1         0
    """

    tours = create_tours(tour_counts, tour_category, parent_col)

    return tours


def process_mandatory_tours(
    state: workflow.State, persons, mandatory_tour_frequency_alts
):
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

    person_columns = [
        "mandatory_tour_frequency",
        "is_worker",
        "school_zone_id",
        "workplace_zone_id",
        "home_zone_id",
        "household_id",
    ]
    assert not persons.mandatory_tour_frequency.isnull().any()

    tours = process_tours(
        persons.mandatory_tour_frequency.dropna(),
        mandatory_tour_frequency_alts,
        tour_category="mandatory",
    )

    tours_merged = pd.merge(
        tours[["person_id", "tour_type"]],
        persons[person_columns],
        left_on="person_id",
        right_index=True,
    )

    # by default work tours are first for work_and_school tours
    # swap tour_nums for non-workers so school tour is 1 and work is 2
    work_and_school_and_student = (
        tours_merged.mandatory_tour_frequency == "work_and_school"
    ) & ~tours_merged.is_worker

    tours.tour_num = tours.tour_num.where(
        ~work_and_school_and_student, 3 - tours.tour_num
    )

    # work tours destination is workplace_zone_id, school tours destination is school_zone_id
    tours["destination"] = tours_merged.workplace_zone_id.where(
        (tours_merged.tour_type == "work"), tours_merged.school_zone_id
    )

    tours["origin"] = tours_merged.home_zone_id

    tours["household_id"] = tours_merged.household_id

    # assign stable (predictable) tour_id
    set_tour_index(state, tours)

    """
               person_id tour_type  tour_type_count  tour_type_num  tour_num  tour_count
    tour_id
    12413245      827549    school                2              1         1           2
    12413244      827549    school                2              2         2           2
    12413264      827550      work                1              1         1           2
    12413266      827550    school                1              1         2           2
    ...
               tour_category  destination   household_id
                   mandatory          102         103992
                   mandatory          102         103992
                   mandatory            9         103992
                   mandatory          102         103992
    """
    return tours


def process_non_mandatory_tours(state: workflow.State, persons, tour_counts):
    """
    This method processes the non_mandatory_tour_frequency column that comes
    out of the model of the same name and turns into a DataFrame that
    represents the non mandatory tours that were generated

    Parameters
    ----------
    persons: pandas.DataFrame
        persons table containing a non_mandatory_tour_frequency column
        which has the index of the chosen alternative as the value
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

    tours = create_tours(tour_counts, tour_category="non_mandatory")

    tours["household_id"] = reindex(persons.household_id, tours.person_id)
    tours["origin"] = reindex(persons.home_zone_id, tours.person_id)

    # assign stable (predictable) tour_id
    set_tour_index(state, tours)

    """
               person_id tour_type  tour_type_count  tour_type_num  tour_num   tour_count
    tour_id
    17008286     1133885  shopping                1              1         1            3
    17008283     1133885  othmaint                1              1         2            3
    17008282     1133885  othdiscr                1              1         3            3
    ...
               tour_category

               non_mandatory
               non_mandatory
               non_mandatory
    """

    return tours


def process_atwork_subtours(
    state: workflow.State,
    work_tours: pd.DataFrame,
    atwork_subtour_frequency_alts: pd.DataFrame,
):
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

    parent_col = "parent_tour_id"
    tours = process_tours(
        work_tours.atwork_subtour_frequency.dropna(),
        atwork_subtour_frequency_alts,
        tour_category="atwork",
        parent_col=parent_col,
    )

    # print tours
    """
               parent_tour_id tour_type  tour_type_count  tour_type_num  tour_num  tour_count
    tour_id
    77147972         77147984       eat                1              1         1           2
    77401056         77147984     maint                1              1         2           2
    80893007         80893019       eat                1              1         1           1

              tour_category
                     atwork
                     atwork
                     atwork
    """

    # merge fields from parent work_tours (note parent tour destination becomes subtour origin)
    work_tours = work_tours[["person_id", "household_id", "tour_num", "destination"]]
    work_tours.rename(
        columns={"tour_num": "parent_tour_num", "destination": "origin"}, inplace=True
    )
    tours = pd.merge(tours, work_tours, left_on=parent_col, right_index=True)

    # assign stable (predictable) tour_id
    set_tour_index(state, tours, parent_tour_num_col="parent_tour_num")

    """
               person_id tour_type  tour_type_count  tour_type_num  tour_num  tour_count
    tour_id
    77147972     5143198       eat                1              1         1           2
    77401056     5143198     maint                1              1         2           2
    80893007     5392867       eat                1              1         1           1

              tour_category   parent_tour_id  household_id
                     atwork         77147984       1625435
                     atwork         77147984       1625435
                     atwork         80893019       2135503
    """

    # don't need this once we have computed index
    del tours["parent_tour_num"]

    return tours


def process_joint_tours(
    state: workflow.State,
    joint_tour_frequency,
    joint_tour_frequency_alts,
    point_persons,
):
    """
    This method processes the joint_tour_frequency column that comes out of
    the model of the same name and turns into a DataFrame that represents the
    joint tours that were generated

    Parameters
    ----------
    joint_tour_frequency : pandas.Series
        household joint_tour_frequency (which came out of the joint tour frequency model)
        indexed by household_id
    joint_tour_frequency_alts: DataFrame
        A DataFrame which has as a unique index with joint_tour_frequency values
        and frequency counts for the tours to be generated for that choice
    point_persons : pandas DataFrame
        table with columns for (at least) person_ids and home_zone_id indexed by household_id

    Returns
    -------
    tours : DataFrame
        An example of a tours DataFrame is supplied as a comment in the
        source code - it has an index which is a tour identifier, a household_id
        column, a tour_type column and tour_type_num and tour_num columns
        which is set to 1 or 2 depending whether it is the first or second joint tour
        made by the household.
    """

    assert not joint_tour_frequency.isnull().any()

    tours = process_tours(
        joint_tour_frequency.dropna(),
        joint_tour_frequency_alts,
        tour_category="joint",
        parent_col="household_id",
    )

    assert not tours.index.duplicated().any()
    assert point_persons.index.name == "household_id"

    # - assign a temp point person to tour so we can create stable index
    tours["person_id"] = reindex(point_persons.person_id, tours.household_id)
    tours["origin"] = reindex(point_persons.home_zone_id, tours.household_id)

    # assign stable (predictable) tour_id
    set_tour_index(state, tours, is_joint=True)

    """
                   household_id tour_type  tour_type_count  tour_type_num  tour_num  tour_count
    tour_id
    3209530              320953      disc                1              1         1           2
    3209531              320953      disc                2              2         2           2
    23267026            2326702      shop                1              1         1           1
    17978574            1797857      main                1              1         1           1

                   tour_category  tour_category_id  person_id
    3209530                joint                 4     577234
    3209531                joint                 4     577234
    23267026               joint                 4    1742708
    17978574               joint                 4    5143198
    """
    return tours


def process_joint_tours_frequency_composition(
    state: workflow.State,
    joint_tour_frequency_composition,
    joint_tour_frequency_composition_alts,
    point_persons,
):
    """
    This method processes the joint_tour_frequency_composition column that comes out of
    the model of the same name and turns into a DataFrame that represents the
    joint tours that were generated

    Parameters
    ----------
    joint_tour_frequency_composition : pandas.Series
        household joint_tour_frequency_composition (which came out of the joint tour frequency composition model)
        indexed by household_id
    joint_tour_frequency_composition_alts: DataFrame
        A DataFrame which has as a unique index with joint_tour_frequency_composition values
        and frequency counts for the tours to be generated for that choice
    point_persons : pandas DataFrame
        table with columns for (at least) person_ids and home_zone_id indexed by household_id

    Returns
    -------
    tours : DataFrame
        An example of a tours DataFrame is supplied as a comment in the
        source code - it has an index which is a tour identifier, a household_id
        column, a tour_type column, composition column and tour_type_num and tour_num columns
        which is set to 1 or 2 depending whether it is the first or second joint tour
        made by the household.
    """

    assert not joint_tour_frequency_composition.isnull().any()

    tours = process_tours_frequency_composition(
        state,
        joint_tour_frequency_composition.dropna(),
        joint_tour_frequency_composition_alts,
        tour_category="joint",
        parent_col="household_id",
    )

    assert not tours.index.duplicated().any()
    assert point_persons.index.name == "household_id"

    # - assign a temp point person to tour so we can create stable index
    tours["person_id"] = reindex(point_persons.person_id, tours.household_id)
    tours["origin"] = reindex(point_persons.home_zone_id, tours.household_id)

    # assign stable (predictable) tour_id
    set_tour_index(state, tours, is_joint=True)

    """
                   household_id tour_type  tour_type_count  tour_type_num  tour_num  tour_count
    tour_id
    3209530              320953      disc                1              1         1           2
    3209531              320953      disc                2              2         2           2
    23267026            2326702      shop                1              1         1           1
    17978574            1797857      main                1              1         1           1

                   tour_category  tour_category_id  person_id
    3209530                joint                 4     577234
    3209531                joint                 4     577234
    23267026               joint                 4    1742708
    17978574               joint                 4    5143198
    """
    return tours


def process_tours_frequency_composition(
    state: workflow.State,
    joint_tour_frequency_composition,
    joint_tour_frequency_composition_alts,
    tour_category,
    parent_col="person_id",
):
    """
    This method processes the joint_tour_frequency_composition column that comes
    out of the model of the same name and turns into a DataFrame that
    represents the tours that were generated

    Parameters
    ----------
    joint_tour_frequency_composition: Series
        A series which has <parent_col> as the index and the chosen alternative
        index as the value
    joint_tour_frequency_composition_alts: DataFrame
        A DataFrame which has as a unique index which relates to the values
        in the series above typically includes columns which are named for trip
        purposes with values which are counts for that trip purpose, and tour composition.  Example
        trip purposes include escort, shopping, othmaint, othdiscr, eatout,
        social, etc. Tour composition includes adults, children, and mixed.
        A row would be an alternative which might be to take
        one shopping trip in a adult-only tour and zero trips of other purposes, etc.
    tour_category : str
        one of 'mandatory', 'non_mandatory', 'atwork', or 'joint'
    parent_col: str
        the name of the index (parent_tour_id for atwork subtours, otherwise person_id)

    Returns
    -------
    tours : pandas.DataFrame
        An example of a tours DataFrame is supplied as a comment in the
        source code - it has an index which is a unique tour identifier,
        a person_id column, and a tour type column which comes from the
        column names of the alternatives DataFrame supplied above.

        tours.tour_type       - tour type (e.g. school, work, shopping, eat)
        tours.composition     - tour composition (e.g. adults, children, mixed)
        tours.tour_type_num   - if there are two 'school' type tours, they will be numbered 1 and 2
        tours.tour_type_count - number of tours of tour_type parent has (parent's max tour_type_num)
        tours.tour_num        - index of tour (of any type) for parent
        tours.tour_count      - number of tours of any type) for parent (parent's max tour_num)
        tours.tour_category   - one of 'mandatory', 'non_mandatory', 'atwork', or 'joint'
    """

    # get the actual alternatives for each person - have to go back to the
    # non_mandatory_tour_frequency_alts dataframe to get this - the choice
    # above just stored the index values for the chosen alts
    tour_counts = joint_tour_frequency_composition_alts.loc[
        joint_tour_frequency_composition
    ]

    # assign person ids to the index
    tour_counts.index = joint_tour_frequency_composition.index

    """
               alt1       alt2     alt3
    <parent_col>
    2588676       2         0         0
    2588677       1         1         0
    """

    tours = create_joint_tours(state, tour_counts, tour_category, parent_col)

    return tours


class JointTourFreqCompContent(PydanticReadable):
    VALUE_MAP: dict[int, str] = {}
    COLUMNS: list[str] = []


class JointTourFreqCompAlts(PydanticReadable):
    PURPOSE: JointTourFreqCompContent = JointTourFreqCompContent()
    COMPOSITION: JointTourFreqCompContent = JointTourFreqCompContent()


class JointTourFreqCompSettings(LogitComponentSettings, extra="forbid"):
    """
    Settings for joint tour frequency and composition.
    """

    ALTS_TABLE_STRUCTURE: JointTourFreqCompAlts = JointTourFreqCompAlts()
    preprocessor: PreprocessorSettings | None = None
    ALTS_PREPROCESSOR: PreprocessorSettings | None = None


def create_joint_tours(
    state: workflow.State,
    tour_counts,
    tour_category,
    parent_col="person_id",
    model_settings: JointTourFreqCompSettings | None = None,
    model_settings_file_name: str = "joint_tour_frequency_composition.yaml",
):
    """
    This method processes the tour_frequency column that comes
    out of the model of the same name and turns into a DataFrame that
    represents the tours that were generated

    Parameters
    ----------
    tour_counts: DataFrame
        table specifying how many tours of each type to create
        one row per person (or parent_tour for atwork subtours)
        one (int) column per tour_type, with number of tours to create
    tour_category : str
        one of 'mandatory', 'non_mandatory', 'atwork', or 'joint'

    Returns
    -------
    tours : pandas.DataFrame
        An example of a tours DataFrame is supplied as a comment in the
        source code - it has an index which is a unique tour identifier,
        a person_id column, and a tour type column which comes from the
        column names of the alternatives DataFrame supplied above.

        tours.tour_type       - tour type (e.g. school, work, shopping, eat)
        tours.composition     - tour composition (e.g. adults, children, mixed)
        tours.tour_type_num   - if there are two 'school' type tours, they will be numbered 1 and 2
        tours.tour_type_count - number of tours of tour_type parent has (parent's max tour_type_num)
        tours.tour_num        - index of tour (of any type) for parent
        tours.tour_count      - number of tours of any type) for parent (parent's max tour_num)
        tours.tour_category   - one of 'mandatory', 'non_mandatory', 'atwork', or 'joint'
    """

    # FIXME - document requirement to ensure adjacent tour_type_nums in tour_num order

    """
               alt1       alt2     alt3
    <parent_col>
    2588676       2         0         0
    2588677       1         1         0
    """

    if model_settings is None:
        model_settings = JointTourFreqCompSettings.read_settings_file(
            state.filesystem, model_settings_file_name
        )

    alts_table_structure = model_settings.ALTS_TABLE_STRUCTURE
    tour_type_dict = alts_table_structure.PURPOSE.VALUE_MAP
    tour_type_cols = alts_table_structure.PURPOSE.COLUMNS
    tour_comp_dict = alts_table_structure.COMPOSITION.VALUE_MAP
    tour_comp_cols = alts_table_structure.COMPOSITION.COLUMNS

    # reformat with the columns given below
    tours_purp = tour_counts[tour_type_cols].stack().reset_index()
    tours_purp.columns = [parent_col, "tour_id_temp", "tour_type"]
    tours_purp["tour_id_temp"] = range(1, 1 + len(tours_purp))
    tours_purp["tour_type"] = tours_purp["tour_type"].map(tour_type_dict)

    """
        <parent_col> tour_id_temp  tour_type
    0     2588676    purpose1           5
    1     2588676    purpose2           0
    2     2588677    purpose1           5
    3     2588677    purpose2           5
    4     2588678    purpose1           6
    5     2588678    purpose2           7

    parent_col is the index from non_mandatory_tour_frequency
    tour_type is the column name from non_mandatory_tour_frequency_alts
    tour_type_count is the count value of the tour's chosen alt's tour_type from alts table
    """
    tours_comp = tour_counts[tour_comp_cols].stack().reset_index()
    tours_comp.columns = [parent_col, "tour_id_temp", "composition"]
    tours_comp["tour_id_temp"] = range(1, 1 + len(tours_comp))
    tours_comp["composition"] = tours_comp["composition"].map(tour_comp_dict)
    cat_tour_comp = pd.api.types.CategoricalDtype(
        ["adults", "children", "mixed"], ordered=False
    )
    tours_comp["composition"] = tours_comp["composition"].astype(cat_tour_comp)

    """
        <parent_col> tour_id_temp  tour_composition
    0     2588676    party1           1
    1     2588676    party2           0
    2     2588677    party1           1
    3     2588677    party2           1
    4     2588678    party1           1
    5     2588678    party2           2

    parent_col is the index from non_mandatory_tour_frequency
    tour_type is the column name from non_mandatory_tour_frequency_alts
    tour_type_count is the count value of the tour's chosen alt's tour_type from alts table
    """
    tours = pd.merge(
        tours_purp, tours_comp, how="left", on=[parent_col, "tour_id_temp"]
    )

    tours = tours[(tours.tour_type.notnull()) & (tours.tour_type.notnull())]

    grouped = tours.groupby([parent_col, "tour_type"])
    tours["tour_type_num"] = grouped.cumcount() + 1
    tours["tour_type_count"] = tours["tour_type_num"] + grouped.cumcount(
        ascending=False
    )

    grouped = tours.groupby(parent_col)
    tours["tour_num"] = grouped.cumcount() + 1
    tours["tour_count"] = tours["tour_num"] + grouped.cumcount(ascending=False)

    """
        <parent_col> tour_type  tour_type_num  tour_type_count tour_num  tour_count
    0     2588676       alt1           1           2               1         4
    0     2588676       alt1           2           2               2         4
    0     2588676       alt2           1           1               3         4
    0     2588676       alt3           1           1               4         4
    """

    # set these here to ensure consistency across different tour categories
    assert tour_category in ["mandatory", "non_mandatory", "atwork", "joint"]
    tours["tour_category"] = tour_category
    cat_tour_category = pd.api.types.CategoricalDtype(
        ["mandatory", "joint", "non_mandatory", "atwork"], ordered=False
    )
    tours["tour_category"] = tours["tour_category"].astype(cat_tour_category)

    # for joint tours, the correct number will be filled in after participation step
    tours["number_of_participants"] = 1

    # index is arbitrary but don't want any duplicates in index
    tours.reset_index(drop=True, inplace=True)

    # downcast
    tours["tour_count"] = tours["tour_count"].astype("int8")
    tours["tour_num"] = tours["tour_num"].astype("int8")
    tours["tour_type_num"] = tours["tour_type_num"].astype("int8")
    tours["tour_type_count"] = tours["tour_type_count"].astype("int8")
    tours["number_of_participants"] = tours["number_of_participants"].astype("int8")
    tours["tour_type"] = tours["tour_type"].astype("category")

    return tours
