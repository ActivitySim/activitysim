# ActivitySim
# See full license in LICENSE.txt.
import logging

import numpy as np
import pandas as pd

from activitysim.core.util import reindex

logger = logging.getLogger(__name__)


RANDOM_CHANNELS = [
    "households",
    "persons",
    "tours",
    "joint_tour_participants",
    "trips",
    "vehicles",
]
TRACEABLE_TABLES = [
    "households",
    "persons",
    "tours",
    "joint_tour_participants",
    "trips",
    "vehicles",
]

CANONICAL_TABLE_INDEX_NAMES = {
    "households": "household_id",
    "persons": "person_id",
    "tours": "tour_id",
    "joint_tour_participants": "participant_id",
    "trips": "trip_id",
    "land_use": "zone_id",
    "vehicles": "vehicle_id",
}

# unfortunately the two places this is needed (joint_tour_participation and estimation.infer
# don't have much in common in terms of data structures
# candidates['participant_id'] = (candidates[joint_tours.index.name] * MAX_PARTICIPANT_PNUM) + candidates.PNUM
MAX_PARTICIPANT_PNUM = 100


def enumerate_tour_types(tour_flavors):
    # tour_flavors: {'eat': 1, 'business': 2, 'maint': 1}
    # channels:      ['eat1', 'business1', 'business2', 'maint1']
    channels = [
        tour_type + str(tour_num)
        for tour_type, max_count in tour_flavors.items()
        for tour_num in range(1, max_count + 1)
    ]
    return channels


def canonical_tours():
    """
        create labels for every the possible tour by combining tour_type/tour_num.

    Returns
    -------
        list of canonical tour labels in alphabetical order
    """

    # FIXME we pathalogically know what the possible tour_types and their max tour_nums are
    # FIXME instead, should get flavors from alts tables (but we would have to know their names...)
    # alts = pipeline.get_table('non_mandatory_tour_frequency_alts')
    # non_mandatory_tour_flavors = {c : alts[c].max() for c in alts.columns}

    # - non_mandatory_channels
    MAX_EXTENSION = 2
    non_mandatory_tour_flavors = {
        "escort": 2 + MAX_EXTENSION,
        "shopping": 1 + MAX_EXTENSION,
        "othmaint": 1 + MAX_EXTENSION,
        "othdiscr": 1 + MAX_EXTENSION,
        "eatout": 1 + MAX_EXTENSION,
        "social": 1 + MAX_EXTENSION,
    }
    non_mandatory_channels = enumerate_tour_types(non_mandatory_tour_flavors)

    # - mandatory_channels
    mandatory_tour_flavors = {"work": 2, "school": 2}
    mandatory_channels = enumerate_tour_types(mandatory_tour_flavors)

    # - atwork_subtour_channels
    # we need to distinguish between subtours of different work tours
    # (e.g. eat1_1 is eat subtour for parent work tour 1 and eat1_2 is for work tour 2)
    atwork_subtour_flavors = {"eat": 1, "business": 2, "maint": 1}
    atwork_subtour_channels = enumerate_tour_types(atwork_subtour_flavors)
    max_work_tours = mandatory_tour_flavors["work"]
    atwork_subtour_channels = [
        "%s_%s" % (c, i + 1)
        for c in atwork_subtour_channels
        for i in range(max_work_tours)
    ]

    # - joint_tour_channels
    joint_tour_flavors = {
        "shopping": 2,
        "othmaint": 2,
        "othdiscr": 2,
        "eatout": 2,
        "social": 2,
    }
    joint_tour_channels = enumerate_tour_types(joint_tour_flavors)
    joint_tour_channels = ["j_%s" % c for c in joint_tour_channels]

    sub_channels = (
        non_mandatory_channels
        + mandatory_channels
        + atwork_subtour_channels
        + joint_tour_channels
    )

    sub_channels.sort()

    return sub_channels


def set_tour_index(tours, parent_tour_num_col=None, is_joint=False):
    """
    The new index values are stable based on the person_id, tour_type, and tour_num.
    The existing index is ignored and replaced.

    This gives us a stable (predictable) tour_id with tours in canonical order
    (when tours are sorted by tour_id, tours for each person
    of the same type will be adjacent and in increasing tour_type_num order)

    It also simplifies attaching random number streams to tours that are stable
    (even across simulations)

    Parameters
    ----------
    tours : DataFrame
        Tours dataframe to reindex.
    """

    tour_num_col = "tour_type_num"
    possible_tours = canonical_tours()
    possible_tours_count = len(possible_tours)

    assert tour_num_col in tours.columns

    # create string tour_id corresonding to keys in possible_tours (e.g. 'work1', 'j_shopping2')
    tours["tour_id"] = tours.tour_type + tours[tour_num_col].map(str)

    if parent_tour_num_col:
        # we need to distinguish between subtours of different work tours
        # (e.g. eat1_1 is eat subtour for parent work tour 1 and eat1_2 is for work tour 2)

        parent_tour_num = tours[parent_tour_num_col]
        if parent_tour_num.dtype != "int64":
            # might get converted to float if non-subtours rows are None (but we try to avoid this)
            logger.error("parent_tour_num.dtype: %s" % parent_tour_num.dtype)
            parent_tour_num = parent_tour_num.astype(np.int64)

        tours["tour_id"] = tours["tour_id"] + "_" + parent_tour_num.map(str)

    if is_joint:
        tours["tour_id"] = "j_" + tours["tour_id"]

    # map recognized strings to ints
    tours.tour_id = tours.tour_id.replace(
        to_replace=possible_tours, value=list(range(possible_tours_count))
    )

    # convert to numeric - shouldn't be any NaNs - this will raise error if there are
    tours.tour_id = pd.to_numeric(tours.tour_id, errors="raise").astype(np.int64)

    tours.tour_id = (tours.person_id * possible_tours_count) + tours.tour_id

    # if tours.tour_id.duplicated().any():
    #     print("\ntours.tour_id not unique\n%s" % tours[tours.tour_id.duplicated(keep=False)])
    #     print(tours[tours.tour_id.duplicated(keep=False)][['survey_tour_id', 'tour_type', 'tour_category']])
    assert not tours.tour_id.duplicated().any()

    tours.set_index("tour_id", inplace=True, verify_integrity=True)

    # we modify tours in place, but return the dataframe for the convenience of the caller
    return tours


def set_trip_index(trips, tour_id_column="tour_id"):

    MAX_TRIPS_PER_LEG = 4  # max number of trips per leg (inbound or outbound) of tour

    # canonical_trip_num: 1st trip out = 1, 2nd trip out = 2, 1st in = 5, etc.
    canonical_trip_num = (~trips.outbound * MAX_TRIPS_PER_LEG) + trips.trip_num
    trips["trip_id"] = (
        trips[tour_id_column] * (2 * MAX_TRIPS_PER_LEG) + canonical_trip_num
    )
    trips.set_index("trip_id", inplace=True, verify_integrity=True)

    # we modify trips in place, but return the dataframe for the convenience of the caller
    return trips
