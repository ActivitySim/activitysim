# ActivitySim
# See full license in LICENSE.txt.
import logging
import re

import numpy as np
import pandas as pd
import re

from activitysim.core import config
from activitysim.core import pipeline
from activitysim.core import simulate

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


def read_alts_file(file_name, set_index=None):
    try:
        alts = simulate.read_model_alts(file_name, set_index=set_index)
    except (RuntimeError, FileNotFoundError):
        logger.warning(f"Could not find file {file_name} to determine tour flavors.")
        return pd.DataFrame()
    return alts


def parse_tour_flavor_from_columns(columns, tour_flavor):
    """
    determines the max number from columns if column name contains tour flavor
    example: columns={'work1', 'work2'} -> 2

    Parameters
    ----------
    columns : list of str
    tour_flavor : str
        string subset that you want to find in columns

    Returns
    -------
    int
        max int found in columns with tour_flavor
    """
    # below produces a list of numbers present in each column containing the tour flavor string
    tour_numbers = [(re.findall(r"\d+", col)) for col in columns if tour_flavor in col]

    # flatten list
    tour_numbers = [int(item) for sublist in tour_numbers for item in sublist]

    # find max
    try:
        max_tour_flavor = max(tour_numbers)
        return max_tour_flavor
    except ValueError:
        # could not find a maximum integer for this flavor in the columns
        return -1


def determine_mandatory_tour_flavors(mtf_settings, model_spec, default_flavors):
    provided_flavors = mtf_settings.get("MANDATORY_TOUR_FLAVORS", None)

    mandatory_tour_flavors = {
        # hard code work and school tours
        "work": parse_tour_flavor_from_columns(model_spec.columns, "work"),
        "school": parse_tour_flavor_from_columns(model_spec.columns, "school"),
    }

    valid_flavors = (mandatory_tour_flavors["work"] >= 1) & (
        mandatory_tour_flavors["school"] >= 1
    )

    if provided_flavors is not None:
        if mandatory_tour_flavors != provided_flavors:
            logger.warning(
                "Specified tour flavors do not match alternative file flavors"
            )
            logger.warning(
                f"{provided_flavors} does not equal {mandatory_tour_flavors}"
            )
        # use provided flavors if provided
        return provided_flavors

    if not valid_flavors:
        # if flavors could not be parsed correctly and no flavors provided, return the default
        logger.warning(
            "Could not determine alts from alt file and no flavors were provided."
        )
        logger.warning(f"Using defaults: {default_flavors}")
        return default_flavors

    return mandatory_tour_flavors


def determine_non_mandatory_tour_max_extension(
    model_settings, extension_probs, default_max_extension=2
):
    provided_max_extension = model_settings.get("MAX_EXTENSION", None)

    max_extension = parse_tour_flavor_from_columns(extension_probs.columns, "tour")

    if provided_max_extension is not None:
        if provided_max_extension != max_extension:
            logger.warning(
                "Specified non mandatory tour extension does not match extension probabilities file"
            )
        return provided_max_extension

    if (max_extension >= 0) & isinstance(max_extension, int):
        return max_extension

    return default_max_extension


def determine_flavors_from_alts_file(
    alts, provided_flavors, default_flavors, max_extension=0
):
    """
    determines the max number from alts for each column containing numbers
    example: alts={'index': ['alt1', 'alt2'], 'escort': [1, 2], 'othdisc': [3, 4]}
             yelds -> {'escort': 2, 'othdisc': 4}

    will return provided flavors if available
    else, return default flavors if alts can't be groked

    Parameters
    ----------
    alts : pd.DataFrame
    provided_flavors : dict, optional
        tour flavors provided by user in the model yaml
    default_flavors : dict
        default tour flavors to fall back on
    max_extension : int
        scale to increase number of tours accross all alternatives

    Returns
    -------
    dict
        tour flavors
    """
    try:
        flavors = {
            c: int(alts[c].max() + max_extension)
            for c in alts.columns
            if all(alts[c].astype(str).str.isnumeric())
        }
        valid_flavors = all(
            [(isinstance(flavor, str) & (num >= 0)) for flavor, num in flavors.items()]
        ) & (len(flavors) > 0)
    except (ValueError, AttributeError):
        valid_flavors = False

    if provided_flavors is not None:
        if flavors != provided_flavors:
            logger.warning(
                f"Specified tour flavors {provided_flavors} do not match alternative file flavors {flavors}"
            )
        # use provided flavors if provided
        return provided_flavors

    if not valid_flavors:
        # if flavors could not be parsed correctly and no flavors provided, return the default
        logger.warning(
            "Could not determine alts from alt file and no flavors were provided."
        )
        logger.warning(f"Using defaults: {default_flavors}")
        return default_flavors

    return flavors


def canonical_tours():
    """
        create labels for every the possible tour by combining tour_type/tour_num.

    Returns
    -------
        list of canonical tour labels in alphabetical order
    """

    # ---- non_mandatory_channels
    nm_model_settings_file_name = "non_mandatory_tour_frequency.yaml"
    nm_model_settings = config.read_model_settings(nm_model_settings_file_name)
    nm_alts = read_alts_file("non_mandatory_tour_frequency_alternatives.csv")

    # first need to determine max extension
    try:
        ext_probs_f = config.config_file_path(
            "non_mandatory_tour_frequency_extension_probs.csv"
        )
        extension_probs = pd.read_csv(ext_probs_f, comment="#")
    except (RuntimeError, FileNotFoundError):
        logger.warning(
            f"non_mandatory_tour_frequency_extension_probs.csv file not found"
        )
        extension_probs = pd.DataFrame()
    max_extension = determine_non_mandatory_tour_max_extension(
        nm_model_settings, extension_probs, default_max_extension=2
    )

    provided_nm_tour_flavors = nm_model_settings.get("NON_MANDATORY_TOUR_FLAVORS", None)
    default_nm_tour_flavors = {
        "escort": 2 + max_extension,
        "shopping": 1 + max_extension,
        "othmaint": 1 + max_extension,
        "othdiscr": 1 + max_extension,
        "eatout": 1 + max_extension,
        "social": 1 + max_extension,
    }

    non_mandatory_tour_flavors = determine_flavors_from_alts_file(
        nm_alts, provided_nm_tour_flavors, default_nm_tour_flavors, max_extension
    )
    non_mandatory_channels = enumerate_tour_types(non_mandatory_tour_flavors)

    logger.info(f"Non-Mandatory tour flavors used are {non_mandatory_tour_flavors}")

    # ---- mandatory_channels
    mtf_model_settings_file_name = "mandatory_tour_frequency.yaml"
    mtf_model_settings = config.read_model_settings(mtf_model_settings_file_name)
    mtf_spec = mtf_model_settings.get("SPEC", "mandatory_tour_frequency.csv")
    mtf_model_spec = read_alts_file(file_name=mtf_spec)
    default_mandatory_tour_flavors = {"work": 2, "school": 2}

    mandatory_tour_flavors = determine_mandatory_tour_flavors(
        mtf_model_settings, mtf_model_spec, default_mandatory_tour_flavors
    )
    mandatory_channels = enumerate_tour_types(mandatory_tour_flavors)

    logger.info(f"Mandatory tour flavors used are {mandatory_tour_flavors}")

    # ---- atwork_subtour_channels
    atwork_model_settings_file_name = "atwork_subtour_frequency.yaml"
    atwork_model_settings = config.read_model_settings(atwork_model_settings_file_name)
    atwork_alts = read_alts_file("atwork_subtour_frequency_alternatives.csv")

    provided_atwork_flavors = atwork_model_settings.get("ATWORK_SUBTOUR_FLAVORS", None)
    default_atwork_flavors = {"eat": 1, "business": 2, "maint": 1}

    atwork_subtour_flavors = determine_flavors_from_alts_file(
        atwork_alts, provided_atwork_flavors, default_atwork_flavors
    )
    atwork_subtour_channels = enumerate_tour_types(atwork_subtour_flavors)

    logger.info(f"Atwork subtour flavors used are {atwork_subtour_flavors}")

    # we need to distinguish between subtours of different work tours
    # (e.g. eat1_1 is eat subtour for parent work tour 1 and eat1_2 is for work tour 2)
    max_work_tours = mandatory_tour_flavors["work"]
    atwork_subtour_channels = [
        "%s_%s" % (c, i + 1)
        for c in atwork_subtour_channels
        for i in range(max_work_tours)
    ]

    # ---- joint_tour_channels
    jtf_model_settings_file_name = "joint_tour_frequency.yaml"
    jtf_model_settings = config.read_model_settings(jtf_model_settings_file_name)
    jtf_alts = read_alts_file("joint_tour_frequency_alternatives.csv")
    provided_joint_flavors = jtf_model_settings.get("JOINT_TOUR_FLAVORS", None)

    default_joint_flavors = {
        "shopping": 2,
        "othmaint": 2,
        "othdiscr": 2,
        "eatout": 2,
        "social": 2,
    }
    joint_tour_flavors = determine_flavors_from_alts_file(
        jtf_alts, provided_joint_flavors, default_joint_flavors
    )
    logger.info(f"Joint tour flavors used are {joint_tour_flavors}")

    joint_tour_channels = enumerate_tour_types(joint_tour_flavors)
    joint_tour_channels = ["j_%s" % c for c in joint_tour_channels]

    sub_channels = (
        non_mandatory_channels
        + mandatory_channels
        + atwork_subtour_channels
        + joint_tour_channels
    )

    # ---- school escort channels
    # only include if model is run
    if pipeline.is_table("school_escort_tours") | (
        "school_escorting" in config.setting("models", default=[])
    ):
        se_model_settings_file_name = "school_escorting.yaml"
        se_model_settings = config.read_model_settings(se_model_settings_file_name)
        num_escortees = se_model_settings.get("NUM_ESCORTEES", 3)
        school_escort_flavors = {"escort": 2 * num_escortees}
        school_escort_channels = enumerate_tour_types(school_escort_flavors)
        school_escort_channels = ["se_%s" % c for c in school_escort_channels]
        logger.info(f"School escort tour flavors used are {school_escort_flavors}")

        sub_channels = sub_channels + school_escort_channels

    sub_channels.sort()

    return sub_channels


def set_tour_index(
    tours, parent_tour_num_col=None, is_joint=False, is_school_escorting=False
):
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

    if is_school_escorting:
        tours["tour_id"] = "se_" + tours["tour_id"]

    # map recognized strings to ints
    tours.tour_id = tours.tour_id.replace(
        to_replace=possible_tours, value=list(range(possible_tours_count))
    )

    # convert to numeric - shouldn't be any NaNs - this will raise error if there are
    tours.tour_id = pd.to_numeric(tours.tour_id, errors="raise").astype(np.int64)

    tours.tour_id = (tours.person_id * possible_tours_count) + tours.tour_id

    if tours.tour_id.duplicated().any():
        print(
            "\ntours.tour_id not unique\n%s"
            % tours[tours.tour_id.duplicated(keep=False)]
        )
        print(
            tours[tours.tour_id.duplicated(keep=False)][
                ["survey_tour_id", "tour_type", "tour_category"]
            ]
        )
    assert not tours.tour_id.duplicated().any()

    tours.set_index("tour_id", inplace=True, verify_integrity=True)

    # we modify tours in place, but return the dataframe for the convenience of the caller
    return tours


def determine_max_trips_per_leg(default_max_trips_per_leg=4):
    model_settings_file_name = "stop_frequency.yaml"
    model_settings = config.read_model_settings(model_settings_file_name)

    # first see if flavors given explicitly
    provided_max_trips_per_leg = model_settings.get("MAX_TRIPS_PER_LEG", None)

    # determine flavors from alternative file
    try:
        alts = read_alts_file("stop_frequency_alternatives.csv")
        trips_per_leg = [
            int(alts[c].max())
            for c in alts.columns
            if all(alts[c].astype(str).str.isnumeric())
        ]
        # adding one for additional trip home or to primary dest
        max_trips_per_leg = max(trips_per_leg) + 1
        if max_trips_per_leg > 1:
            valid_max_trips = True
    except (ValueError, RuntimeError):
        valid_max_trips = False

    if provided_max_trips_per_leg is not None:
        if provided_max_trips_per_leg != max_trips_per_leg:
            logger.warning(
                "Provided max number of stops on tour does not match with stop frequency alternatives file"
            )
        return provided_max_trips_per_leg

    if valid_max_trips:
        return max_trips_per_leg

    return default_max_trips_per_leg


def set_trip_index(trips, tour_id_column="tour_id"):
    # max number of trips per leg (inbound or outbound) of tour
    #  = stops + 1 for primary half-tour destination
    max_trips_per_leg = determine_max_trips_per_leg()

    # canonical_trip_num: 1st trip out = 1, 2nd trip out = 2, 1st in = 5, etc.
    canonical_trip_num = (~trips.outbound * max_trips_per_leg) + trips.trip_num
    trips["trip_id"] = (
        trips[tour_id_column] * (2 * max_trips_per_leg) + canonical_trip_num
    )
    trips.set_index("trip_id", inplace=True, verify_integrity=True)

    # we modify trips in place, but return the dataframe for the convenience of the caller
    return trips
