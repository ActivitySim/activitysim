# ActivitySim
# See full license in LICENSE.txt.

import pandas as pd

from activitysim.abm.models.util.canonical_ids import (
    determine_flavors_from_alts_file,
    determine_mandatory_tour_flavors,
)


def test_mandatory_tour_flavors():
    mtf_settings = {}
    default_mandatory_tour_flavors = {"work": 2, "school": 2}

    # first test using default
    mandatory_tour_flavors = determine_mandatory_tour_flavors(
        mtf_settings,
        pd.DataFrame(columns=["random_name"]),
        default_mandatory_tour_flavors,
    )

    assert mandatory_tour_flavors == default_mandatory_tour_flavors

    # creating dummy spec with different values
    model_spec = pd.DataFrame(
        data={
            "Label": ["dummy"],
            "Description": ["dummy"],
            "Expression": [""],
            "work1": [1],
            "work2": [1],
            "work3": [1],
            "school1": [1],
            "school2": [1],
            "school3": [1],
            "work_and_school": [1],
        }
    )

    # second test reading from spec
    mandatory_tour_flavors = determine_mandatory_tour_flavors(
        mtf_settings, model_spec, default_mandatory_tour_flavors
    )
    assert mandatory_tour_flavors == {"work": 3, "school": 3}

    # third test is reading flavors from settings
    mtf_settings["MANDATORY_TOUR_FLAVORS"] = {"work": 3, "school": 2}
    mandatory_tour_flavors = determine_mandatory_tour_flavors(
        mtf_settings, model_spec, default_mandatory_tour_flavors
    )

    assert mandatory_tour_flavors == {"work": 3, "school": 2}


def test_tour_flavors_from_alt_files():
    # alternative tour frequency files are used in joint, atwork, and non-mandatory tour frequency models
    # this unit test checks the output from determining flavors from an alt file

    default_tour_flavors = {
        "escort": 2,
        "othmaint": 1,
        "othdiscr": 1,
    }

    # first test using default
    tour_flavors = determine_flavors_from_alts_file(
        pd.DataFrame(columns=["random_name"]),
        provided_flavors=None,
        default_flavors=default_tour_flavors,
    )

    assert tour_flavors == default_tour_flavors

    # second test is reading from alts file
    alts = pd.DataFrame(
        data={
            "Alts": ["alt1", "alt2", "alt3", "alt4"],
            "escort": [0, 1, 2, 3],
            "othmaint": [0, 0, 0, 0],
            "othdiscr": [1, 2, 0, 0],
        }
    )

    tour_flavors = determine_flavors_from_alts_file(
        alts, provided_flavors=None, default_flavors=default_tour_flavors
    )
    assert tour_flavors == {"escort": 3, "othmaint": 0, "othdiscr": 2}

    # now with max extension applied
    tour_flavors = determine_flavors_from_alts_file(
        alts,
        provided_flavors=None,
        default_flavors=default_tour_flavors,
        max_extension=2,
    )
    assert tour_flavors == {"escort": 5, "othmaint": 2, "othdiscr": 4}

    # now with provided tour flavors (which will ignore the max extension supplied too)
    tour_flavors = determine_flavors_from_alts_file(
        alts,
        provided_flavors={"escort": 3, "othmaint": 3, "othdiscr": 3},
        default_flavors=default_tour_flavors,
        max_extension=2,
    )
    assert tour_flavors == {"escort": 3, "othmaint": 3, "othdiscr": 3}
