import numpy as np
import pandas as pd
import pytest

from activitysim.abm.models.vehicle_type_choice import (
    get_combinatorial_vehicle_alternatives,
)


def test_get_combinatorial_vehicle_alternatives():
    test_alts_cats_dict = {
        "body_type": ["Car", "Van"],
        "age": [1, 2],
        "fuel_type": ["Gas", "PEV"],
    }
    alts_wide, alts_long = get_combinatorial_vehicle_alternatives(test_alts_cats_dict)

    expected_wide = pd.DataFrame(
        {
            "body_type_Car": {0: 1, 1: 1, 2: 1, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0},
            "body_type_Van": {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1},
            "age_1": {0: 1, 1: 1, 2: 0, 3: 0, 4: 1, 5: 1, 6: 0, 7: 0},
            "age_2": {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1, 7: 1},
            "fuel_type_Gas": {0: 1, 1: 0, 2: 1, 3: 0, 4: 1, 5: 0, 6: 1, 7: 0},
            "fuel_type_PEV": {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0, 7: 1},
            "body_type": {
                0: "Car",
                1: "Car",
                2: "Car",
                3: "Car",
                4: "Van",
                5: "Van",
                6: "Van",
                7: "Van",
            },
            "age": {0: "1", 1: "1", 2: "2", 3: "2", 4: "1", 5: "1", 6: "2", 7: "2"},
            "fuel_type": {
                0: "Gas",
                1: "PEV",
                2: "Gas",
                3: "PEV",
                4: "Gas",
                5: "PEV",
                6: "Gas",
                7: "PEV",
            },
        }
    )

    expected_long = pd.DataFrame(
        {
            "body_type": {
                0: "Car",
                1: "Car",
                2: "Car",
                3: "Car",
                4: "Van",
                5: "Van",
                6: "Van",
                7: "Van",
            },
            "age": {0: "1", 1: "1", 2: "2", 3: "2", 4: "1", 5: "1", 6: "2", 7: "2"},
            "fuel_type": {
                0: "Gas",
                1: "PEV",
                2: "Gas",
                3: "PEV",
                4: "Gas",
                5: "PEV",
                6: "Gas",
                7: "PEV",
            },
        }
    )

    pd.testing.assert_frame_equal(alts_wide, expected_wide, check_dtype=False)
    pd.testing.assert_frame_equal(alts_long, expected_long, check_dtype=False)
