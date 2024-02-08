# ActivitySim
# See full license in LICENSE.txt.

import pandas as pd
import pandas.testing as pdt

from activitysim.abm.models.vehicle_type_choice import (
    get_combinatorial_vehicle_alternatives,
    construct_model_alternatives,
    VehicleTypeChoiceSettings,
)
from activitysim.core import workflow


def test_vehicle_type_alts():
    state = workflow.State.make_default(__file__)

    alts_cats_dict = {
        "body_type": ["Car", "SUV"],
        "fuel_type": ["Gas", "BEV"],
        "age": [1, 2, 3],
    }

    alts_wide, alts_long = get_combinatorial_vehicle_alternatives(alts_cats_dict)

    # alts are initially constructed combinatorially
    assert len(alts_long) == 12, "alts_long should have 12 rows"
    assert len(alts_wide) == 12, "alts_wide should have 12 rows"

    model_settings = VehicleTypeChoiceSettings.model_construct()
    model_settings.combinatorial_alts = alts_cats_dict
    model_settings.PROBS_SPEC = None
    model_settings.WRITE_OUT_ALTS_FILE = False

    # constructing veh type data with missing alts
    vehicle_type_data = pd.DataFrame(
        data={
            "body_type": ["Car", "Car", "Car", "SUV", "SUV"],
            "fuel_type": ["Gas", "Gas", "BEV", "Gas", "BEV"],
            "age": ["1", "2", "3", "1", "2"],
            "dummy_data": [1, 2, 3, 4, 5],
        },
        index=[0, 1, 2, 3, 4],
    )

    alts_wide, alts_long = construct_model_alternatives(
        state, model_settings, alts_cats_dict, vehicle_type_data
    )

    # should only have alts left that are in the file
    assert len(alts_long) == 5, "alts_long should have 5 rows"

    # indexes need to be the same to choices match alts
    pdt.assert_index_equal(alts_long.index, alts_wide.index)

    # columns need to be in correct order for downstream configs
    pdt.assert_index_equal(
        alts_long.columns, pd.Index(["body_type", "age", "fuel_type"])
    )
