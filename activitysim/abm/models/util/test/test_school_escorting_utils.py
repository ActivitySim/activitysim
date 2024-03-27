# ActivitySim
# See full license in LICENSE.txt.
import os
from ast import literal_eval
import pandas as pd
import numpy as np
import pandas.testing as pdt

from activitysim.abm.models.util.school_escort_tours_trips import (
    create_bundle_attributes,
)


def test_create_bundle_attributes():

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    dtype_dict = {
        "escortees": "str",
        "escortee_nums": "str",
        "school_destinations": "str",
        "school_starts": "str",
        "school_ends": "str",
        "school_tour_ids": "str",
    }

    inbound_input = pd.read_csv(
        os.path.join(data_dir, "create_bundle_attributes__input_inbound.csv"),
        index_col=0,
    )
    inbound_output = pd.read_csv(
        os.path.join(data_dir, "create_bundle_attributes__output_inbound.csv"),
        index_col=0,
        dtype=dtype_dict,
    )

    outbound_input = pd.read_csv(
        os.path.join(data_dir, "create_bundle_attributes__input_outbound_cond.csv"),
        index_col=0,
    )
    outbound_output = pd.read_csv(
        os.path.join(data_dir, "create_bundle_attributes__output_outbound_cond.csv"),
        index_col=0,
        dtype=dtype_dict,
    )

    # need to convert columns from string back to list
    list_columns = ["outbound_order", "inbound_order", "child_order"]
    for col in list_columns:
        inbound_input[col] = inbound_input[col].apply(
            lambda x: x.strip("[]").split(" ")
        )
        outbound_input[col] = outbound_input[col].apply(
            lambda x: x.strip("[]").split(" ")
        )
        inbound_output[col] = inbound_output[col].apply(
            lambda x: x.strip("[]").split(" ")
        )
        outbound_output[col] = outbound_output[col].apply(
            lambda x: x.strip("[]").split(" ")
        )

    inbound_result = create_bundle_attributes(inbound_input)
    pdt.assert_frame_equal(inbound_result, inbound_output, check_dtype=False)

    outbound_result = create_bundle_attributes(outbound_input)
    pdt.assert_frame_equal(outbound_result, outbound_output, check_dtype=False)


if __name__ == "__main__":
    test_create_bundle_attributes()
