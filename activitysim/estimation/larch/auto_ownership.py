import os
from typing import Collection

import numpy as np
import pandas as pd
import yaml
from larch import DataFrames, Model, P, X
from larch.util import Dict

from .general import (
    apply_coefficients,
    dict_of_linear_utility_from_spec,
    remove_apostrophes,
)
from .simple_simulate import simple_simulate_data


def auto_ownership_model(
    name="auto_ownership",
    edb_directory="output/estimation_data_bundle/{name}/",
    return_data=False,
):
    data = simple_simulate_data(
        name=name,
        edb_directory=edb_directory,
        values_index_col="household_id",
    )
    coefficients = data.coefficients
    # coef_template = data.coef_template # not used
    spec = data.spec
    chooser_data = data.chooser_data
    settings = data.settings

    altnames = list(spec.columns[3:])
    altcodes = range(len(altnames))

    chooser_data = remove_apostrophes(chooser_data)
    chooser_data.fillna(0, inplace=True)

    # Remove choosers with invalid observed choice
    chooser_data = chooser_data[chooser_data["override_choice"] >= 0]

    m = Model()
    # One of the alternatives is coded as 0, so
    # we need to explicitly initialize the MNL nesting graph
    # and set to root_id to a value other than zero.
    m.initialize_graph(alternative_codes=altcodes, root_id=99)

    m.utility_co = dict_of_linear_utility_from_spec(
        spec,
        "Label",
        dict(zip(altnames, altcodes)),
    )

    apply_coefficients(coefficients, m)

    d = DataFrames(
        co=chooser_data,
        av=True,
        alt_codes=altcodes,
        alt_names=altnames,
    )

    m.dataservice = d
    m.choice_co_code = "override_choice"

    if return_data:
        return (
            m,
            Dict(
                edb_directory=data.edb_directory,
                chooser_data=chooser_data,
                coefficients=coefficients,
                spec=spec,
                altnames=altnames,
                altcodes=altcodes,
            ),
        )

    return m
