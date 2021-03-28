import os
import numpy as np
import pandas as pd
import yaml
from typing import Collection
from larch.util import Dict
from pathlib import Path

from .general import (
    remove_apostrophes,
    construct_nesting_tree,
    linear_utility_from_spec,
    explicit_value_parameters,
    apply_coefficients,
    clean_values,
    simple_simulate_data,
)
from larch import Model, DataFrames, P, X


def tour_mode_choice_model(
    edb_directory="output/estimation_data_bundle/{name}/", return_data=False,
):
    data = simple_simulate_data(name="tour_mode_choice", edb_directory=edb_directory,)
    coefficients = data.coefficients
    coef_template = data.coef_template
    spec = data.spec
    chooser_data = data.chooser_data
    settings = data.settings

    chooser_data = clean_values(
        chooser_data,
        data.alt_names,
        alt_names_to_codes=data.alt_names_to_codes,
        choice_code="override_choice_code",
    )

    tree = construct_nesting_tree(data.alt_names, settings["NESTS"])

    purposes = list(coef_template.columns)

    # Setup purpose specific models
    m = {purpose: Model(graph=tree) for purpose in purposes}
    for alt_code, alt_name in tree.elemental_names().items():
        # Read in base utility function for this alt_name
        u = linear_utility_from_spec(
            spec, x_col="Label", p_col=alt_name, ignore_x=("#",),
        )
        for purpose in purposes:
            # Modify utility function based on template for purpose
            u_purp = sum(
                (P(coef_template[purpose].get(i.param, i.param)) * i.data * i.scale)
                for i in u
            )
            m[purpose].utility_co[alt_code] = u_purp

    for model in m.values():
        explicit_value_parameters(model)
    apply_coefficients(coefficients, m)

    avail = {}
    for acode, aname in data.alt_codes_to_names.items():
        unavail_cols = list(
            (
                chooser_data[i.data]
                if i.data in chooser_data
                else chooser_data.eval(i.data)
            )
            for i in m[purposes[0]].utility_co[acode]
            if i.param == "-999"
        )
        if len(unavail_cols):
            avail[acode] = sum(unavail_cols) == 0
        else:
            avail[acode] = 1
    avail = pd.DataFrame(avail).astype(np.int8)
    avail.index = chooser_data.index

    d = DataFrames(
        co=chooser_data, av=avail, alt_codes=data.alt_codes, alt_names=data.alt_names,
    )

    for purpose, model in m.items():
        model.dataservice = d.selector_co(f"tour_type=='{purpose}'")
        model.choice_co_code = "override_choice_code"

    from larch.model.model_group import ModelGroup

    mg = ModelGroup(m.values())

    if return_data:
        return (
            mg,
            Dict(
                edb_directory=Path(edb_directory),
                chooser_data=chooser_data,
                avail=avail,
                coefficients=coefficients,
                coef_template=coef_template,
                spec=spec,
            ),
        )

    return mg
