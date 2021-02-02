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
    cv_to_ca,
)
from larch import Model, DataFrames, P, X


def size_coefficients_from_spec(size_spec):
    size_coef = size_spec.stack().reset_index()
    size_coef.index = size_coef.iloc[:, 0] + "_" + size_coef.iloc[:, 1]
    size_coef = size_coef.loc[size_coef.iloc[:, 2] > 0]
    size_coef["constrain"] = "F"
    one_each = size_coef.groupby("segment").first().reset_index()
    size_coef.loc[one_each.iloc[:, 0] + "_" + one_each.iloc[:, 1], "constrain"] = "T"
    size_coef = size_coef.iloc[:, 2:]
    size_coef.columns = ["value", "constrain"]
    size_coef.index.name = "coefficient_name"
    size_coef["value"] = np.log(size_coef["value"])
    return size_coef


def location_choice_model(
    model_selector="workplace",
    edb_directory="output/estimation_data_bundle/{model_selector}_location/",
    coefficients_file="{model_selector}_location_coefficients.csv",
    spec_file="{model_selector}_location_SPEC.csv",
    size_spec_file="{model_selector}_location_size_terms.csv",
    alt_values_file="{model_selector}_location_alternatives_combined.csv",
    chooser_file="{model_selector}_location_choosers_combined.csv",
    settings_file="{model_selector}_location_model_settings.yaml",
    landuse_file="{model_selector}_location_landuse.csv",
    return_data=False,
):
    edb_directory = edb_directory.format(model_selector=model_selector)

    def _read_csv(filename, **kwargs):
        filename = filename.format(model_selector=model_selector)
        return pd.read_csv(os.path.join(edb_directory, filename), **kwargs)

    coefficients = _read_csv(coefficients_file, index_col="coefficient_name",)
    spec = _read_csv(spec_file)
    alt_values = _read_csv(alt_values_file)
    chooser_data = _read_csv(chooser_file)
    landuse = _read_csv(landuse_file, index_col="zone_id")
    master_size_spec = _read_csv(size_spec_file)

    settings_file = settings_file.format(model_selector=model_selector)
    with open(os.path.join(edb_directory, settings_file), "r") as yf:
        settings = yaml.load(yf, Loader=yaml.SafeLoader,)
    CHOOSER_SEGMENT_COLUMN_NAME = settings["CHOOSER_SEGMENT_COLUMN_NAME"]
    SEGMENT_IDS = settings["SEGMENT_IDS"]

    # filter size spec for this location choice only
    size_spec = (
        master_size_spec.query(f"model_selector == '{model_selector}'")
        .drop(columns="model_selector")
        .set_index("segment")
    )
    size_spec = size_spec.loc[:, size_spec.max() > 0]

    size_coef = size_coefficients_from_spec(size_spec)

    # Remove shadow pricing and pre-existing size expression for re-estimation
    spec = (
        spec.set_index("Label")
        .drop(
            index=[
                "util_size_variable",  # pre-computed size (will be re-estimated)
                "util_utility_adjustment",  # shadow pricing (ignored in estimation)
            ]
        )
        .reset_index()
    )

    m = Model()
    if len(spec.columns) == 4:  # ['Label', 'Description', 'Expression', 'coefficient']
        m.utility_ca = linear_utility_from_spec(
            spec, x_col="Label", p_col="coefficient", ignore_x=("local_dist",),
        )
    else:
        m.utility_ca = linear_utility_from_spec(
            spec,
            x_col="Label",
            p_col=SEGMENT_IDS,
            ignore_x=("local_dist",),
            segment_id=CHOOSER_SEGMENT_COLUMN_NAME,
        )

    m.quantity_ca = sum(
        P(f"{i}_{q}") * X(q) * X(f"{CHOOSER_SEGMENT_COLUMN_NAME}=={SEGMENT_IDS[i]}")
        for i in size_spec.index
        for q in size_spec.columns
        if size_spec.loc[i, q] != 0
    )

    apply_coefficients(coefficients, m)
    apply_coefficients(size_coef, m, minimum=-6, maximum=6)

    x_co = chooser_data.set_index("person_id")
    x_ca = cv_to_ca(alt_values.set_index(["person_id", "variable"]))

    # label segments with names
    SEGMENT_IDS_REVERSE = {v: k for k, v in SEGMENT_IDS.items()}
    x_co["_segment_label"] = x_co[CHOOSER_SEGMENT_COLUMN_NAME].apply(
        lambda x: SEGMENT_IDS_REVERSE[x]
    )

    # compute total size values by segment
    for segment in size_spec.index:
        total_size_segment = pd.Series(0, index=landuse.index)
        x_co["total_size_" + segment] = 0
        for land_use_field in size_spec.loc[segment].index:
            total_size_segment += (
                landuse[land_use_field] * size_spec.loc[segment, land_use_field]
            )
        x_co["total_size_" + segment] = total_size_segment.loc[
            x_co["override_choice"]
        ].to_numpy()

    # for each chooser, collate the appropriate total size value
    x_co["total_size_segment"] = 0
    for segment in size_spec.index:
        labels = "total_size_" + segment
        rows = x_co["_segment_label"] == segment
        x_co.loc[rows, "total_size_segment"] = x_co[labels][rows]

    # Remove choosers with invalid observed choice (appropriate total size value = 0)
    valid_observed_zone = x_co["total_size_segment"] > 0
    x_co = x_co[valid_observed_zone]
    x_ca = x_ca[x_ca.index.get_level_values("person_id").isin(x_co.index)]

    # Merge land use characteristics into CA data
    x_ca_1 = pd.merge(x_ca, landuse, on="zone_id", how="left")
    x_ca_1.index = x_ca.index

    # Availability of choice zones
    av = x_ca_1["util_no_attractions"].apply(lambda x: False if x == 1 else True)

    d = DataFrames(co=x_co, ca=x_ca_1, av=av,)
    m.dataservice = d
    m.choice_co_code = "override_choice"

    if return_data:
        return (
            m,
            Dict(
                edb_directory=Path(edb_directory),
                alt_values=alt_values,
                chooser_data=chooser_data,
                coefficients=coefficients,
                landuse=landuse,
                spec=spec,
                size_spec=size_spec,
                master_size_spec=master_size_spec,
                model_selector=model_selector,
            ),
        )

    return m


def update_size_spec(model, data, result_dir=Path('.'), output_file=None):
    master_size_spec = data.master_size_spec
    size_spec = data.size_spec
    model_selector = data.model_selector

    # Write size coefficients into size_spec
    for c in size_spec.columns:
        for i in size_spec.index:
            param_name = f"{i}_{c}"
            j = (master_size_spec['segment'] == i) & (master_size_spec['model_selector'] == model_selector)
            try:
                master_size_spec.loc[j, c] = np.exp(model.get_value(param_name))
            except KeyError:
                pass

    # Rescale each row to total 1, not mathematically needed
    # but to maintain a consistent approach from existing ASim
    master_size_spec.iloc[:, 2:] = (
        master_size_spec.iloc[:, 2:].div(master_size_spec.iloc[:, 2:].sum(1), axis=0)
    )

    if output_file is not None:
        os.makedirs(result_dir, exist_ok=True)
        master_size_spec.reset_index().to_csv(
            result_dir/output_file,
            index=False,
        )

    return master_size_spec
