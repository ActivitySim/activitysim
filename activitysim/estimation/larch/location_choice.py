import os
from pathlib import Path
from typing import Collection

import numpy as np
import pandas as pd
import yaml
from larch import DataFrames, Model, P, X
from larch.util import Dict

from .general import (
    apply_coefficients,
    construct_nesting_tree,
    cv_to_ca,
    explicit_value_parameters,
    linear_utility_from_spec,
    remove_apostrophes,
    str_repr,
)


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
    name="workplace_location",
    edb_directory="output/estimation_data_bundle/{name}/",
    coefficients_file="{name}_coefficients.csv",
    spec_file="{name}_SPEC.csv",
    size_spec_file="{name}_size_terms.csv",
    alt_values_file="{name}_alternatives_combined.csv",
    chooser_file="{name}_choosers_combined.csv",
    settings_file="{name}_model_settings.yaml",
    landuse_file="{name}_landuse.csv",
    return_data=False,
):
    model_selector = name.replace("_location", "")
    model_selector = model_selector.replace("_destination", "")
    model_selector = model_selector.replace("_subtour", "")
    model_selector = model_selector.replace("_tour", "")
    if model_selector == "joint":
        model_selector = "non_mandatory"
    edb_directory = edb_directory.format(name=name)

    def _read_csv(filename, **kwargs):
        filename = filename.format(name=name)
        return pd.read_csv(os.path.join(edb_directory, filename), **kwargs)

    coefficients = _read_csv(
        coefficients_file,
        index_col="coefficient_name",
    )
    spec = _read_csv(spec_file, comment="#")
    alt_values = _read_csv(alt_values_file)
    chooser_data = _read_csv(chooser_file)
    landuse = _read_csv(landuse_file, index_col="zone_id")
    master_size_spec = _read_csv(size_spec_file)

    # remove temp rows from spec, ASim uses them to calculate the other values written
    # to the EDB, but they are not actually part of the utility function themselves.
    spec = spec.loc[~spec.Expression.isna()]
    spec = spec.loc[~spec.Expression.str.startswith("_")].copy()

    settings_file = settings_file.format(name=name)
    with open(os.path.join(edb_directory, settings_file), "r") as yf:
        settings = yaml.load(
            yf,
            Loader=yaml.SafeLoader,
        )

    include_settings = settings.get("include_settings")
    if include_settings:
        include_settings = os.path.join(edb_directory, include_settings)
    if include_settings and os.path.exists(include_settings):
        with open(include_settings, "r") as yf:
            more_settings = yaml.load(
                yf,
                Loader=yaml.SafeLoader,
            )
        settings.update(more_settings)

    CHOOSER_SEGMENT_COLUMN_NAME = settings.get("CHOOSER_SEGMENT_COLUMN_NAME")
    SEGMENT_IDS = settings.get("SEGMENT_IDS")
    if SEGMENT_IDS is None:
        SEGMENTS = settings.get("SEGMENTS")
        if SEGMENTS is not None:
            SEGMENT_IDS = {i: i for i in SEGMENTS}

    SIZE_TERM_SELECTOR = settings.get("SIZE_TERM_SELECTOR", model_selector)

    # filter size spec for this location choice only
    size_spec = (
        master_size_spec.query(f"model_selector == '{SIZE_TERM_SELECTOR}'")
        .drop(columns="model_selector")
        .set_index("segment")
    )
    size_spec = size_spec.loc[:, size_spec.max() > 0]

    size_coef = size_coefficients_from_spec(size_spec)

    indexes_to_drop = [
        "util_size_variable",  # pre-computed size (will be re-estimated)
        "util_size_variable_atwork",  # pre-computed size (will be re-estimated)
        "util_utility_adjustment",  # shadow pricing (ignored in estimation)
        "@df['size_term'].apply(np.log1p)",  # pre-computed size (will be re-estimated)
    ]
    if "Label" in spec.columns:
        indexes_to_drop = [i for i in indexes_to_drop if i in spec.Label.to_numpy()]
        label_column_name = "Label"
    elif "Expression" in spec.columns:
        indexes_to_drop = [
            i for i in indexes_to_drop if i in spec.Expression.to_numpy()
        ]
        label_column_name = "Expression"
    else:
        raise ValueError("cannot find Label or Expression in spec file")

    expression_labels = None
    if label_column_name == "Expression":
        expression_labels = {
            expr: f"variable_label{n:04d}"
            for n, expr in enumerate(spec.Expression.to_numpy())
        }

    # Remove shadow pricing and pre-existing size expression for re-estimation
    spec = spec.set_index(label_column_name).drop(index=indexes_to_drop).reset_index()

    if label_column_name == "Expression":
        spec.insert(0, "Label", spec["Expression"].map(expression_labels))
        alt_values["variable"] = alt_values["variable"].map(expression_labels)
        label_column_name = "Label"

    if name == "trip_destination":
        CHOOSER_SEGMENT_COLUMN_NAME = "primary_purpose"
        primary_purposes = spec.columns[3:]
        SEGMENT_IDS = {pp: pp for pp in primary_purposes}

    chooser_index_name = chooser_data.columns[0]
    x_co = chooser_data.set_index(chooser_index_name)
    x_ca = cv_to_ca(alt_values.set_index([chooser_index_name, alt_values.columns[1]]))

    if CHOOSER_SEGMENT_COLUMN_NAME is not None:
        # label segments with names
        SEGMENT_IDS_REVERSE = {v: k for k, v in SEGMENT_IDS.items()}
        x_co["_segment_label"] = x_co[CHOOSER_SEGMENT_COLUMN_NAME].apply(
            lambda x: SEGMENT_IDS_REVERSE[x]
        )
    else:
        x_co["_segment_label"] = size_spec.index[0]

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
    x_ca = x_ca[x_ca.index.get_level_values(chooser_index_name).isin(x_co.index)]

    # Merge land use characteristics into CA data
    try:
        x_ca_1 = pd.merge(x_ca, landuse, on="zone_id", how="left")
    except KeyError:
        # Missing the zone_id variable?
        # Use the alternative id's instead, which assumes no sampling of alternatives
        x_ca_1 = pd.merge(
            x_ca,
            landuse,
            left_on=x_ca.index.get_level_values(1),
            right_index=True,
            how="left",
        )
    x_ca_1.index = x_ca.index

    # Availability of choice zones
    if "util_no_attractions" in x_ca_1:
        av = (
            x_ca_1["util_no_attractions"]
            .apply(lambda x: False if x == 1 else True)
            .astype(np.int8)
        )
    elif "@df['size_term']==0" in x_ca_1:
        av = (
            x_ca_1["@df['size_term']==0"]
            .apply(lambda x: False if x == 1 else True)
            .astype(np.int8)
        )
    else:
        av = 1

    d = DataFrames(co=x_co, ca=x_ca_1, av=av)

    m = Model(dataservice=d)
    if len(spec.columns) == 4 and all(
        spec.columns == ["Label", "Description", "Expression", "coefficient"]
    ):
        m.utility_ca = linear_utility_from_spec(
            spec,
            x_col="Label",
            p_col=spec.columns[-1],
            ignore_x=("local_dist",),
        )
    elif (
        len(spec.columns) == 4
        and all(spec.columns[:3] == ["Label", "Description", "Expression"])
        and len(SEGMENT_IDS) == 1
        and spec.columns[3] == list(SEGMENT_IDS.values())[0]
    ):
        m.utility_ca = linear_utility_from_spec(
            spec,
            x_col="Label",
            p_col=spec.columns[-1],
            ignore_x=("local_dist",),
        )
    else:
        m.utility_ca = linear_utility_from_spec(
            spec,
            x_col=label_column_name,
            p_col=SEGMENT_IDS,
            ignore_x=("local_dist",),
            segment_id=CHOOSER_SEGMENT_COLUMN_NAME,
        )

    if CHOOSER_SEGMENT_COLUMN_NAME is None:
        assert len(size_spec) == 1
        m.quantity_ca = sum(
            P(f"{i}_{q}") * X(q)
            for i in size_spec.index
            for q in size_spec.columns
            if size_spec.loc[i, q] != 0
        )
    else:
        m.quantity_ca = sum(
            P(f"{i}_{q}")
            * X(q)
            * X(f"{CHOOSER_SEGMENT_COLUMN_NAME}=={str_repr(SEGMENT_IDS[i])}")
            for i in size_spec.index
            for q in size_spec.columns
            if size_spec.loc[i, q] != 0
        )

    apply_coefficients(coefficients, m, minimum=-25, maximum=25)
    apply_coefficients(size_coef, m, minimum=-6, maximum=6)

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
                settings=settings,
            ),
        )

    return m


def update_size_spec(model, data, result_dir=Path("."), output_file=None):
    master_size_spec = data.master_size_spec
    size_spec = data.size_spec
    model_selector = data.model_selector

    # Write size coefficients into size_spec
    for c in size_spec.columns:
        for i in size_spec.index:
            param_name = f"{i}_{c}"
            j = (master_size_spec["segment"] == i) & (
                master_size_spec["model_selector"] == model_selector
            )
            try:
                master_size_spec.loc[j, c] = np.exp(model.get_value(param_name))
            except KeyError:
                pass

    # Rescale each row to total 1, not mathematically needed
    # but to maintain a consistent approach from existing ASim
    master_size_spec.iloc[:, 2:] = master_size_spec.iloc[:, 2:].div(
        master_size_spec.iloc[:, 2:].sum(1), axis=0
    )

    if output_file is not None:
        os.makedirs(result_dir, exist_ok=True)
        master_size_spec.reset_index().to_csv(
            result_dir / output_file,
            index=False,
        )

    return master_size_spec


def workplace_location_model(**kwargs):
    unused = kwargs.pop("name", None)
    return location_choice_model(
        name="workplace_location",
        **kwargs,
    )


def school_location_model(**kwargs):
    unused = kwargs.pop("name", None)
    return location_choice_model(
        name="school_location",
        **kwargs,
    )


def atwork_subtour_destination_model(**kwargs):
    unused = kwargs.pop("name", None)
    return location_choice_model(
        name="atwork_subtour_destination",
        **kwargs,
    )


def joint_tour_destination_model(**kwargs):
    # goes with non_mandatory_tour_destination
    unused = kwargs.pop("name", None)
    if "coefficients_file" not in kwargs:
        kwargs["coefficients_file"] = "non_mandatory_tour_destination_coefficients.csv"
    return location_choice_model(
        name="joint_tour_destination",
        **kwargs,
    )


def non_mandatory_tour_destination_model(**kwargs):
    # goes with joint_tour_destination
    unused = kwargs.pop("name", None)
    return location_choice_model(
        name="non_mandatory_tour_destination",
        **kwargs,
    )


def trip_destination_model(**kwargs):
    unused = kwargs.pop("name", None)
    return location_choice_model(
        name="trip_destination",
        **kwargs,
    )
