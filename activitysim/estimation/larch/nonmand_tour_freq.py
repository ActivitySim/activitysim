import itertools
import logging
import os
import re
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import yaml
from larch import DataFrames, Model, P, X
from larch.log import logger_name
from larch.util import Dict

from .general import (
    apply_coefficients,
    cv_to_ca,
    linear_utility_from_spec,
    remove_apostrophes,
)

_logger = logging.getLogger(logger_name)


def interaction_simulate_data(
    name="non_mandatory_tour_frequency",
    edb_directory="output/estimation_data_bundle/{name}/",
    settings_file="{name}_model_settings.yaml",
    spec_file="{name}_SPEC.csv",
    alt_def_file="{name}_alternatives.csv",
    coefficients_files="{segment_name}/{name}_coefficients_{segment_name}.csv",
    chooser_data_files="{segment_name}/{name}_choosers_combined.csv",
    alt_values_files="{segment_name}/{name}_interaction_expression_values.csv",
):
    edb_directory = edb_directory.format(name=name)

    def _read_csv(filename, **kwargs):
        filename = filename.format(name=name)
        return pd.read_csv(os.path.join(edb_directory, filename), **kwargs)

    settings_file = settings_file.format(name=name)
    with open(os.path.join(edb_directory, settings_file), "r") as yf:
        settings = yaml.load(
            yf,
            Loader=yaml.SafeLoader,
        )

    coefficients = {}
    chooser_data = {}
    alt_values = {}

    segment_names = [s["NAME"] for s in settings["SPEC_SEGMENTS"]]

    for segment_name in segment_names:
        coefficients[segment_name] = _read_csv(
            coefficients_files.format(name=name, segment_name=segment_name),
            index_col="coefficient_name",
        )
        chooser_data[segment_name] = _read_csv(
            chooser_data_files.format(name=name, segment_name=segment_name),
        )
        alt_values[segment_name] = _read_csv(
            alt_values_files.format(name=name, segment_name=segment_name),
        )

    spec = _read_csv(
        spec_file,
    )
    spec = remove_apostrophes(spec, ["Label"])
    # alt_names = list(spec.columns[3:])
    # alt_codes = np.arange(1, len(alt_names) + 1)
    # alt_names_to_codes = dict(zip(alt_names, alt_codes))
    # alt_codes_to_names = dict(zip(alt_codes, alt_names))

    alt_def = _read_csv(alt_def_file.format(name=name), index_col=0)

    return Dict(
        edb_directory=Path(edb_directory),
        settings=settings,
        chooser_data=chooser_data,
        coefficients=coefficients,
        alt_values=alt_values,
        spec=spec,
        alt_def=alt_def,
    )


def link_same_value_coefficients(segment_names, coefficients, spec):
    # Assume all coefficients with exactly equal current values are
    # actually the same estimated coefficient value and should be
    # treated as such by Larch.  Comment out this function where called to relax
    # this assumption, although be careful about the number of unique
    # parameters to estimate in these models.

    relabel_coef = {}
    for segment_name in segment_names:
        coef_backwards_map = dict(
            [(j, i) for i, j in coefficients[segment_name]["value"].items()]
        )
        relabel_coef[segment_name] = r = coefficients[segment_name]["value"].map(
            coef_backwards_map
        )
        spec[segment_name] = spec[segment_name].map(r)
    return relabel_coef


def unavail_parameters(model):
    return model.pf.index[(model.pf.value < -900) & (model.pf.holdfast != 0)]


def unavail_data_cols(model):
    locks = unavail_parameters(model)
    return [i.data for i in model.utility_ca if i.param in locks]


def unavail(model, x_ca):
    lock_data = unavail_data_cols(model)
    if len(lock_data):
        unav = x_ca[lock_data[0]] > 0
        for j in lock_data[1:]:
            unav |= x_ca[j] > 0
    return unav


def nonmand_tour_freq_model(
    edb_directory="output/estimation_data_bundle/{name}/",
    return_data=False,
):
    data = interaction_simulate_data(
        name="non_mandatory_tour_frequency",
        edb_directory=edb_directory,
    )

    settings = data.settings
    segment_names = [s["NAME"] for s in settings["SPEC_SEGMENTS"]]
    data.relabel_coef = link_same_value_coefficients(
        segment_names, data.coefficients, data.spec
    )
    spec = data.spec
    coefficients = data.coefficients
    chooser_data = data.chooser_data
    alt_values = data.alt_values
    alt_def = data.alt_def

    m = {}
    for segment_name in segment_names:
        segment_model = m[segment_name] = Model()
        # One of the alternatives is coded as 0, so
        # we need to explicitly initialize the MNL nesting graph
        # and set to root_id to a value other than zero.
        segment_model.initialize_graph(alternative_codes=alt_def.index, root_id=9999)

        # Utility specifications
        segment_model.utility_ca = linear_utility_from_spec(
            spec,
            x_col="Label",
            p_col=segment_name,
        )
        apply_coefficients(coefficients[segment_name], segment_model)
        segment_model.choice_co_code = "override_choice"

        # Attach Data
        x_co = (
            chooser_data[segment_name]
            .set_index("person_id")
            .rename(columns={"TAZ": "HOMETAZ"})
        )
        x_ca = cv_to_ca(alt_values[segment_name].set_index(["person_id", "variable"]))
        d = DataFrames(
            co=x_co,
            ca=x_ca,
            av=~unavail(segment_model, x_ca),
        )
        m[segment_name].dataservice = d

    if return_data:
        return m, data
    return m
