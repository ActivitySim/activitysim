import numpy as np
import pandas as pd
import re
import os
import yaml
import itertools
from typing import Mapping
from larch import P, X, DataFrames, Model
from larch.model.abstract_model import AbstractChoiceModel
from larch.model.tree import NestingTree
from larch.util import Dict
from pathlib import Path

import logging
from larch.log import logger_name

_logger = logging.getLogger(logger_name)


def cv_to_ca(alt_values, dtype="float64"):
    """
    Convert a choosers-variables DataFrame to an idca DataFrame.

    Parameters
    ----------
    alt_values : pandas.DataFrame
        This DataFrame should be in choosers-variables format,
        with one row per chooser and variable, and one column per
        alternative.  The id's for the choosers and variables
        must be in that order, in a two-level MultiIndex.
    dtype : dtype
        Convert the incoming data to this type.  Set to None to
        skip data conversion.

    Returns
    -------
    pandas.DataFrame
        The resulting DataFrame is transformed into Larch's
        idca format, with one row per chooser (case) and
        alternative, and one column per variable.
    """

    # Read the source file, converting to a tall stack of
    # data with a 3 level multiindex
    x_ca_tall = alt_values.stack()

    # Set the last level index name to 'altid'
    x_ca_tall.index.rename("altid", -1, inplace=True)
    c_, v_, a_ = x_ca_tall.index.names

    # case and alt id's should be integers.
    # Easier to manipulate dtypes of a multiindex if we just convert
    # it to columns first.  If this ends up being a performance issue
    # we can look at optimizing this in the future.
    x_ca_tall = x_ca_tall.reset_index()
    x_ca_tall[c_] = x_ca_tall[c_].astype(int)
    x_ca_tall[a_] = x_ca_tall[a_].astype(int)
    x_ca_tall = x_ca_tall.set_index([c_, v_, a_])

    if dtype is not None:
        # Convert instances where data is 'False' or 'True' to numbers
        x_ca_tall[x_ca_tall == "False"] = 0
        x_ca_tall[x_ca_tall == "True"] = 1

        # Convert data to float64 to optimize computation speed in larch
        x_ca_tall = x_ca_tall.astype(dtype)

    # Unstack the variables dimension
    x_ca = x_ca_tall.unstack(1)

    # Code above added a dummy top level to columns, remove it here.
    x_ca.columns = x_ca.columns.droplevel(0)

    return x_ca


def prevent_overlapping_column_names(x_ca, x_co):
    """
    Rename columns in idca data to prevent overlapping names.

    Parameters
    ----------
    x_ca, x_co : pandas.DataFrame
        The idca and idco data, respectively

    Returns
    -------
    x_ca, x_co
    """
    renaming = {i: f"{i}_ca" for i in x_ca.columns if i in x_co.columns}
    x_ca.rename(columns=renaming, inplace=True)
    return x_ca, x_co


def linear_utility_from_spec(spec, x_col, p_col, ignore_x=(), segment_id=None):
    """
    Create a linear function from a spec DataFrame.

    Parameters
    ----------
    spec : pandas.DataFrame
        A spec for an ActivitySim model.
    x_col: str
        The name of the columns in spec representing the data.
    p_col: str or dict
        The name of the columns in spec representing the parameters.
        Give as a string for a single column, or as a dict to
        have segments on multiple columns. If given as a dict,
        the keys give the names of the columns to use, and the
        values give the identifiers that will need to match the
        loaded `segment_id` value.
    ignore_x : Collection, optional
        Labels in the spec file to ignore.  Typically this
        includes variables that are pre-processed by ActivitySim
        and therefore don't need to be made available in Larch.
    segment_id : str, optional
        The CHOOSER_SEGMENT_COLUMN_NAME identified for ActivitySim.
        This value is ignored if `p_col` is a string, and required
        if `p_col` is a dict.

    Returns
    -------
    LinearFunction_C
    """
    if isinstance(p_col, dict):
        if segment_id is None:
            raise ValueError("segment_id must be given if p_col is a dict")
        partial_utility = {}
        for seg_p_col, segval in p_col.items():
            partial_utility[seg_p_col] = linear_utility_from_spec(
                spec, x_col, seg_p_col, ignore_x,
            ) * X(f"{segment_id}=={segval}")
        return sum(partial_utility.values())
    return sum(
        P(getattr(i, p_col)) * X(getattr(i, x_col))
        for i in spec.itertuples()
        if (getattr(i, x_col) not in ignore_x) and not pd.isna(getattr(i, p_col))
    )


def dict_of_linear_utility_from_spec(spec, x_col, p_col, ignore_x=()):
    """
    Create a linear function from a spec DataFrame.

    Parameters
    ----------
    spec : pandas.DataFrame
        A spec for an ActivitySim model.
    x_col: str
        The name of the columns in spec representing the data.
    p_col: dict
        The name of the columns in spec representing the parameters.
        The keys give the names of the columns to use, and the
        values will become the keys of the output dictionary.
    ignore_x : Collection, optional
        Labels in the spec file to ignore.  Typically this
        includes variables that are pre-processed by ActivitySim
        and therefore don't need to be made available in Larch.
    segment_id : str, optional
        The CHOOSER_SEGMENT_COLUMN_NAME identified for ActivitySim.
        This value is ignored if `p_col` is a string, and required
        if `p_col` is a dict.

    Returns
    -------
    dict
    """
    utils = {}
    for altname, altcode in p_col.items():
        utils[altcode] = linear_utility_from_spec(
            spec, x_col, altname, ignore_x=ignore_x
        )
    return utils


def explicit_value_parameters_from_spec(spec, p_col, model):
    """
    Define and lock parameters given as fixed values in the spec.

    Parameters
    ----------
    spec : pandas.DataFrame
        A spec for an ActivitySim model.
    p_col : str or dict
        The name of the columns in spec representing the parameters.
        Give as a string for a single column, or as a dict to
        have segments on multiple columns. If given as a dict,
        the keys give the names of the columns to use, and the
        values give the identifiers that will need to match the
        loaded `segment_id` value.  Only the keys are used in this
        function.
    model : larch.Model
        The model to insert fixed value parameters.

    Returns
    -------

    """
    if isinstance(p_col, dict):
        for p_col_ in p_col:
            explicit_value_parameters_from_spec(spec, p_col_, model)
    else:
        for i in spec.itertuples():
            try:
                j = float(getattr(i, p_col))
            except Exception:
                pass
            else:
                model.set_value(
                    getattr(i, p_col), value=j, holdfast=True,
                )


def explicit_value_parameters(model):
    """
    Define and lock parameters given as fixed values.

    Parameters
    ----------
    model : larch.Model
        The model to insert fixed value parameters.

    Returns
    -------

    """
    for i in model.pf.index:
        try:
            j = float(i)
        except Exception:
            pass
        else:
            model.set_value(
                i,
                value=j,
                initvalue=j,
                nullvalue=j,
                minimum=j,
                maximum=j,
                holdfast=True,
            )


def apply_coefficients(coefficients, model, minimum=None, maximum=None):
    """
    Read the coefficients CSV file to a DataFrame and set model parameters.

    Parameters
    ----------
    coefficients : pandas.DataFrame
        The coefficients table in the ActivitySim data bundle
        for this model.
    model : Model
        Apply coefficient values and constraints to this model.

    """
    if isinstance(model, dict):
        for m in model.values():
            apply_coefficients(coefficients, m)
    else:
        assert isinstance(coefficients, pd.DataFrame)
        assert all(coefficients.columns == ["value", "constrain"])
        assert coefficients.index.name == "coefficient_name"
        assert isinstance(model, AbstractChoiceModel)
        explicit_value_parameters(model)
        for i in coefficients.itertuples():
            if i.Index in model:
                model.set_value(
                    i.Index,
                    value=i.value,
                    holdfast=(i.constrain == "T"),
                    minimum=minimum,
                    maximum=maximum,
                )


def apply_coef_template(linear_utility, template_col, condition=None):
    """
    Apply a coefficient template over a linear utility function.

    Parameters
    ----------
    linear_utility : LinearFunction_C
    template_col : Mapping
    condition : any

    Returns
    -------
    LinearFunction_C
    """
    result = sum(
        P("*".join(template_col.get(ip, ip) for ip in i.param.split("*")))
        * i.data
        * i.scale
        for i in linear_utility
    )
    if condition is not None:
        result = result * condition
    return result


def construct_nesting_tree(alternatives, nesting_settings):
    """
    Construct a NestingTree from ActivitySim settings.

    Parameters
    ----------
    alternatives : Mapping or Sequence
        If given as a Mapping (dict), the keys are the alternative names
        as strings, and the values are alternative code numbers to use
        in larch.  If given as a Sequence, the values are the alternative
        names, and unique sequential codes will be created starting from 1.
    nesting_settings : Mapping
        The 'NESTS' section of the ActivitySim config file.

    Returns
    -------
    NestingTree
    """
    if not isinstance(alternatives, Mapping):
        alt_names = list(alternatives)
        alt_codes = np.arange(1, len(alt_names) + 1)
        alternatives = dict(zip(alt_names, alt_codes))

    tree = NestingTree()
    nest_names_to_codes = alternatives.copy()
    nest_names_to_codes["root"] = 0
    for alt_name, alt_code in alternatives.items():
        tree.add_node(alt_code, name=alt_name)

    def make_nest(cfg, parent_code=0):
        nonlocal nest_names_to_codes
        if cfg["name"] != "root":
            if cfg["name"] not in nest_names_to_codes:
                n = tree.new_node(
                    name=cfg["name"],
                    parameter=str(cfg["coefficient"]),
                    parent=parent_code,
                )
                nest_names_to_codes[cfg["name"]] = n
            else:
                tree.add_edge(parent_code, nest_names_to_codes[cfg["name"]])
        for a in cfg["alternatives"]:
            if isinstance(a, str):
                tree.add_edge(nest_names_to_codes[cfg["name"]], nest_names_to_codes[a])
            else:
                make_nest(a, parent_code=nest_names_to_codes[cfg["name"]])

    make_nest(nesting_settings)

    return tree


def remove_apostrophes(df, from_columns=None):
    """
    Remove apostrophes from columns names and from data in given columns.

    This function operates in-place on DataFrames.

    Parameters
    ----------
    df : pandas.DataFrame
    from_columns : Collection, optional

    Returns
    -------
    pandas.DataFrame
    """
    df.columns = df.columns.str.replace("'", "")
    if from_columns:
        for c in from_columns:
            df[c] = df[c].str.replace("'", "")
    return df


def clean_values(
    values,
    alt_names,
    choice_col="override_choice",
    alt_names_to_codes=None,
    choice_code="override_choice_code",
):
    """

    Parameters
    ----------
    values : pd.DataFrame
    alt_names : Collection
    override_choice : str
        The columns of `values` containing the observed choices.
    alt_names_to_codes : Mapping, optional
        If the `override_choice` column contains alternative names,
        use this mapping to convert the names into alternative code
        numbers.
    choice_code : str, default 'override_choice_code'
        The name of the observed choice code number column that will
        be added to `values`.

    Returns
    -------
    pd.DataFrame
    """
    values = remove_apostrophes(values)
    values.fillna(0, inplace=True)
    # Retain only choosers with valid observed choice
    values = values[values[choice_col].isin(alt_names)]
    # Convert choices to code numbers
    if alt_names_to_codes is not None:
        values[choice_code] = values[choice_col].map(alt_names_to_codes)
    else:
        values[choice_code] = values[choice_col]
    return values


def simple_simulate_data(
    name="tour_mode_choice",
    edb_directory="output/estimation_data_bundle/{name}/",
    coefficients_file="{name}_coefficients.csv",
    coefficients_template="{name}_coefficients_template.csv",
    spec_file="{name}_SPEC.csv",
    settings_file="{name}_model_settings.yaml",
    chooser_data_file="{name}_values_combined.csv",
    values_index_col="tour_id",
):
    edb_directory = edb_directory.format(name=name)

    def _read_csv(filename, **kwargs):
        filename = filename.format(name=name)
        return pd.read_csv(os.path.join(edb_directory, filename), **kwargs)

    coefficients = _read_csv(coefficients_file, index_col="coefficient_name",)

    try:
        coef_template = _read_csv(coefficients_template, index_col="coefficient_name",)
    except FileNotFoundError:
        coef_template = None

    spec = _read_csv(spec_file,)
    spec = remove_apostrophes(spec, ["Label"])
    alt_names = list(spec.columns[3:])
    alt_codes = np.arange(1, len(alt_names) + 1)
    alt_names_to_codes = dict(zip(alt_names, alt_codes))
    alt_codes_to_names = dict(zip(alt_codes, alt_names))

    chooser_data = _read_csv(chooser_data_file, index_col=values_index_col,)

    settings_file = settings_file.format(name=name)
    with open(os.path.join(edb_directory, settings_file), "r") as yf:
        settings = yaml.load(yf, Loader=yaml.SafeLoader,)

    return Dict(
        edb_directory=Path(edb_directory),
        settings=settings,
        chooser_data=chooser_data,
        coefficients=coefficients,
        coef_template=coef_template,
        spec=spec,
        alt_names=alt_names,
        alt_codes=alt_codes,
        alt_names_to_codes=alt_names_to_codes,
        alt_codes_to_names=alt_codes_to_names,
    )


def update_coefficients(model, data, result_dir=Path('.'), output_file=None):
    coefficients = data.coefficients.copy()
    est_names = [j for j in coefficients.index if j in model.pf.index]
    coefficients.loc[est_names, "value"] = model.pf.loc[est_names, "value"]
    if output_file is not None:
        os.makedirs(result_dir, exist_ok=True)
        coefficients.reset_index().to_csv(
            result_dir/output_file,
            index=False,
        )
    return coefficients
