import importlib
import itertools
import logging
import os
import re
from pathlib import Path

import larch
import numpy as np
import pandas as pd
import yaml
from larch import DataFrames, Model, P, X
from larch.log import logger_name
from larch.model.model_group import ModelGroup
from larch.util import Dict

from ...abm.models.util import cdap
from .general import apply_coefficients, explicit_value_parameters

_logger = logging.getLogger(logger_name)


def generate_alternatives(n_persons):
    """
    Generate a dictionary of CDAP alternatives.

    The keys are the names of the patterns, and
    the values are the alternative code numbers.

    Parameters
    ----------
    n_persons : int

    Returns
    -------
    Dict
    """
    basic_patterns = ["M", "N", "H"]
    alt_names = list(
        "".join(i) for i in itertools.product(basic_patterns, repeat=n_persons)
    )
    alt_codes = np.arange(1, len(alt_names) + 1)
    return Dict(zip(alt_names, alt_codes))


def apply_replacements(expression, prefix, tokens):
    """
    Convert general person terms to specific person terms for the CDAP model.

    Parameters
    ----------
    expression : str
        An expression from the "Expression" column
        of cdap_INDIV_AND_HHSIZE1_SPEC.csv, or similar.
    prefix : str
        A prefix to attach to each token in `expression`.
    tokens : list-like of str
        A list of tokens to edit within an expression.

    Returns
    -------
    expression : str
        The modified expression
    """
    for i in tokens:
        expression = re.sub(rf"\b{i}\b", f"{prefix}_{i}", expression)
    return expression


def cdap_base_utility_by_person(model, n_persons, spec, alts=None, value_tokens=()):
    """
    Build the base utility by person for each pattern.

    Parameters
    ----------
    model : larch.Model
    n_persons : int
    spec : pandas.DataFrame
        The base utility by person spec provided by
        the ActivitySim framework.
    alts : dict, optional
        The keys are the names of the patterns, and
        the values are the alternative code numbers,
        as created by `generate_alternatives`.  If not
        given, the alts are automatically regenerated
        using that function.
    value_tokens : list-like of str, optional
        A list of tokens to edit within an the expressions,
        generally the column names of the provided values
        from the estimation data bundle.  Only used when
        `n_persons` is more than 1.
    """
    if n_persons == 1:
        for i in spec.index:
            if not pd.isna(spec.loc[i, "M"]):
                model.utility_co[1] += X(spec.Expression[i]) * P(spec.loc[i, "M"])
            if not pd.isna(spec.loc[i, "N"]):
                model.utility_co[2] += X(spec.Expression[i]) * P(spec.loc[i, "N"])
            if not pd.isna(spec.loc[i, "H"]):
                model.utility_co[3] += X(spec.Expression[i]) * P(spec.loc[i, "H"])
    else:
        if alts is None:
            alts = generate_alternatives(n_persons)
        person_numbers = range(1, n_persons + 1)
        for pnum in person_numbers:
            for i in spec.index:
                for aname, anum in alts.items():
                    z = pnum - 1
                    if not pd.isna(spec.loc[i, aname[z]]):
                        x = apply_replacements(
                            spec.Expression[i], f"p{pnum}", value_tokens
                        )
                        model.utility_co[anum] += X(x) * P(spec.loc[i, aname[z]])


def interact_pattern(n_persons, select_persons, tag):
    """
    Compile a regex pattern to match CDAP alternatives.

    Parameters
    ----------
    n_persons : int
    select_persons : list-like of int
        The persons to be selected.
    tag : str
        The activity letter, currently one of {M,N,H}.

    Returns
    -------
    re.compile
    """

    pattern = ""
    p = 1
    while len(pattern) < n_persons:
        pattern += tag if p in select_persons else "."
        p += 1
    return re.compile(pattern)


def cdap_interaction_utility(model, n_persons, alts, interaction_coef, coefficients):

    person_numbers = list(range(1, n_persons + 1))

    matcher = re.compile("coef_[HMN]_.*")
    interact_coef_map = {}
    for c in coefficients.index:
        if matcher.search(c):
            c_split = c.split("_")
            for j in c_split[2:]:
                interact_coef_map[(c_split[1], j)] = c
                if all((i == "x" for i in j)):  # wildcards also map to empty
                    interact_coef_map[(c_split[1], "")] = c

    for (cardinality, activity), coefs in interaction_coef.groupby(
        ["cardinality", "activity"]
    ):
        _logger.info(
            f"{n_persons} person households, interaction cardinality {cardinality}, activity {activity}"
        )
        if cardinality > n_persons:
            continue
        elif cardinality == n_persons:
            this_aname = activity * n_persons
            this_altnum = alts[this_aname]
            for rowindex, row in coefs.iterrows():
                expression = "&".join(
                    f"(p{p}_ptype == {t})"
                    for (p, t) in zip(person_numbers, row.interaction_ptypes)
                    if t != "*"
                )
                if expression:
                    if (activity, row.interaction_ptypes) in interact_coef_map:
                        linear_component = X(expression) * P(
                            interact_coef_map[(activity, row.interaction_ptypes)]
                        )
                    else:
                        linear_component = X(expression) * P(row.coefficient)
                else:
                    if (activity, row.interaction_ptypes) in interact_coef_map:
                        linear_component = P(
                            interact_coef_map[(activity, row.interaction_ptypes)]
                        )
                    else:
                        linear_component = P(row.coefficient)
                _logger.debug(
                    f"utility_co[{this_altnum} {this_aname}] += {linear_component}"
                )
                model.utility_co[this_altnum] += linear_component
        elif cardinality < n_persons:
            for combo in itertools.combinations(person_numbers, cardinality):
                pattern = interact_pattern(n_persons, combo, activity)
                for aname, anum in alts.items():
                    if pattern.match(aname):
                        for rowindex, row in coefs.iterrows():
                            expression = "&".join(
                                f"(p{p}_ptype == {t})"
                                for (p, t) in zip(combo, row.interaction_ptypes)
                                if t != "*"
                            )
                            # interaction terms without ptypes (i.e. with wildcards)
                            # only apply when the household size matches the cardinality
                            if expression != "":
                                if (
                                    activity,
                                    row.interaction_ptypes,
                                ) in interact_coef_map:
                                    linear_component = X(expression) * P(
                                        interact_coef_map[
                                            (activity, row.interaction_ptypes)
                                        ]
                                    )
                                else:
                                    linear_component = X(expression) * P(
                                        row.coefficient
                                    )
                                _logger.debug(
                                    f"utility_co[{anum} {aname}] += {linear_component}"
                                )
                                model.utility_co[anum] += linear_component


def cdap_split_data(households, values):
    if "cdap_rank" not in values:
        raise ValueError("assign cdap_rank to values first")
    # only process the first 5 household members
    values = values[values.cdap_rank <= 5]
    cdap_data = {}
    for hhsize, hhs_part in households.groupby(households.hhsize.clip(1, 5)):
        if hhsize == 1:
            v = pd.merge(values, hhs_part.household_id, on="household_id").set_index(
                "household_id"
            )
        else:
            v = (
                pd.merge(values, hhs_part.household_id, on="household_id")
                .set_index(["household_id", "cdap_rank"])
                .unstack()
            )
            v.columns = [f"p{i[1]}_{i[0]}" for i in v.columns]
            for agglom in ["override_choice", "model_choice"]:
                v[agglom] = v[[f"p{p}_{agglom}" for p in range(1, hhsize + 1)]].sum(1)
        cdap_data[hhsize] = v
    return cdap_data


def cdap_dataframes(households, values):
    data = cdap_split_data(households, values)
    dfs = {}
    for hhsize in data.keys():
        alts = generate_alternatives(hhsize)
        dfs[hhsize] = DataFrames(
            co=data[hhsize],
            alt_names=alts.keys(),
            alt_codes=alts.values(),
            av=1,
            ch=data[hhsize].override_choice.map(alts),
        )
    return dfs


# def _cdap_model(households, values, spec1, interaction_coef, coefficients):
#     cdap_data = cdap_dataframes(households, values)
#     m = {}
#     _logger.info(f"building for model 1")
#     m[1] = Model(dataservice=cdap_data[1])
#     cdap_base_utility_by_person(m[1], n_persons=1, spec=spec1)
#     m[1].choice_any = True
#     m[1].availability_any = True
#
#     # Add cardinality into interaction_coef if not present
#     if 'cardinality' not in interaction_coef:
#         interaction_coef['cardinality'] = interaction_coef['interaction_ptypes'].str.len()
#     for s in [2, 3, 4, 5]:
#         _logger.info(f"building for model {s}")
#         m[s] = Model(dataservice=cdap_data[s])
#         alts = generate_alternatives(s)
#         cdap_base_utility_by_person(m[s], s, spec1, alts, values.columns)
#         cdap_interaction_utility(m[s], s, alts, interaction_coef, coefficients)
#         m[s].choice_any = True
#         m[s].availability_any = True
#
#     result = ModelGroup(m.values())
#     explicit_value_parameters(result)
#     apply_coefficients(coefficients, result)
#     return result


def cdap_data(
    name="cdap",
    edb_directory="output/estimation_data_bundle/{name}/",
    coefficients_file="{name}_coefficients.csv",
    interaction_coeffs_file="{name}_interaction_coefficients.csv",
    households_file="../../final_households.csv",
    persons_file="../../final_persons.csv",
    spec1_file="{name}_INDIV_AND_HHSIZE1_SPEC.csv",
    settings_file="{name}_model_settings.yaml",
    chooser_data_file="{name}_values_combined.csv",
):
    edb_directory = edb_directory.format(name=name)
    if not os.path.exists(edb_directory):
        raise FileNotFoundError(edb_directory)

    def read_csv(filename, **kwargs):
        filename = filename.format(name=name)
        return pd.read_csv(os.path.join(edb_directory, filename), **kwargs)

    def read_yaml(filename, **kwargs):
        filename = filename.format(name=name)
        with open(os.path.join(edb_directory, filename), "rt") as f:
            return yaml.load(f, Loader=yaml.SafeLoader, **kwargs)

    settings = read_yaml(settings_file)

    try:
        hhs = read_csv(households_file)
    except FileNotFoundError:
        hhs = pd.read_csv(households_file)

    try:
        persons = read_csv(persons_file)
    except FileNotFoundError:
        persons = pd.read_csv(persons_file)

    person_type_map = settings.get("PERSON_TYPE_MAP")
    if person_type_map is None:
        raise KeyError("PERSON_TYPE_MAP missing from cdap_settings.yaml")

    person_rank = cdap.assign_cdap_rank(persons, person_type_map)

    coefficients = read_csv(
        coefficients_file,
        index_col="coefficient_name",
        comment="#",
    )

    interaction_coef = read_csv(
        interaction_coeffs_file,
        dtype={"interaction_ptypes": str},
        keep_default_na=False,
        comment="#",
    )

    spec1 = read_csv(spec1_file, comment="#")
    values = read_csv(chooser_data_file, comment="#")
    values["cdap_rank"] = person_rank

    return Dict(
        edb_directory=Path(edb_directory),
        person_data=values,
        spec1=spec1,
        interaction_coef=interaction_coef,
        coefficients=coefficients,
        households=hhs,
        settings=settings,
    )


def cdap_model(
    edb_directory="output/estimation_data_bundle/{name}/",
    coefficients_file="{name}_coefficients.csv",
    interaction_coeffs_file="{name}_interaction_coefficients.csv",
    households_file="../../final_households.csv",
    persons_file="../../final_persons.csv",
    spec1_file="{name}_INDIV_AND_HHSIZE1_SPEC.csv",
    settings_file="{name}_model_settings.yaml",
    chooser_data_file="{name}_values_combined.csv",
    return_data=False,
):
    d = cdap_data(
        name="cdap",
        edb_directory=edb_directory,
        coefficients_file=coefficients_file,
        interaction_coeffs_file=interaction_coeffs_file,
        households_file=households_file,
        persons_file=persons_file,
        spec1_file=spec1_file,
        settings_file=settings_file,
        chooser_data_file=chooser_data_file,
    )

    households = d.households
    values = d.person_data
    spec1 = d.spec1
    interaction_coef = d.interaction_coef
    coefficients = d.coefficients

    cdap_dfs = cdap_dataframes(households, values)
    m = {}
    _logger.info(f"building for model 1")
    m[1] = Model(dataservice=cdap_dfs[1])
    cdap_base_utility_by_person(m[1], n_persons=1, spec=spec1)
    m[1].choice_any = True
    m[1].availability_any = True

    # Add cardinality into interaction_coef if not present
    if "cardinality" not in interaction_coef:
        interaction_coef["cardinality"] = interaction_coef[
            "interaction_ptypes"
        ].str.len()
    for s in [2, 3, 4, 5]:
        _logger.info(f"building for model {s}")
        m[s] = Model(dataservice=cdap_dfs[s])
        alts = generate_alternatives(s)
        cdap_base_utility_by_person(m[s], s, spec1, alts, values.columns)
        cdap_interaction_utility(m[s], s, alts, interaction_coef, coefficients)
        m[s].choice_any = True
        m[s].availability_any = True

    model = ModelGroup(m.values())
    explicit_value_parameters(model)
    apply_coefficients(coefficients, model)
    if return_data:
        return model, d
    return model
