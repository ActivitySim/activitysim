import os
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from larch.util import Dict
from larch import Model, DataFrames

from .general import (
    remove_apostrophes,
    dict_of_linear_utility_from_spec,
    apply_coefficients,
    construct_nesting_tree,
)


def construct_availability(model, chooser_data, alt_codes_to_names):
    """
    Construct an availability dataframe based on -999 parameters.

    Parameters
    ----------
    model : larch.Model
    chooser_data : pandas.DataFrame
    alt_codes_to_names : Mapping[int,str]

    Returns
    -------
    pandas.DataFrame
    """
    avail = {}
    for acode, aname in alt_codes_to_names.items():
        unavail_cols = list(
            (
                chooser_data[i.data]
                if i.data in chooser_data
                else chooser_data.eval(i.data)
            )
            for i in model.utility_co[acode]
            if i.param == "-999"
        )
        if len(unavail_cols):
            avail[acode] = sum(unavail_cols) == 0
        else:
            avail[acode] = 1
    avail = pd.DataFrame(avail).astype(np.int8)
    avail.index = chooser_data.index
    return avail


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

    settings_file = settings_file.format(name=name)
    with open(os.path.join(edb_directory, settings_file), "r") as yf:
        settings = yaml.load(yf, Loader=yaml.SafeLoader,)

    try:
        coefficients = _read_csv(coefficients_file, index_col="coefficient_name",)

        try:
            coef_template = _read_csv(coefficients_template, index_col="coefficient_name",)
        except FileNotFoundError:
            coef_template = None

        spec = _read_csv(spec_file, comment="#")
        spec = remove_apostrophes(spec, ["Label"])

        # remove temp rows from spec, ASim uses them to calculate the other values written
        # to the EDB, but they are not actually part of the utility function themselves.
        spec = spec.loc[~spec.Expression.isna()]
        spec = spec.loc[~spec.Expression.str.startswith("_")].copy()

        alt_names = list(spec.columns[3:])
        alt_codes = np.arange(1, len(alt_names) + 1)
        alt_names_to_codes = dict(zip(alt_names, alt_codes))
        alt_codes_to_names = dict(zip(alt_codes, alt_names))

        chooser_data = _read_csv(chooser_data_file, index_col=values_index_col,)

    except Exception:
        # when an error happens in reading anything other than settings, print settings
        from pprint import pprint
        pprint(settings)
        raise

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


def simple_simulate_model(
        name,
        edb_directory="output/estimation_data_bundle/{name}/",
        return_data=False,
        choices=None,
        construct_avail=False,
        values_index_col="household_id",
):
    data = simple_simulate_data(
        name=name, edb_directory=edb_directory, values_index_col=values_index_col,
    )
    coefficients = data.coefficients
    # coef_template = data.coef_template # not used
    spec = data.spec
    chooser_data = data.chooser_data
    settings = data.settings

    alt_names = data.alt_names
    alt_codes = data.alt_codes

    from .general import clean_values
    chooser_data = clean_values(
        chooser_data,
        alt_names_to_codes=choices or data.alt_names_to_codes,
        choice_code="override_choice_code",
    )

    if settings.get('LOGIT_TYPE') == 'NL':
        tree = construct_nesting_tree(data.alt_names, settings["NESTS"])
        m = Model(graph=tree)
    else:
        m = Model(alts=data.alt_codes_to_names)

    m.utility_co = dict_of_linear_utility_from_spec(
        spec, "Label", dict(zip(alt_names, alt_codes)),
    )

    apply_coefficients(coefficients, m)

    if construct_avail:
        avail = construct_availability(m, chooser_data, data.alt_codes_to_names)
    else:
        avail = True

    d = DataFrames(co=chooser_data, av=avail, alt_codes=alt_codes, alt_names=alt_names, )

    m.dataservice = d
    m.choice_co_code = "override_choice_code"

    if return_data:
        return (
            m,
            Dict(
                edb_directory=data.edb_directory,
                chooser_data=chooser_data,
                coefficients=coefficients,
                spec=spec,
                alt_names=alt_names,
                alt_codes=alt_codes,
                settings=settings,
            ),
        )

    return m


def auto_ownership_model(
    name="auto_ownership",
    edb_directory="output/estimation_data_bundle/{name}/",
    return_data=False,
):
    return simple_simulate_model(
        name=name,
        edb_directory=edb_directory,
        return_data=return_data,
        choices={i: i+1 for i in range(5)},  # choices are coded in data as integers,
                                             # not 'cars0' etc as appears in the spec
    )


def free_parking_model(
    name="free_parking",
    edb_directory="output/estimation_data_bundle/{name}/",
    return_data=False,
):
    return simple_simulate_model(
        name=name,
        edb_directory=edb_directory,
        return_data=return_data,
        choices={True: 1, False: 2},  # True is free parking, False is paid parking, names match spec positions
    )


def mandatory_tour_frequency_model(
    name="mandatory_tour_frequency",
    edb_directory="output/estimation_data_bundle/{name}/",
    return_data=False,
):
    return simple_simulate_model(
        name=name,
        edb_directory=edb_directory,
        return_data=return_data,
    )


def joint_tour_frequency_model(
    name="joint_tour_frequency",
    edb_directory="output/estimation_data_bundle/{name}/",
    return_data=False,
):
    return simple_simulate_model(
        name=name,
        edb_directory=edb_directory,
        return_data=return_data,
    )


def atwork_subtour_frequency_model(
    name="atwork_subtour_frequency",
    edb_directory="output/estimation_data_bundle/{name}/",
    return_data=False,
):
    return simple_simulate_model(
        name=name,
        edb_directory=edb_directory,
        values_index_col="tour_id",
        return_data=return_data,
    )


def joint_tour_composition_model(
    name="joint_tour_composition",
    edb_directory="output/estimation_data_bundle/{name}/",
    return_data=False,
):
    return simple_simulate_model(
        name=name,
        edb_directory=edb_directory,
        return_data=return_data,
    )


def joint_tour_participation_model(
    name="joint_tour_participation",
    edb_directory="output/estimation_data_bundle/{name}/",
    return_data=False,
):
    return simple_simulate_model(
        name=name,
        edb_directory=edb_directory,
        return_data=return_data,
        values_index_col="participant_id",
        choices={
            0: 1,  # 0 means participate, alternative 1
            1: 2,  # 1 means not participate, alternative 2
        },
    )
