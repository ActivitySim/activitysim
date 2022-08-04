# ActivitySim
# See full license in LICENSE.txt.

import itertools
import logging
import os

import numpy as np
import pandas as pd

from activitysim.core import (
    assign,
    config,
    expressions,
    inject,
    logit,
    los,
    pipeline,
    simulate,
    tracing,
)
from activitysim.core.interaction_simulate import interaction_simulate
from activitysim.core.util import assign_in_place

from .util import estimation

logger = logging.getLogger(__name__)


def append_probabilistic_vehtype_type_choices(choices, model_settings, trace_label):
    """
    Select a fuel type for the provided body type and age of the vehicle.

    Make probabilistic choices based on the `PROBS_SPEC` file.

    Parameters
    ----------
    choices : pandas.DataFrame
        selection of {body_type}_{age} to append vehicle type to
    probs_spec_file : str
    trace_label : str

    Returns
    -------
    choices : pandas.DataFrame
        table of chosen vehicle types
    """
    probs_spec_file = model_settings.get("PROBS_SPEC", None)
    probs_spec = pd.read_csv(config.config_file_path(probs_spec_file), comment="#")

    fleet_year = model_settings.get("FLEET_YEAR")
    probs_spec["age"] = (1 + fleet_year - probs_spec["vehicle_year"]).astype(int)
    probs_spec["vehicle_type"] = (
        probs_spec[["body_type", "age"]].astype(str).agg("_".join, axis=1)
    )

    # left join vehicles to probs
    choosers = pd.merge(
        choices.reset_index(), probs_spec, on="vehicle_type", how="left", indicator=True
    ).set_index("vehicle_id")

    # checking to make sure all alternatives have probabilities
    missing_alts = choosers.loc[choosers._merge == "left_only", ["vehicle_type"]]
    assert (
        len(missing_alts) == 0
    ), f"missing probabilities for alternatives:\n {missing_alts}"

    # chooser columns here should just be the fuel type probabilities
    non_prob_cols = ["_merge", "vehicle_type", "body_type", "age", "vehicle_year"]
    choosers.drop(columns=non_prob_cols, inplace=True)

    # probs should sum to 1 with residual probs resulting in choice of 'fail'
    chooser_probs = choosers.div(choosers.sum(axis=1), axis=0).fillna(0)
    chooser_probs["fail"] = 1 - chooser_probs.sum(axis=1).clip(0, 1)

    # make probabilistic choices
    prob_choices, rands = logit.make_choices(
        chooser_probs, trace_label=trace_label, trace_choosers=choosers
    )

    # convert alt choice index to vehicle type attribute
    prob_choices = chooser_probs.columns[prob_choices.values].to_series(
        index=prob_choices.index
    )
    failed = prob_choices == chooser_probs.columns.get_loc("fail")
    prob_choices = prob_choices.where(~failed, "NOT CHOSEN")

    # add new attribute to logit choice vehicle types
    choices["vehicle_type"] = choices["vehicle_type"] + "_" + prob_choices

    return choices


def annotate_vehicle_type_choice_households(model_settings, trace_label):
    """
    Add columns to the households table in the pipeline according to spec.

    Parameters
    ----------
    model_settings : dict
    trace_label : str
    """
    households = inject.get_table("households").to_frame()
    expressions.assign_columns(
        df=households,
        model_settings=model_settings.get("annotate_households"),
        trace_label=tracing.extend_trace_label(trace_label, "annotate_households"),
    )
    pipeline.replace_table("households", households)


def annotate_vehicle_type_choice_persons(model_settings, trace_label):
    """
    Add columns to the persons table in the pipeline according to spec.

    Parameters
    ----------
    model_settings : dict
    trace_label : str
    """
    persons = inject.get_table("persons").to_frame()
    expressions.assign_columns(
        df=persons,
        model_settings=model_settings.get("annotate_persons"),
        trace_label=tracing.extend_trace_label(trace_label, "annotate_persons"),
    )
    pipeline.replace_table("persons", households)


def annotate_vehicle_type_choice_vehicles(model_settings, trace_label):
    """
    Add columns to the vehicles table in the pipeline according to spec.

    Parameters
    ----------
    model_settings : dict
    trace_label : str
    """
    vehicles = inject.get_table("vehicles").to_frame()
    expressions.assign_columns(
        df=vehicles,
        model_settings=model_settings.get("annotate_vehicles"),
        trace_label=tracing.extend_trace_label(trace_label, "annotate_vehicles"),
    )
    pipeline.replace_table("vehicles", vehicles)


def get_combinatorial_vehicle_alternatives(alts_cats_dict):
    """
    Build a pandas dataframe containing columns for each vehicle alternative.

    Rows will correspond to the alternative number and will be 0 except for the
    1 in the column corresponding to that alternative.

    Parameters
    ----------
    alts_cats_dict : dict
    model_settings : dict

    Returns
    -------
    alts_wide : pd.DataFrame in wide format expanded using pandas get_dummies function
    alts_long : pd.DataFrame in long format
    """
    cat_cols = list(alts_cats_dict.keys())  # e.g. fuel type, body type, age
    alts_long = pd.DataFrame(
        list(itertools.product(*alts_cats_dict.values())), columns=alts_cats_dict.keys()
    ).astype(str)
    alts_wide = pd.get_dummies(alts_long)  # rows will sum to num_cats

    alts_wide = pd.concat([alts_wide, alts_long], axis=1)
    return alts_wide, alts_long


def construct_model_alternatives(model_settings, alts_cats_dict, vehicle_type_data):
    """
    Construct the table of vehicle type alternatives.

    Vehicle type data is joined to the alternatives table for use in utility expressions.

    Parameters
    ----------
    model_settings : dict
    alts_cats_dict : dict
        nested dictionary of vehicle body, age, and fuel options
    vehicle_type_data : pandas.DataFrame

    Returns
    -------
    alts_wide : pd.DataFrame
        includes column indicators and data for each alternative
    alts_long : pd.DataFrame
        rows just list the alternatives
    """
    probs_spec_file = model_settings.get("PROBS_SPEC", None)
    if probs_spec_file:
        # do not include alternatives from fuel_type if they are given probabilisticly
        del alts_cats_dict["fuel_type"]
    alts_wide, alts_long = get_combinatorial_vehicle_alternatives(alts_cats_dict)

    # merge vehicle type data to alternatives if data is provided
    if (vehicle_type_data is not None) and (probs_spec_file is None):

        alts_wide = pd.merge(
            alts_wide,
            vehicle_type_data,
            how="left",
            on=["body_type", "fuel_type", "age"],
            indicator=True,
        )

        # checking to make sure all alternatives have data
        missing_alts = alts_wide.loc[
            alts_wide._merge == "left_only", ["body_type", "fuel_type", "age"]
        ]

        assert (
            len(missing_alts) == 0
        ), f"missing vehicle data for alternatives:\n {missing_alts}"
        alts_wide.drop(columns="_merge", inplace=True)

    # converting age to integer to allow interactions in utilities
    alts_wide["age"] = alts_wide["age"].astype(int)

    # store alts in primary configs dir for inspection
    configs_dirs = inject.get_injectable("configs_dir")
    configs_dirs = configs_dirs if isinstance(configs_dirs, list) else [configs_dirs]

    if model_settings.get("WRITE_OUT_ALTS_FILE", False):
        alts_wide.to_csv(
            os.path.join(configs_dirs[0]), "vehicle_type_choice_aternatives.csv"
        )

    return alts_wide, alts_long


def get_vehicle_type_data(model_settings, vehicle_type_data_file):
    """
    Read in the vehicle type data and computes the vehicle age.

    Parameters
    ----------
    model_settings : dict
    vehicle_type_data_file : str
        name of vehicle type data file found in config folder

    Returns
    -------
    vehicle_type_data : pandas.DataFrame
        table of vehicle type data with required body_type, age, and fuel_type columns
    """
    vehicle_type_data = pd.read_csv(
        config.config_file_path(vehicle_type_data_file), comment="#"
    )
    fleet_year = model_settings.get("FLEET_YEAR")

    vehicle_type_data["age"] = (
        1 + fleet_year - vehicle_type_data["vehicle_year"]
    ).astype(str)
    vehicle_type_data["vehicle_type"] = (
        vehicle_type_data[["body_type", "age", "fuel_type"]]
        .astype(str)
        .agg("_".join, axis=1)
    )

    return vehicle_type_data


def iterate_vehicle_type_choice(
    vehicles_merged,
    model_settings,
    model_spec,
    locals_dict,
    estimator,
    chunk_size,
    trace_label,
):
    """
    Select vehicle type for each household vehicle sequentially.

    Iterate through household vehicle numbers and select a vehicle type of
    the form {body_type}_{age}_{fuel_type}. The preprocessor is run for each
    iteration on the entire chooser table, not just the one for the current
    vehicle number.  This allows for computation of terms involving the presence
    of other household vehicles.

    Vehicle type data is read in according to the specification and joined to
    the alternatives. It can optionally be included in the output vehicles table
    by specifying the `COLS_TO_INCLUDE_IN_VEHICLE_TABLE` option in the model yaml.

    Parameters
    ----------
    vehicles_merged : orca.DataFrameWrapper
        vehicle list owned by each household merged with households table
    model_settings : dict
        yaml model settings file as dict
    model_spec : pandas.DataFrame
        omnibus spec file with expressions in index and one column per segment
    locals_dict : dict
        additional variables available when writing expressions
    estimator : Estimator object
    chunk_size : orca.injectable
    trace_label : str

    Returns
    -------
    all_choices : pandas.DataFrame
        single table of selected vehicle types and associated data
    all_choosers : pandas.DataFrame
        single table of chooser data with preprocessor variables included
    """
    # - model settings
    nest_spec = config.get_logit_model_settings(model_settings)
    vehicle_type_data_file = model_settings.get("VEHICLE_TYPE_DATA_FILE", None)
    probs_spec_file = model_settings.get("PROBS_SPEC", None)
    alts_cats_dict = model_settings.get("combinatorial_alts", False)

    # adding vehicle type data to be available to locals_dict regardless of option
    if vehicle_type_data_file:
        vehicle_type_data = get_vehicle_type_data(
            model_settings, vehicle_type_data_file
        )
        locals_dict.update({"vehicle_type_data": vehicle_type_data})

    # - Preparing alternatives
    # create alts on-the-fly as cartesian product of categorical values
    if alts_cats_dict:
        # do not include fuel types as alternatives if probability file is supplied
        alts_wide, alts_long = construct_model_alternatives(
            model_settings, alts_cats_dict, vehicle_type_data
        )

    # - preparing choosers for iterating
    vehicles_merged = vehicles_merged.to_frame()
    vehicles_merged["already_owned_veh"] = ""
    logger.info("Running %s with %d vehicles", trace_label, len(vehicles_merged))
    all_choosers = []
    all_choices = []

    # - Selecting each vehicle in the household sequentially
    # This is necessary to determine and use utility terms that include other
    #   household vehicles
    for veh_num in range(1, vehicles_merged.vehicle_num.max() + 1):
        # - preprocessor
        # running preprocessor on entire vehicle table to enumerate vehicle types
        # already owned by the household
        choosers = vehicles_merged
        preprocessor_settings = model_settings.get("preprocessor", None)
        if preprocessor_settings:
            expressions.assign_columns(
                df=choosers,
                model_settings=preprocessor_settings,
                locals_dict=locals_dict,
                trace_label=trace_label,
            )

        # only make choices for vehicles that have not been selected yet
        choosers = choosers[choosers["vehicle_num"] == veh_num]
        logger.info(
            "Running %s for vehicle number %s with %d vehicles",
            trace_label,
            veh_num,
            len(choosers),
        )

        # if there were so many alts that they had to be created programmatically,
        # by combining categorical variables, then the utility expressions should make
        # use of interaction terms to accommodate alt-specific coefficients and constants
        simulation_type = model_settings.get("SIMULATION_TYPE", "interaction_simulate")
        assert (simulation_type == "interaction_simulate") or (
            simulation_type == "simple_simulate"
        ), "SIMULATION_TYPE needs to be interaction_simulate or simple_simulate"

        log_alt_losers = config.setting("log_alt_losers", False)

        if simulation_type == "interaction_simulate":
            assert (
                alts_cats_dict is not None
            ), "Need to supply combinatorial_alts in yaml"

            choices = interaction_simulate(
                choosers=choosers,
                alternatives=alts_wide,
                spec=model_spec,
                log_alt_losers=log_alt_losers,
                locals_d=locals_dict,
                chunk_size=chunk_size,
                trace_label=trace_label,
                trace_choice_name="vehicle_type",
                estimator=estimator,
            )

        # otherwise, "simple simulation" should suffice, with a model spec that enumerates
        # each alternative as a distinct column in the .csv
        elif simulation_type == "simple_simulate":
            choices = simulate.simple_simulate(
                choosers=choosers,
                spec=model_spec,
                log_alt_losers=log_alt_losers,
                nest_spec=nest_spec,
                locals_d=locals_dict,
                chunk_size=chunk_size,
                trace_label=trace_label,
                trace_choice_name="vehicle_type",
                estimator=estimator,
            )

        if isinstance(choices, pd.Series):
            choices = choices.to_frame("choice")

        choices.rename(columns={"choice": "vehicle_type"}, inplace=True)

        if alts_cats_dict:
            alts = (
                alts_long[alts_long.columns]
                .apply(lambda row: "_".join(row.values.astype(str)), axis=1)
                .values
            )
        else:
            alts = model_spec.columns
        choices["vehicle_type"] = choices["vehicle_type"].map(dict(enumerate(alts)))

        # STEP II: append probabilistic vehicle type attributes
        if probs_spec_file is not None:
            choices = append_probabilistic_vehtype_type_choices(
                choices, model_settings, trace_label
            )

        vehicles_merged.loc[choices.index, "already_owned_veh"] = choices[
            "vehicle_type"
        ]
        all_choices.append(choices)
        all_choosers.append(choosers)

    # adding all vehicle numbers to one dataframe
    all_choices = pd.concat(all_choices)
    all_choosers = pd.concat(all_choosers)

    # appending vehicle type data to the vehicle table
    additional_cols = model_settings.get("COLS_TO_INCLUDE_IN_VEHICLE_TABLE")
    if additional_cols:
        additional_cols.append("vehicle_type")
        all_choices = (
            all_choices.reset_index()
            .merge(vehicle_type_data[additional_cols], how="left", on="vehicle_type")
            .set_index("vehicle_id")
        )

    return all_choices, all_choosers


@inject.step()
def vehicle_type_choice(
    persons, households, vehicles, vehicles_merged, chunk_size, trace_hh_id
):
    """Assign a vehicle type to each vehicle in the `vehicles` table.

    If a "SIMULATION_TYPE" is set to simple_simulate in the
    vehicle_type_choice.yaml config file, then the model specification .csv file
    should contain one column of coefficients for each distinct alternative. This
    format corresponds to ActivitySim's :func:`activitysim.core.simulate.simple_simulate`
    format. Otherwise, this model will construct a table of alternatives, at run time,
    based on all possible combinations of values of the categorical variables enumerated
    as "combinatorial_alts" in the .yaml config. In this case, the model leverages
    ActivitySim's :func:`activitysim.core.interaction_simulate` model design, in which
    the model specification .csv has only one column of coefficients, and the utility
    expressions can turn coefficients on or off based on attributes of either
    the chooser _or_ the alternative.

    As an optional second step, the user may also specify a "PROBS_SPEC" .csv file in
    the main .yaml config, corresponding to a lookup table of additional vehicle
    attributes and probabilities to be sampled and assigned to vehicles after the logit
    choices have been made. The rows of the "PROBS_SPEC" file must include all body type
    and vehicle age choices assigned in the logit model. These additional attributes are
    concatenated with the selected alternative from the logit model to form a single
    vehicle type name to be stored in the `vehicles` table as the vehicle_type column.

    Only one household vehicle is selected at a time to allow for the introduction of
    owned vehicle related attributes. For example, a household may be less likely to
    own a second van if they already own one. The model is run sequentially through
    household vehicle numbers. The preprocessor is run for each iteration on the entire
    vehicles table to allow for computation of terms involving the presence of other
    household vehicles.

    The user may also augment the `households` or `persons` tables with new vehicle
    type-based fields specified via expressions in "annotate_households_vehicle_type.csv"
    and "annotate_persons_vehicle_type.csv", respectively.

    Parameters
    ----------
    persons : orca.DataFrameWrapper
    households : orca.DataFrameWrapper
    vehicles : orca.DataFrameWrapper
    vehicles_merged : orca.DataFrameWrapper
    chunk_size : orca.injectable
    trace_hh_id : orca.injectable
    """
    trace_label = "vehicle_type_choice"
    model_settings_file_name = "vehicle_type_choice.yaml"
    model_settings = config.read_model_settings(model_settings_file_name)

    estimator = estimation.manager.begin_estimation("vehicle_type")

    model_spec_raw = simulate.read_model_spec(file_name=model_settings["SPEC"])
    coefficients_df = simulate.read_model_coefficients(model_settings)
    model_spec = simulate.eval_coefficients(model_spec_raw, coefficients_df, estimator)

    constants = config.get_model_constants(model_settings)

    locals_dict = {}
    locals_dict.update(constants)
    locals_dict.update(coefficients_df)

    choices, choosers = iterate_vehicle_type_choice(
        vehicles_merged,
        model_settings,
        model_spec,
        locals_dict,
        estimator,
        chunk_size,
        trace_label,
    )

    if estimator:
        estimator.write_model_settings(model_settings, model_settings_file_name)
        estimator.write_spec(model_settings)
        estimator.write_coefficients(coefficients_df, model_settings)
        estimator.write_choosers(choosers)

        # FIXME #interaction_simulate_estimation_requires_chooser_id_in_df_column
        #  shuold we do it here or have interaction_simulate do it?
        # chooser index must be duplicated in column or it will be omitted from interaction_dataset
        # estimation requires that chooser_id is either in index or a column of interaction_dataset
        # so it can be reformatted (melted) and indexed by chooser_id and alt_id
        assert choosers.index.name == "vehicle_id"
        assert "vehicle_id" not in choosers.columns
        choosers["vehicle_id"] = choosers.index

        # FIXME set_alt_id - do we need this for interaction_simulate estimation bundle tables?
        estimator.set_alt_id("alt_id")
        estimator.set_chooser_id(choosers.index.name)

        estimator.write_choices(choices)
        choices = estimator.get_survey_values(
            choices, "vehicles", "vehicle_type_choice"
        )
        estimator.write_override_choices(choices)
        estimator.end_estimation()

    # update vehicles table
    # vehicles = pd.merge(vehicles.to_frame(), choices, left_index=True, right_index=True)
    vehicles = pd.concat([vehicles.to_frame(), choices], axis=1)
    pipeline.replace_table("vehicles", vehicles)

    # - annotate tables
    if model_settings.get("annotate_households"):
        annotate_vehicle_type_choice_households(model_settings, trace_label)
    if model_settings.get("annotate_persons"):
        annotate_vehicle_type_choice_persons(model_settings, trace_label)
    if model_settings.get("annotate_vehicles"):
        annotate_vehicle_type_choice_vehicles(model_settings, trace_label)

    tracing.print_summary(
        "vehicle_type_choice", vehicles.vehicle_type, value_counts=True
    )

    if trace_hh_id:
        tracing.trace_df(vehicles, label="vehicle_type_choice", warn_if_empty=True)
