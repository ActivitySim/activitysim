# //////////////////////////////////////////////////////////////////////////////
# ////                                                                       ///
# //// Copyright RSG, 2019-2020.                                             ///
# //// Rights to use and modify are granted to the                           ///
# //// San Diego Association of Governments and partner agencies.            ///
# //// This copyright notice must be preserved.                              ///
# ////                                                                       ///
# //// import/input_checker.py                                               ///
# ////                                                                       ///
# ////                                                                       ///
# ////                                                                       ///
# ////                                                                       ///
# //////////////////////////////////////////////////////////////////////////////
#
# Reviews all inputs to ActivitySim for possible issues that will result in model errors
#
#
# Files referenced:
# 	input_checker\config\input_checker_spec.csv
# 	input_checker\config\inputs_list.csv
#
# Script example:
# python C:\ABM_runs\maint_2020_RSG\Tasks\input_checker\emme_toolbox\emme\toolbox\import\input_checker.py


import os, sys, logging
import win32com.client as com
import numpy as np
import pandas as pd
import datetime
import warnings
import pandas as pd
import numpy as np
import pandera as pa
import pydantic
import time
from collections import defaultdict

from activitysim.core import config, inject

from activitysim.core.input import read_input_table

logger = logging.getLogger(__name__)

# warnings.filterwarnings("ignore")

_join = os.path.join
_dir = os.path.dirname

global TABLE_STORE
TABLE_STORE = {}


class ValidationWarning(UserWarning):
    pass


def create_table_store(input_checker_settings):
    """
    creating a global variable called TABLE_STORE to be able to access
    all tables at any point in the input checker python code

    FIXME: can we do this better with the state object?
    """
    # looping through all tables listed in input_checker.yaml
    for table_settings in input_checker_settings["table_list"]:
        table_name = table_settings["name"]
        logger.info("reading in table for input checking: %s" % table_name)

        # read with ActivitySim's input reader if ActivitySim input
        if table_settings["is_activitysim_input"]:
            table = read_input_table(table_name).reset_index()

        # otherwise read csv file directly with pandas
        else:
            path = table_settings.get("path", None)
            if path:
                table = pd.read_csv(os.path.join(path, table_name))
            else:
                table = pd.read_csv(config.data_file_path(table_name))

        # add pandas dataframes to TABLE_STORE dictionary with table name as key
        TABLE_STORE[table_name] = table


def add_child_to_parent_list(pydantic_lists, parent_table_name, children_settings):
    """
    Code to efficiently add children to the parent list.

    If "parent" is household and "child" is persons, the following code is equivalent to:
    for household in h_list:
        person_list = []
        for person in p_list:
            if household["household_id"] == person["household_id"]:
                person_list.append(person)
        household["persons"] = person_list
    """
    child_table_name = children_settings["table_name"]
    child_variable_name = children_settings["child_name"]
    merge_variable = children_settings["merged_on"]

    # need to create child list if it does not yet exist
    if child_table_name not in pydantic_lists.keys():
        pydantic_lists[child_table_name] = TABLE_STORE[child_table_name].to_dict(
            orient="records"
        )

    child_list = pydantic_lists[child_table_name]
    parent_list = pydantic_lists[parent_table_name]

    logger.info(
        f"Adding {child_table_name} to {parent_table_name} based on {merge_variable}"
    )

    parent_children = defaultdict(list)
    for child in child_list:
        children = child[merge_variable]
        parent_children[children].append(child)

    for parent in parent_list:
        children = parent[merge_variable]
        parent[child_variable_name] = parent_children.get(children, [])

    pydantic_lists[parent_table_name] = parent_list

    return pydantic_lists


def validate_with_pandera(
    pandera_checker, table_name, validation_settings, v_errors, v_warnings
):
    """
    Validating with pandera.  Grabs the relevant class for the table and runs the validate() function.

    Structure of the code is as follows:
    households = pd.DataFrame()
    in pandera_checker:
        class Household(pa.DataFrameModel)...
    validator_class = pandera_checker.Household()
    validator_class.validate(households)

    Warnings & errors are captured and written out in full after all tables are checked.
    """

    validator_class = getattr(pandera_checker, validation_settings["class"])
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            validator_class.validate(TABLE_STORE[table_name])
            v_warnings[table_name] = caught_warnings
    except pa.errors.SchemaError as e:
        v_errors[table_name].append(e)

    return v_errors, v_warnings


def validate_with_pydantic(
    pydantic_checker,
    table_name,
    validation_settings,
    v_errors,
    v_warnings,
    pydantic_lists,
):
    """
    Validing table wiht pydantic. Uses a helper class to perform the validation.

    Strucutre of the code is as follows:
    households = pd.DataFrame()
    in pydantic_checker:
        class Household(pydantic.BaseModel)...
        class HouseholdValidator(pydantic.BaseModel) list_of_households...
    validator_class = pydantic_checker.HouseholdValidator()
    h_list = households.to_dict() with optional addition of child class (aka Persons)
    validator_class(list_of_households=h_list)

    Warnings & errors are captured and written out in full after all tables are checked.
    """
    if table_name not in pydantic_lists.keys():
        pydantic_lists[table_name] = TABLE_STORE[table_name].to_dict(orient="records")

    if validation_settings.get("children") is not None:
        # FIXME need to add functionality for if there are multiple children
        pydantic_lists = add_child_to_parent_list(
            pydantic_lists, table_name, validation_settings["children"]
        )

    validator_class = getattr(pydantic_checker, validation_settings["helper_class"])

    attr = validation_settings["helper_class_attribute"]

    try:
        validator_instance = validator_class(**{attr: pydantic_lists[table_name]})
    except pydantic.error_wrappers.ValidationError as e:
        v_errors[table_name].append(e)
    except ValidationWarning as w:
        v_warnings[table_name].append(w)

    return v_errors, v_warnings, pydantic_lists


def report_errors(input_checker_settings, v_warnings, v_errors):
    kill_run = False
    for table_settings in input_checker_settings["table_list"]:
        table_name = table_settings["name"]
        errors = v_errors[table_name]
        if len(errors) > 0:
            kill_run = True
            logger.error(f"Encountered {len(errors)} errors in table {table_name}")
            [print(error) for error in errors]

        warns = v_warnings[table_name]
        if len(warns) > 0:
            logger.warn(f"Encountered {len(warns)} warnings in table {table_name}")
            [print(warn) for warn in warns]

    return kill_run


@inject.step()
def input_checker_data_model():

    input_checker_settings = config.read_model_settings(
        "input_checker.yaml", mandatory=True
    )

    pydantic_checker_file = input_checker_settings["input_checker_code"].get("pydantic")
    pandera_checker_file = input_checker_settings["input_checker_code"].get("pandera")

    assert (pydantic_checker_file is not None) | (
        pandera_checker_file is not None
    ), "Need to specify the `pydantic` or `pandera` options for the `input_checker` setting"

    print(inject.get_injectable("data_model_dir"))
    data_model_dir = inject.get_injectable("data_model_dir")[0]

    sys.path.append(data_model_dir)

    create_table_store(input_checker_settings)

    # import the datamodel.input.py after the TABLE_STORE is initialized so functions have access to the variable
    pandera_checker = __import__(pandera_checker_file)
    pydantic_checker = __import__(pydantic_checker_file)

    v_errors = {}
    v_warnings = {}
    pydantic_lists = {}

    for table_settings in input_checker_settings["table_list"]:
        validation_settings = table_settings["validation"]
        table_name = table_settings["name"]

        # initializing validation error and warning tracking
        v_errors[table_name] = []
        v_warnings[table_name] = []

        assert validation_settings["method"] in ["pandera", "pydantic"]

        if validation_settings["method"] == "pandera":
            logger.info(f"performing Pandera check on {table_settings['name']}")
            v_errors, v_warnings = validate_with_pandera(
                pandera_checker, table_name, validation_settings, v_errors, v_warnings
            )

        if validation_settings["method"] == "pydantic":
            logger.info(f"performing Pydantic check on {table_settings['name']}")
            v_errors, v_warnings, pydantic_lists = validate_with_pydantic(
                pydantic_checker,
                table_name,
                validation_settings,
                v_errors,
                v_warnings,
                pydantic_lists,
            )

    kill_run = report_errors(input_checker_settings, v_warnings, v_errors)

    if kill_run:
        logger.error("Run would be killed due to input checker failure!!")
        # raise RuntimeError("Encountered error in input checker, see input_checker.log for details")
