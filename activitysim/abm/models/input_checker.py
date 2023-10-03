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
import warnings
import pandas as pd
import numpy as np
import pandera as pa
import pydantic
from collections import defaultdict

from activitysim.core import workflow

from activitysim.core.input import read_input_table

logger = logging.getLogger(__name__)

# warnings.filterwarnings("ignore")

_join = os.path.join
_dir = os.path.dirname

global TABLE_STORE
TABLE_STORE = {}


def create_table_store(state, input_checker_settings):
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
            table = read_input_table(state, table_name).reset_index()

        # otherwise read csv file directly with pandas
        else:
            path = table_settings.get("path", None)
            if path:
                table = pd.read_csv(os.path.join(path, table_name + ".csv"))
            else:
                table = pd.read_csv(state.filesystem.get_data_file_path(table_name))

        # add pandas dataframes to TABLE_STORE dictionary with table name as key
        TABLE_STORE[table_name] = table

    # FIXME: need to have state object available in the input checker. Is there a better way to pass this?
    TABLE_STORE["state"] = state


def add_child_to_parent_list(pydantic_lists, parent_table_name, children_settings):
    """
    Code to efficiently add children to the parent list.  Used for input checking with pydantic.

    If "parent" is household and "child" is persons, the following code is equivalent to:
    for household in h_list:
        person_list = []
        for person in p_list:
            if household["household_id"] == person["household_id"]:
                person_list.append(person)
        household["persons"] = person_list

    Above code is slow with nested for loops. Actual implementation is much faster.
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
    input_checker, table_name, validation_settings, v_errors, v_warnings
):
    """
    Validating with pandera.  Grabs the relevant class for the table and runs the validate() function.

    Structure of the code is as follows:
    households = pd.DataFrame()
    in input_checker:
        class Household(pa.DataFrameModel)...
    validator_class = input_checker.Household()
    validator_class.validate(households)

    Warnings & errors are captured and written out in full after all tables are checked.
    """

    validator_class = getattr(input_checker, validation_settings["class"])

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        try:
            validator_class.validate(TABLE_STORE[table_name], lazy=True)
        except pa.errors.SchemaErrors as e:
            v_errors[table_name].append(e)

        for warning in caught_warnings:
            if 'dataframe validator' in str(warning.message):
                v_warnings[table_name].append(
                    'Failed dataframe validator: '+
                    str(warning.message).split('\n')[-1])
            elif 'element-wise validator' in str(warning.message):
                v_warnings[table_name].append(
                    'Failed element-wise validator: <'+
                    ' '.join(str(warning.message).split('\n')[0].split(' ')[1:3])+'\n\t'+
                    '\n\t'.join(str(warning.message).split('\n')[1:])
                    )
     
    return v_errors, v_warnings


def validate_with_pydantic(
    input_checker,
    table_name,
    validation_settings,
    v_errors,
    v_warnings,
    pydantic_lists,
):
    """
    Validing table wiht pydantic. Uses a helper class to perform the validation.

    FIXME: Not fully built out!! Went with Pandera instead of pydantic,
           but leaving this in for future development

    Strucutre of the code is as follows:
    households = pd.DataFrame()
    in input_checker:
        class Household(pydantic.BaseModel)...
        class HouseholdValidator(pydantic.BaseModel) list_of_households...
    validator_class = input_checker.HouseholdValidator()
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

    validator_class = getattr(input_checker, validation_settings["helper_class"])

    attr = validation_settings["helper_class_attribute"]

    try:
        validator_instance = validator_class(**{attr: pydantic_lists[table_name]})
    except pydantic.error_wrappers.ValidationError as e:
        v_errors[table_name].append(e)
    # FIXME need to implement validation warning
    # except ValidationWarning as w:
    # v_warnings[table_name].append(w)

    return v_errors, v_warnings, pydantic_lists


def report_errors(state, input_checker_settings, v_warnings, v_errors):
    # creating a new log file to report out warnings and errors
    out_log_file = state.get_log_file_path("input_checker.log")
    if os.path.exists(out_log_file):
        os.remove(out_log_file)

    # logging overall statistics first before printing details
    for table_settings in input_checker_settings["table_list"]:
        table_name = table_settings["name"]

        # printing out any errors
        errors = v_errors[table_name]
        warns = v_warnings[table_name]

        msg = f"Encountered {sum(len(entry.schema_errors) for entry in errors)} errors and {len(warns)} warnings in table {table_name}"

        # first printing to activitysim log file
        if len(errors) > 0:
            logger.error(msg)
        elif len(warns) > 0:
            logger.warning(msg)
        else:
            logger.info(msg)

        # printing to input_checker.log file
        print(msg, file=open(out_log_file, "a"))

    # now reporting details to just input_checker.log
    input_check_failure = False
    # looping through each input table first
    for table_settings in input_checker_settings["table_list"]:
        table_name = table_settings["name"]

        # printing out any errors
        errors = v_errors[table_name]
        if len(errors) > 0:
            input_check_failure = True
            print(f"{table_name} errors:", file=open(out_log_file, "a"))

            for error_group in errors:
                print('Error Counts\n------------',file=open(out_log_file,'a'))
                for error_type in error_group.error_counts:
                    print(f"{error_type}\t{error_group.error_counts[error_type]}",file=open(out_log_file,'a'))
                print("\n", file=open(out_log_file, "a"))

                for error in error_group.schema_errors:
                    print(str(error),file=open(out_log_file,'a'))
                    print("\n", file=open(out_log_file, "a"))

        # printing out any warnings
        warns = v_warnings[table_name]
        if len(warns) > 0:
            print(f"{table_name} warnings:", file=open(out_log_file, "a"))
            [print(warn, file=open(out_log_file, "a")) for warn in warns]
            print("\n", file=open(out_log_file, "a"))

    if (len(v_warnings) > 0) | (len(v_errors) > 0):
        logger.info("See the input_checker.log for full details on errors and warnings")

    return input_check_failure


@workflow.step()
def input_checker(state: workflow.State):

    input_checker_settings = state.filesystem.read_model_settings(
        "input_checker.yaml", mandatory=True
    )

    input_checker_file = input_checker_settings.get(
        "input_checker_code", "input_checks.py"
    )

    data_model_dir = state.get_injectable("data_model_dir")[0]
    logger.info("Data model directory:", data_model_dir)

    # FIXME: path doesn't recognize windows path object, so converting to string.
    # Is there a better way to get the data model directory than just adding it to the path?
    # Can't just import input_checker because enums.py also needs to be recognized.
    sys.path.append(str(data_model_dir))

    input_checker_file_full = os.path.join(data_model_dir, input_checker_file)

    assert os.path.exists(
        input_checker_file_full
    ), f"Cannot find input checker file {input_checker_file} in data_model_dir {data_model_dir}"

    create_table_store(state, input_checker_settings)

    # import the input checker code after the TABLE_STORE is initialized so functions have access to the variable
    input_checker = __import__(os.path.splitext(input_checker_file)[0])

    # intializing data objects for errors, warnings, and pydantic data
    v_errors = {}
    v_warnings = {}
    pydantic_lists = {}

    # loop through each input table checker
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
                input_checker, table_name, validation_settings, v_errors, v_warnings
            )

        if validation_settings["method"] == "pydantic":
            logger.info(f"performing Pydantic check on {table_settings['name']}")
            v_errors, v_warnings, pydantic_lists = validate_with_pydantic(
                input_checker,
                table_name,
                validation_settings,
                v_errors,
                v_warnings,
                pydantic_lists,
            )

    input_check_failure = report_errors(
        state, input_checker_settings, v_warnings, v_errors
    )

    # free memory from input checker tables
    for key, value in TABLE_STORE.items():
        del value
    TABLE_STORE.clear()

    if input_check_failure:
        logger.error("Run is killed due to input checker failure!!")
        raise RuntimeError(
            "Encountered error in input checker, see input_checker.log for details"
        )
