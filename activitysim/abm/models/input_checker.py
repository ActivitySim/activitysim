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

# from simpledbf import Dbf5

import pandas as pd
import numpy as np
import pandera as pa

from activitysim.core import config, inject

from activitysim.core.input import read_input_table

logger = logging.getLogger(__name__)

# warnings.filterwarnings("ignore")

_join = os.path.join
_dir = os.path.dirname

global TABLE_STORE
TABLE_STORE = {}

class InputChecker:
    def __init__(self):
        # project_dir = _dir(_m.Modeller().desktop.project.path)
        # self.path = _dir(project_dir)
        self.input_checker_path = ""
        self.inputs_list_path = ""
        self.prop_input_paths = {}
        self.inputs_list = pd.DataFrame()
        self.inputs = {}
        self.results = {}
        self.result_list = {}
        self.problem_ids = {}
        self.report_stat = {}
        self.num_fatal = int()
        self.num_warning = int()
        self.num_logical = int()

        self.trace_label = "input_checker"
        self.model_settings_file_name = "input_checker.yaml"
        model_settings = config.read_model_settings(self.model_settings_file_name)

        # input_item_list = self.read_model_spec(file_name=model_settings["INPUT_ITEM_LIST"])
        # model_spec = self.read_model_spec(file_name=model_settings["SPEC"])

        # TEMP fix for now since the read_model_spec is not working for some reason
        file_name = model_settings["INPUT_ITEM_LIST"]
        file_path = config.config_file_path(file_name)
        self.inputs_list = pd.read_csv(file_path, comment="#")

        file_name = model_settings["SPEC"]
        file_path = config.config_file_path(file_name)
        self.input_checker_spec = pd.read_csv(file_path, comment="#")

        logger.info("Running %s", self.trace_label)


    def read_inputs(self):

        self.inputs["persons"] = read_input_table("persons")
        self.inputs["persons"].reset_index(inplace=True)
        self.inputs["households"] = read_input_table("households")
        self.inputs["households"].reset_index(inplace=True)
        self.inputs["land_use"] = read_input_table("land_use")
        self.inputs["land_use"].reset_index(inplace=True)

        # check to see if input list has any additional files to read in
        if not self.inputs_list.empty:
            self.inputs_list = self.inputs_list.loc[
                [not i for i in (self.inputs_list["Input_Table"].str.startswith("#"))]
            ]

            for _, row in self.inputs_list.iterrows():

                print("Adding Input: " + row["Input_Table"])

                table_name = row["Input_Table"]
                column_map = row["Column_Map"]
                fields_to_export = row["Fields"].split(",")

                # TOFIX: inpit path is not set; dbf5 is not working
                input_path = self.prop_input_paths[table_name]
                input_ext = os.path.splitext(input_path)[1]
                if input_ext == ".csv":
                    df = pd.read_csv(_join(self.path, input_path))
                    self.inputs[table_name] = df
                    print(" - " + table_name + " added")
                else:
                    dbf = Dbf5(_join(_dir(self.path), input_path))
                    df = dbf.to_dataframe()
                    self.inputs[table_name] = df
                    print(" - " + table_name + " added")



@inject.step()
def input_checker_data_model():

    print(inject.get_injectable('data_model_dir'))
    data_model_dir = inject.get_injectable('data_model_dir')[0]
    import sys
    sys.path.append(data_model_dir)
    
    ic = InputChecker()

    # load specified tables
    ic.read_inputs()

    for table_name, table in ic.inputs.items():
        # add to global table store for easy access
        # FIXME replace with the state object
        TABLE_STORE[table_name] = table
        

    # import the datamodel.input.py after the TABLE_STORE is initialized so functions have access to the variable
    import input as dm_input

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        dm_input.Person.validate(TABLE_STORE['persons'], lazy=True)
        dm_input.Household.validate(TABLE_STORE['households'], lazy=True)
        dm_input.Landuse.validate(TABLE_STORE['land_use'], lazy=True)
        for warning in caught_warnings:
            print(warning.message)

    print("Input Checker Finished!")
    
    



