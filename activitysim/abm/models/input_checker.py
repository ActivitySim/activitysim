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

from activitysim.core import config, inject

from activitysim.core.input import read_input_table

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

_join = os.path.join
_dir = os.path.dirname


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

        self.read_inputs()
        self.check_inputs()

    # TODO: make sure we do not need to set index
    # def read_model_spec(file_name):

    #     assert isinstance(file_name, str)
    #     # if not file_name.lower().endswith(".csv"):
    #     #     file_name = "%s.csv" % (file_name,)

    #     file_path = config.config_file_path(file_name)

    #     try:
    #         spec = pd.read_csv(file_path, comment="#")
    #     except Exception as err:
    #         logger.error(f"read_model_spec error reading {file_path}")
    #         logger.error(f"read_model_spec error {type(err).__name__}: {str(err)}")
    #         raise (err)

    #     return spec

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

    def checks(self):
        # read all input DFs into memory
        for key, df in self.inputs.items():
            expr = key + " = df"
            exec(expr)

        # copy of locals(), a dictionary of all local variables
        calc_dict = locals()

        # read list of checks from CSV file
        # self.input_checker_spec = pd.read_csv(self.input_checker_spec_path)

        # remove all commented checks from the checks list
        self.input_checker_spec = self.input_checker_spec.loc[
            [not i for i in (self.input_checker_spec["Test"].str.startswith("#"))]
        ]

        # loop through list of checks and conduct all checks
        # checks must evaluate to True if inputs are correct
        for _, row in self.input_checker_spec.iterrows():

            test = row["Test"]
            table = row["Input_Table"]
            id_col = row["Input_ID_Column"]
            expr = row["Expression"]
            test_vals = row["Test_Vals"]
            if not (pd.isnull(row["Test_Vals"])):
                test_vals = test_vals.split(",")
                test_vals = [txt.strip() for txt in test_vals]
            test_type = row["Type"]
            Severity = row["Severity"]
            stat_expr = row["Report_Statistic"]

            if test_type == "Test":

                logger.info("Performing Check: " + row["Test"])

                if pd.isnull(row["Test_Vals"]):

                    # perform test
                    out = eval(expr, calc_dict)

                    # check if test result is a series
                    if str(type(out)) == "<class 'pandas.core.series.Series'>":
                        # for series, the test must be evaluated across all items
                        # result is False if a single False is found
                        self.results[test] = not (False in out.values)

                        # reverse results list since we need all False IDs
                        reverse_results = [not i for i in out.values]
                        error_expr = table + "." + id_col + "[reverse_results]"
                        error_id_list = eval(error_expr)

                        # report first 25 problem IDs in the log
                        if error_id_list.size > 25:
                            self.problem_ids[test] = error_id_list.iloc[range(25)]
                        else:
                            self.problem_ids[test] = (
                                error_id_list if error_id_list.size > 0 else []
                            )

                        # compute report statistics
                        if pd.isnull(stat_expr):
                            self.report_stat[test] = ""
                        else:
                            stat_list = eval(stat_expr)
                            self.report_stat[test] = stat_list[reverse_results]
                    else:
                        self.results[test] = out
                        self.problem_ids[test] = []
                        if pd.isnull(stat_expr):
                            self.report_stat[test] = ""
                        else:
                            self.report_stat[test] = eval(stat_expr)
                else:
                    # loop through test_vals and perform test for each item
                    self.result_list[test] = []
                    for test_val in test_vals:
                        # perform test (test result must not be of type Series)
                        out = eval(expr)

                        # compute report statistic
                        if pd.isnull(stat_expr):
                            self.report_stat[test] = ""
                        else:
                            self.report_stat[test] = eval(stat_expr)

                        # append to list
                        self.result_list[test].append(out)
                    self.results[test] = not (False in self.result_list[test])
                    self.problem_ids[test] = []

                print(" - Check Complete")

            else:
                # perform calculation
                print("Performing Calculation: " + row["Test"])
                calc_expr = test + " = " + expr
                exec(calc_expr, {}, calc_dict)
                print(" - Calculation Complete")

    def check_inputs(self):
        # logger.info("Started running input checker...")

        # self.input_checker_path = _join(self.path, "input_checker")
        # self.inputs_list_path = _join(
        #     self.input_checker_path, "config", "inputs_list.csv"
        # )
        # self.input_checker_spec_path = _join(
        #     self.input_checker_path, "config", "input_checker_spec.csv"
        # )

        # attributes = {"path": self.path}
        # gen_utils.log_snapshot("Run Input Checker", str(self), attributes)

        # file_paths = [self.inputs_list_path, self.input_checker_spec_path]
        # for path in file_paths:
        #     if not os.path.exists(path):
        #         raise Exception("missing file '%s'" % (path))

        self.read_inputs()

        logger.info("Conducting input checks...")
        self.checks()

        logger.info("Writing logs...")
        self.write_log()

        logger.info("Checking for fatal errors...")
        self.check_num_fatal()

        logger.info("Finisehd running input checker")

    def write_log(self):
        # function to write out the input checker log file
        # there are three blocks:
        #   - Introduction
        #   - Action Required: FATAL, LOGICAL, WARNINGS
        #   - List of passed checks

        # create log file
        now = datetime.datetime.now()

        # create log directory if it doesn't already exist
        log_path = _join(self.input_checker_path, "logs")
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        f = open(
            config.output_file_path(f'inputCheckerLog{now.strftime("[%Y-%m-%d]")}.log'),
            "w",
        )

        # define re-usable elements
        seperator1 = "###########################################################"
        seperator2 = "***********************************************************"

        # write out Header
        f.write(seperator1 + seperator1 + "\r\n")
        f.write(seperator1 + seperator1 + "\r\n\r\n")
        f.write("\t ActivitySim Input Checker Log File \r\n")
        f.write("\t ____________________________ \r\n\r\n\r\n")
        f.write("\t Log created on: " + now.strftime("%Y-%m-%d %H:%M") + "\r\n\r\n")
        f.write("\t Notes:-\r\n")
        f.write(
            "\t The ActivitySim Input Checker performs various QA/QC checks on ActivitySim inputs as specified by the user.\r\n"
        )
        f.write(
            "\t The Input Checker allows the user to specify three severity levels for each QA/QC check:\r\n\r\n"
        )
        f.write("\t 1) FATAL  2) LOGICAL  3) WARNING\r\n\r\n")
        f.write(
            "\t FATAL Checks:   The failure of these checks would result in a FATAL errors in the ActivitySim ABM run.\r\n"
        )
        f.write(
            "\t                 In case of FATAL failure, the Input Checker returns a return code of 1 to the\r\n"
        )
        f.write(
            "\t                 main ActivitySim ABM model, cauing the model run to halt.\r\n"
        )
        f.write(
            "\t LOGICAL Checks: The failure of these checks indicate logical inconsistencies in the inputs.\r\n"
        )
        f.write(
            "\t                 With logical errors in inputs, the ActivitySim ABM outputs may not be meaningful.\r\n"
        )
        f.write(
            "\t WARNING Checks: The failure of Warning checks would indicate problems in data that would not.\r\n"
        )
        f.write(
            "\t                 halt the run or affect model outputs but might indicate an issue with inputs.\r\n\r\n\r\n"
        )
        f.write("\t The results of all the checks are organized as follows: \r\n\r\n")
        f.write("\t IMMEDIATE ACTION REQUIRED:\r\n")
        f.write("\t -------------------------\r\n")
        f.write(
            "\t A log under this heading will be generated in case of failure of a FATAL check\r\n\r\n"
        )
        f.write("\t ACTION REQUIRED:\r\n")
        f.write("\t ---------------\r\n")
        f.write(
            "\t A log under this heading will be generated in case of failure of a LOGICAL check\r\n\r\n"
        )
        f.write("\t WARNINGS:\r\n")
        f.write("\t ---------------\r\n")
        f.write(
            "\t A log under this heading will be generated in case of failure of a WARNING check\r\n\r\n"
        )
        f.write("\t LOG OF ALL PASSED CHECKS:\r\n")
        f.write("\t -----------\r\n")
        f.write("\t A complete listing of results of all passed checks\r\n\r\n")
        f.write(seperator1 + seperator1 + "\r\n")
        f.write(seperator1 + seperator1 + "\r\n\r\n\r\n\r\n")

        # combine results, input_checker_spec and inputs_list
        self.input_checker_spec["result"] = self.input_checker_spec["Test"].map(
            self.results
        )
        checks_df = pd.merge(
            self.input_checker_spec, self.inputs_list, how="left", on="Input_Table"
        )
        checks_df = checks_df[checks_df.Type == "Test"]
        checks_df["reverse_result"] = [not i for i in checks_df.result]

        # get all FATAL failures
        self.num_fatal = checks_df.result[
            (checks_df.Severity == "Fatal") & (checks_df.reverse_result)
        ].count()

        # get all LOGICAL failures
        self.num_logical = checks_df.result[
            (checks_df.Severity == "Logical") & (checks_df.reverse_result)
        ].count()

        # get all WARNING failures
        self.num_warning = checks_df.result[
            (checks_df.Severity == "Warning") & (checks_df.reverse_result)
        ].count()

        def write_check_log(self, fh, row):
            # define constants
            seperator2 = "-----------------------------------------------------------"

            # integerize problem ID list
            problem_ids = self.problem_ids[row["Test"]]
            problem_ids = [int(x) for x in problem_ids]

            # write check summary
            fh.write("\r\n\r\n" + seperator2 + seperator2)
            fh.write("\r\n\t Input File Name: " + (row["Input_Table"]))
            fh.write(
                "\r\n\t Input Description: "
                + (
                    row["Input_Description"]
                    if not pd.isnull(row["Input_Description"])
                    else ""
                )
            )
            fh.write("\r\n\t Test Name: " + row["Test"])
            fh.write(
                "\r\n\t Test_Description: "
                + (
                    row["Test_Description"]
                    if not pd.isnull(row["Test_Description"])
                    else ""
                )
            )
            fh.write("\r\n\t Test Severity: " + row["Severity"])
            fh.write(
                "\r\n\r\n\t TEST RESULT: " + ("PASSED" if row["result"] else "FAILED")
            )

            # display problem IDs for failed column checks
            if (not row["result"]) & (len(problem_ids) > 0):
                fh.write(
                    "\r\n\t TEST failed for following values of ID Column: "
                    + row["Input_ID_Column"]
                    + " (only upto 25 IDs displayed)"
                )
                fh.write(
                    "\r\n\t "
                    + row["Input_ID_Column"]
                    + ": "
                    + ",".join(map(str, problem_ids[0:25]))
                )
                if not (pd.isnull(row["Report_Statistic"])):
                    this_report_stat = self.report_stat[row["Test"]]
                    fh.write(
                        "\r\n\t Test Statistics: "
                        + ",".join(map(str, this_report_stat[0:25]))
                    )
                fh.write("\r\n\t Total number of failures: " + str(len(problem_ids)))
            else:
                if not (pd.isnull(row["Report_Statistic"])):
                    fh.write(
                        "\r\n\t Test Statistic: " + str(self.report_stat[row["Test"]])
                    )

            # display result for each test val if it was specified
            if not (pd.isnull(row["Test_Vals"])):
                fh.write("\r\n\t TEST results for each test val")
                result_tuples = zip(
                    row["Test_Vals"].split(","), self.result_list[row["Test"]]
                )
                fh.write("\r\n\t ")
                fh.write(
                    ",".join("[{} - {}]".format(x[0], x[1]) for x in result_tuples)
                )

            fh.write("\r\n" + seperator2 + seperator2 + "\r\n\r\n")

        # write out IMMEDIATE ACTION REQUIRED section if needed
        if self.num_fatal > 0:
            fatal_checks = checks_df[
                (checks_df.Severity == "Fatal") & (checks_df.reverse_result)
            ]
            f.write("\r\n\r\n" + seperator2 + seperator2 + "\r\n")
            f.write(seperator2 + seperator2 + "\r\n\r\n")
            f.write("\t" + "IMMEDIATE ACTION REQUIRED \r\n")
            f.write("\t" + "------------------------- \r\n\r\n")
            f.write(seperator2 + seperator2 + "\r\n")
            f.write(seperator2 + seperator2 + "\r\n")

            # write out log for each check
            for item, row in fatal_checks.iterrows():
                # self.write_check_log(f, row, self.problem_ids[row['Test']])
                # write_check_log(self, f, row, self.problem_ids[row['Test']])
                write_check_log(self, f, row)

        # write out ACTION REQUIRED section if needed
        if self.num_logical > 0:
            logical_checks = checks_df[
                (checks_df.Severity == "Logical") & (checks_df.reverse_result)
            ]
            f.write("\r\n\r\n" + seperator2 + seperator2 + "\r\n")
            f.write(seperator2 + seperator2 + "\r\n\r\n")
            f.write("\t" + "ACTION REQUIRED \r\n")
            f.write("\t" + "--------------- \r\n\r\n")
            f.write(seperator2 + seperator2 + "\r\n")
            f.write(seperator2 + seperator2 + "\r\n")

            # write out log for each check
            for item, row in logical_checks.iterrows():
                write_check_log(self, f, row)

        # write out WARNINGS section if needed
        if self.num_warning > 0:
            warning_checks = checks_df[
                (checks_df.Severity == "Warning") & (checks_df.reverse_result)
            ]
            f.write("\r\n\r\n" + seperator2 + seperator2 + "\r\n")
            f.write(seperator2 + seperator2 + "\r\n\r\n")
            f.write("\t" + "WARNINGS \r\n")
            f.write("\t" + "-------- \r\n")
            f.write(seperator2 + seperator2 + "\r\n")
            f.write(seperator2 + seperator2 + "\r\n")

            # write out log for each check
            for item, row in warning_checks.iterrows():
                write_check_log(self, f, row)

        # write out the complete listing of all checks that passed
        passed_checks = checks_df[(checks_df.result)]
        f.write("\r\n\r\n" + seperator2 + seperator2 + "\r\n")
        f.write(seperator2 + seperator2 + "\r\n\r\n")
        f.write("\t" + "LOG OF ALL PASSED CHECKS \r\n")
        f.write("\t" + "------------------------ \r\n")
        f.write(seperator2 + seperator2 + "\r\n")
        f.write(seperator2 + seperator2 + "\r\n")

        # write out log for each check
        for item, row in passed_checks.iterrows():
            write_check_log(self, f, row)

        f.close()
        # write out a summary of results from input checker for main model
        f = open(
            _join(self.input_checker_path, "logs", ("inputCheckerSummary" + ".txt")),
            "w",
        )
        f.write("\r\n" + seperator2 + "\r\n")
        f.write("\t Summary of Input Checker Fails \r\n")
        f.write(seperator2 + "\r\n\r\n")
        f.write(" Number of Fatal Errors: " + str(self.num_fatal))
        f.write("\r\n\r\n Number of Logical Errors: " + str(self.num_logical))
        f.write("\r\n\r\n Number of Warnings: " + str(self.num_warning) + "\r\n\r\n")
        f.close()

    def check_num_fatal(self):
        # return code to the main model based on input checks and results
        if self.num_fatal > 0:
            logger.info("At least one fatal error in the inputs.")
            logger.info("Input Checker Failed")
            sys.exit(2)


@inject.step()
def input_checker():

    print("input checker!")
    InputChecker()
