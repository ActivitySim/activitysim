# ActivitySim
# See full license in LICENSE.txt.

import logging
import sys

import pandas as pd

from activitysim.core import chunk, config, inject, mem, mp_tasks, pipeline, tracing

# from activitysim import abm


logger = logging.getLogger("activitysim")


def cleanup_output_files():

    active_log_files = [
        h.baseFilename
        for h in logger.root.handlers
        if isinstance(h, logging.FileHandler)
    ]
    tracing.delete_output_files("log", ignore=active_log_files)

    tracing.delete_output_files("h5")
    tracing.delete_output_files("csv")
    tracing.delete_output_files("txt")
    tracing.delete_output_files("yaml")
    tracing.delete_output_files("prof")


def run(run_list, injectables=None):

    if run_list["multiprocess"]:
        logger.info("run multiprocess simulation")
        mp_tasks.run_multiprocess(run_list, injectables)
    else:
        logger.info("run single process simulation")
        pipeline.run(models=run_list["models"], resume_after=run_list["resume_after"])
        pipeline.close_pipeline()
        mem.log_global_hwm()


def log_settings(injectables):

    settings = [
        "households_sample_size",
        "chunk_size",
        "multiprocess",
        "num_processes",
        "resume_after" "use_shadow_pricing",
        "hh_ids",
    ]

    for k in settings:
        logger.info("setting %s: %s" % (k, config.setting(k)))

    for k in injectables:
        logger.info("injectable %s: %s" % (k, inject.get_injectable(k)))


if __name__ == "__main__":

    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.set_option.html
    # pd.set_option('display.max_columns', 50)

    data_dir = "E:/projects/clients/ASIM/data/mtc_tm1"
    data_dir = "/Users/jeff.doyle/work/activitysim-data/mtc_tm1/data"
    data_dir = "../example/data"

    # inject.add_injectable('data_dir', '/Users/jeff.doyle/work/activitysim-data/mtc_tm1/data')
    inject.add_injectable("data_dir", ["ancillary_data", data_dir])
    # inject.add_injectable('data_dir', ['ancillary_data', '../activitysim/abm/test/data'])
    inject.add_injectable("configs_dir", ["configs", "../example/configs"])

    injectables = config.handle_standard_args()

    tracing.config_logger()
    config.filter_warnings()

    log_settings(injectables)

    t0 = tracing.print_elapsed_time()

    # cleanup if not resuming
    if not config.setting("resume_after", False):
        cleanup_output_files()

    run_list = mp_tasks.get_run_list()

    if run_list["multiprocess"]:
        # do this after config.handle_standard_args, as command line args may override injectables
        injectables = list(
            set(injectables) | set(["data_dir", "configs_dir", "output_dir"])
        )
        injectables = {k: inject.get_injectable(k) for k in injectables}
    else:
        injectables = None

    run(run_list, injectables)

    # pipeline.open_pipeline('_')
    #
    # households_df = pipeline.get_table('households')
    # print("households_df\n", households_df.head())
    #
    # tours_df = pipeline.get_table('tours')
    # print("tours_df\n", tours_df.head())
    #
    # pipeline.close_pipeline()

    t0 = tracing.print_elapsed_time("everything", t0)
