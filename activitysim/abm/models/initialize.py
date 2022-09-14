# ActivitySim
# See full license in LICENSE.txt.
import logging
import os
import warnings

import pandas as pd

from activitysim.abm.tables import shadow_pricing
from activitysim.core import chunk, config, expressions, inject, mem, pipeline, tracing
from activitysim.core.steps.output import (
    track_skim_usage,
    write_data_dictionary,
    write_tables,
)

# We are using the naming conventions in the mtc_asim.h5 example
# file for our default list. This provides backwards compatibility
# with previous versions of ActivitySim in which only 'input_store'
# is given in the settings file.
DEFAULT_TABLE_LIST = [
    {
        "tablename": "households",
        "h5_tablename": "households",
        "index_col": "household_id",
    },
    {"tablename": "persons", "h5_tablename": "persons", "index_col": "person_id"},
    {"tablename": "land_use", "h5_tablename": "land_use_taz", "index_col": "TAZ"},
]

logger = logging.getLogger(__name__)


def annotate_tables(model_settings, trace_label):

    trace_label = tracing.extend_trace_label(trace_label, "annotate_tables")

    chunk.log_rss(trace_label)

    annotate_tables = model_settings.get("annotate_tables", [])

    if not annotate_tables:
        logger.warning(
            f"{trace_label} - annotate_tables setting is empty - nothing to do!"
        )

    assert isinstance(
        annotate_tables, list
    ), f"annotate_tables settings should be a list but is {type(annotate_tables)}"

    t0 = tracing.print_elapsed_time()

    for table_info in annotate_tables:

        tablename = table_info["tablename"]

        chunk.log_rss(f"{trace_label}.pre-get_table.{tablename}")

        df = inject.get_table(tablename).to_frame()
        chunk.log_df(trace_label, tablename, df)

        # - rename columns
        column_map = table_info.get("column_map", None)
        if column_map:

            warnings.warn(
                f"Setting 'column_map' has been changed to 'rename_columns'. "
                f"Support for 'column_map' in annotate_tables  will be removed in future versions.",
                FutureWarning,
            )

            logger.info(f"{trace_label} - renaming {tablename} columns {column_map}")
            df.rename(columns=column_map, inplace=True)

        # - annotate
        annotate = table_info.get("annotate", None)
        if annotate:
            logger.info(
                f"{trace_label} - annotating {tablename} SPEC {annotate['SPEC']}"
            )
            expressions.assign_columns(
                df=df, model_settings=annotate, trace_label=trace_label
            )

        chunk.log_df(trace_label, tablename, df)

        # - write table to pipeline
        pipeline.replace_table(tablename, df)

        del df
        chunk.log_df(trace_label, tablename, None)


@inject.step()
def initialize_landuse():

    trace_label = "initialize_landuse"

    with chunk.chunk_log(trace_label, base=True):

        model_settings = config.read_model_settings(
            "initialize_landuse.yaml", mandatory=True
        )

        annotate_tables(model_settings, trace_label)

        # instantiate accessibility (must be checkpointed to be be used to slice accessibility)
        accessibility = pipeline.get_table("accessibility")
        chunk.log_df(trace_label, "accessibility", accessibility)


@inject.step()
def initialize_households():

    trace_label = "initialize_households"

    with chunk.chunk_log(trace_label, base=True):

        chunk.log_rss(f"{trace_label}.inside-yield")

        households = inject.get_table("households").to_frame()
        assert not households._is_view
        chunk.log_df(trace_label, "households", households)
        del households
        chunk.log_df(trace_label, "households", None)

        persons = inject.get_table("persons").to_frame()
        assert not persons._is_view
        chunk.log_df(trace_label, "persons", persons)
        del persons
        chunk.log_df(trace_label, "persons", None)

        model_settings = config.read_model_settings(
            "initialize_households.yaml", mandatory=True
        )
        annotate_tables(model_settings, trace_label)

        # - initialize shadow_pricing size tables after annotating household and person tables
        # since these are scaled to model size, they have to be created while single-process
        # this can now be called as a stand alone model step instead, add_size_tables
        add_size_tables = model_settings.get("add_size_tables", True)
        if add_size_tables:
            # warnings.warn(f"Calling add_size_tables from initialize will be removed in the future.", FutureWarning)
            shadow_pricing.add_size_tables()

        # - preload person_windows
        person_windows = inject.get_table("person_windows").to_frame()
        chunk.log_df(trace_label, "person_windows", person_windows)


@inject.injectable(cache=True)
def preload_injectables():
    """
    preload bulky injectables up front - stuff that isn't inserted into the pipeline
    """

    logger.info("preload_injectables")

    inject.add_step("track_skim_usage", track_skim_usage)
    inject.add_step("write_data_dictionary", write_data_dictionary)
    inject.add_step("write_tables", write_tables)

    table_list = config.setting("input_table_list")

    # default ActivitySim table names and indices
    if table_list is None:
        logger.warning(
            "No 'input_table_list' found in settings. This will be a "
            "required setting in upcoming versions of ActivitySim."
        )

        new_settings = inject.get_injectable("settings")
        new_settings["input_table_list"] = DEFAULT_TABLE_LIST
        inject.add_injectable("settings", new_settings)

    # FIXME undocumented feature
    if config.setting("write_raw_tables"):

        # write raw input tables as csv (before annotation)
        csv_dir = config.output_file_path("raw_tables")
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)  # make directory if needed

        table_names = [t["tablename"] for t in table_list]
        for t in table_names:
            df = inject.get_table(t).to_frame()
            df.to_csv(os.path.join(csv_dir, "%s.csv" % t), index=True)

    t0 = tracing.print_elapsed_time()

    if config.setting("benchmarking", False):
        # we don't want to pay for skim_dict inside any model component during
        # benchmarking, so we'll preload skim_dict here.  Preloading is not needed
        # for regular operation, as activitysim components can load-on-demand.
        if inject.get_injectable("skim_dict", None) is not None:
            t0 = tracing.print_elapsed_time("preload skim_dict", t0, debug=True)

    return True
