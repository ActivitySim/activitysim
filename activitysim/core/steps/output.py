# ActivitySim
# See full license in LICENSE.txt.
import logging
import sys

import numpy as np
import pandas as pd

from activitysim.core import config, inject, pipeline
from activitysim.core.config import setting

logger = logging.getLogger(__name__)


def track_skim_usage(output_dir):
    """
    write statistics on skim usage (diagnostic to detect loading of un-needed skims)

    FIXME - have not yet implemented a facility to avoid loading of unused skims

    FIXME - if resume_after, this will only reflect skims used after resume

    Parameters
    ----------
    output_dir: str

    """
    pd.options.display.max_columns = 500
    pd.options.display.max_rows = 100

    skim_dict = inject.get_injectable("skim_dict")

    mode = "wb" if sys.version_info < (3,) else "w"
    with open(config.output_file_path("skim_usage.txt"), mode) as output_file:

        print("\n### skim_dict usage", file=output_file)
        for key in skim_dict.get_skim_usage():
            print(key, file=output_file)

        unused = set(k for k in skim_dict.skim_info.base_keys) - set(
            k for k in skim_dict.get_skim_usage()
        )

        for key in unused:
            print(key, file=output_file)


def previous_write_data_dictionary(output_dir):
    """
    Write table_name, number of rows, columns, and bytes for each checkpointed table

    Parameters
    ----------
    output_dir: str

    """

    model_settings = config.read_model_settings("write_data_dictionary")
    txt_format = model_settings.get("txt_format", "data_dict.txt")
    csv_format = model_settings.get("csv_format", "data_dict.csv")

    if txt_format:

        output_file_path = config.output_file_path(txt_format)

        pd.options.display.max_columns = 500
        pd.options.display.max_rows = 100

        output_tables = pipeline.checkpointed_tables()

        # write data dictionary for all checkpointed_tables

        with open(output_file_path, "w") as output_file:
            for table_name in output_tables:
                df = inject.get_table(table_name, None).to_frame()

                print("\n### %s %s" % (table_name, df.shape), file=output_file)
                print("index:", df.index.name, df.index.dtype, file=output_file)
                print(df.dtypes, file=output_file)


def write_data_dictionary(output_dir):
    """
    Write table schema for all tables

    model settings
        txt_format: output text file name (default data_dict.txt) or empty to suppress txt output
        csv_format: output csv file name (default data_dict.tcsvxt) or empty to suppress txt output

        schema_tables: list of tables to include in output (defaults to all checkpointed tables)

    for each table, write column names, dtype, and checkpoint added)

    text format writes individual table schemas to a single text file
    csv format writes all tables together with an additional table_name column

    Parameters
    ----------
    output_dir: str

    """

    model_settings = config.read_model_settings("write_data_dictionary")
    txt_format = model_settings.get("txt_format", "data_dict.txt")
    csv_format = model_settings.get("csv_format", "data_dict.csv")

    if not (csv_format or txt_format):
        logger.warning(
            f"write_data_dictionary step invoked but neither 'txt_format' nor 'csv_format' specified"
        )
        return

    table_names = pipeline.registered_tables()

    # use table_names list from model_settings, if provided
    schema_tables = model_settings.get("tables", None)
    if schema_tables:
        table_names = [c for c in schema_tables if c in table_names]

    # initialize schema as dict of dataframe[table_name, column_name, dtype, checkpoint]
    schema = dict()
    final_shapes = dict()
    for table_name in table_names:
        df = pipeline.get_table(table_name)

        final_shapes[table_name] = df.shape

        if df.index.name and df.index.name not in df.columns:
            df = df.reset_index()
        info = (
            df.dtypes.astype(str)
            .to_frame("dtype")
            .reset_index()
            .rename(columns={"index": "column_name"})
        )
        info["checkpoint"] = ""

        info.insert(loc=0, column="table_name", value=table_name)
        schema[table_name] = info

    # annotate schema.info with name of checkpoint columns were first seen
    for _, row in pipeline.get_checkpoints().iterrows():

        checkpoint_name = row[pipeline.CHECKPOINT_NAME]

        for table_name in table_names:

            # no change to table in this checkpoint
            if row.get(table_name, None) != checkpoint_name:
                continue

            # get the checkpointed version of the table
            df = pipeline.get_table(table_name, checkpoint_name)

            if df.index.name and df.index.name not in df.columns:
                df = df.reset_index()

            info = schema.get(table_name, None)

            # tag any new columns with checkpoint name
            prev_columns = info[info.checkpoint != ""].column_name.values
            new_cols = [c for c in df.columns.values if c not in prev_columns]
            is_new_column_this_checkpoont = info.column_name.isin(new_cols)
            info.checkpoint = np.where(
                is_new_column_this_checkpoont, checkpoint_name, info.checkpoint
            )

            schema[table_name] = info

    schema_df = pd.concat(schema.values())

    if csv_format:
        schema_df.to_csv(config.output_file_path(csv_format), header=True, index=False)

    if txt_format:
        with open(config.output_file_path(txt_format), "w") as output_file:

            # get max schema column widths from omnibus table
            col_width = {c: schema_df[c].str.len().max() + 2 for c in schema_df}

            for table_name in table_names:
                info = schema.get(table_name, None)

                columns_to_print = ["column_name", "dtype", "checkpoint"]
                info = info[columns_to_print].copy()

                # normalize schema columns widths across all table schemas for unified output formatting
                for c in info:
                    info[c] = info[c].str.pad(col_width[c], side="right")
                info.columns = [c.ljust(col_width[c]) for c in info.columns]

                info = info.to_string(index=False)

                print(
                    f"###\n### {table_name} {final_shapes[table_name]}\n###\n",
                    file=output_file,
                )
                print(f"{info}\n", file=output_file)


def write_tables(output_dir):
    """
    Write pipeline tables as csv files (in output directory) as specified by output_tables list
    in settings file.

    'output_tables' can specify either a list of output tables to include or to skip
    if no output_tables list is specified, then all checkpointed tables will be written

    To write all output tables EXCEPT the households and persons tables:

    ::

      output_tables:
        action: skip
        tables:
          - households
          - persons

    To write ONLY the households table:

    ::

      output_tables:
        action: include
        tables:
           - households

    To write tables into a single HDF5 store instead of individual CSVs, use the h5_store flag:

    ::

      output_tables:
        h5_store: True
        action: include
        tables:
           - households

    Parameters
    ----------
    output_dir: str

    """

    output_tables_settings_name = "output_tables"

    output_tables_settings = setting(output_tables_settings_name)

    if output_tables_settings is None:
        logger.info("No output_tables specified in settings file. Nothing to write.")
        return

    action = output_tables_settings.get("action")
    tables = output_tables_settings.get("tables")
    prefix = output_tables_settings.get("prefix", "final_")
    h5_store = output_tables_settings.get("h5_store", False)
    sort = output_tables_settings.get("sort", False)

    registered_tables = pipeline.registered_tables()
    if action == "include":
        # interpret empty or missing tables setting to mean include all registered tables
        output_tables_list = tables if tables is not None else registered_tables
    elif action == "skip":
        output_tables_list = [t for t in registered_tables if t not in tables]
    else:
        raise "expected %s action '%s' to be either 'include' or 'skip'" % (
            output_tables_settings_name,
            action,
        )

    for table_name in output_tables_list:

        if table_name == "checkpoints":
            df = pipeline.get_checkpoints()
        else:
            if table_name not in registered_tables:
                logger.warning("Skipping '%s': Table not found." % table_name)
                continue
            df = pipeline.get_table(table_name)

            if sort:
                traceable_table_indexes = inject.get_injectable(
                    "traceable_table_indexes", {}
                )

                if df.index.name in traceable_table_indexes:
                    df = df.sort_index()
                    logger.debug(
                        f"write_tables sorting {table_name} on index {df.index.name}"
                    )
                else:
                    # find all registered columns we can use to sort this table
                    # (they are ordered appropriately in traceable_table_indexes)
                    sort_columns = [
                        c for c in traceable_table_indexes if c in df.columns
                    ]
                    if len(sort_columns) > 0:
                        df = df.sort_values(by=sort_columns)
                        logger.debug(
                            f"write_tables sorting {table_name} on columns {sort_columns}"
                        )
                    else:
                        logger.debug(
                            f"write_tables sorting {table_name} on unrecognized index {df.index.name}"
                        )
                        df = df.sort_index()

        if h5_store:
            file_path = config.output_file_path("%soutput_tables.h5" % prefix)
            df.to_hdf(file_path, key=table_name, mode="a", format="fixed")
        else:
            file_name = "%s%s.csv" % (prefix, table_name)
            file_path = config.output_file_path(file_name)

            # include the index if it has a name or is a MultiIndex
            write_index = df.index.name is not None or isinstance(
                df.index, pd.MultiIndex
            )

            df.to_csv(file_path, index=write_index)
