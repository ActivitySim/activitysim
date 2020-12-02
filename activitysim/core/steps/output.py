# ActivitySim
# See full license in LICENSE.txt.
import logging
import sys
import pandas as pd

from activitysim.core import pipeline
from activitysim.core import inject
from activitysim.core import config

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

    skim_dict = inject.get_injectable('skim_dict')

    mode = 'wb' if sys.version_info < (3,) else 'w'
    with open(config.output_file_path('skim_usage.txt'), mode) as output_file:

        print("\n### skim_dict usage", file=output_file)
        for key in skim_dict.get_skim_usage():
            print(key, file=output_file)

        unused = set(k for k in skim_dict.skim_info.base_keys) - set(k for k in skim_dict.get_skim_usage())

        for key in unused:
            print(key, file=output_file)


def write_data_dictionary(output_dir):
    """
    Write table_name, number of rows, columns, and bytes for each checkpointed table

    Parameters
    ----------
    output_dir: str

    """
    pd.options.display.max_columns = 500
    pd.options.display.max_rows = 100

    output_tables = pipeline.checkpointed_tables()

    # write data dictionary for all checkpointed_tables

    mode = 'wb' if sys.version_info < (3,) else 'w'
    with open(config.output_file_path('data_dict.txt'), mode) as output_file:
        for table_name in output_tables:
            df = inject.get_table(table_name, None).to_frame()

            print("\n### %s %s" % (table_name, df.shape), file=output_file)
            print('index:', df.index.name, df.index.dtype, file=output_file)
            print(df.dtypes, file=output_file)


# def xwrite_data_dictionary(output_dir):
#     """
#     Write table_name, number of rows, columns, and bytes for each checkpointed table
#
#     Parameters
#     ----------
#     output_dir: str
#
#     """
#     pd.options.display.max_columns = 500
#     pd.options.display.max_rows = 100
#
#     checkpoints = pipeline.get_checkpoints()
#     tables = OrderedDict()
#
#     table_names = [c for c in checkpoints if c not in pipeline.NON_TABLE_COLUMNS]
#
#     with open(config.output_file_path('data_dict.txt'), 'wb') as file:
#
#         for index, row in checkpoints.iterrows():
#
#             checkpoint = row[pipeline.CHECKPOINT_NAME]
#
#             print("\n##########################################", file=file)
#             print("# %s" % checkpoint, file=file)
#             print("##########################################", file=file)
#
#             for table_name in table_names:
#
#                 if row[table_name] == '' and table_name in tables:
#                     print("\n### %s dropped %s" % (checkpoint, table_name, ), file=file)
#                     del tables[table_name]
#
#                 if row[table_name] == checkpoint:
#                     df = pipeline.get_table(table_name, checkpoint)
#                     info = tables.get(table_name, None)
#                     if info is None:
#
#                         print("\n### %s created %s %s\n" %
#                               (checkpoint, table_name, df.shape), file=file)
#
#                         print(df.dtypes, file=file)
#                         print('index:', df.index.name, df.index.dtype, file=file)
#
#                     else:
#                         new_cols = [c for c in df.columns.values if c not in info['columns']]
#                         dropped_cols = [c for c in info['columns'] if c not in df.columns.values]
#                         new_rows = df.shape[0] - info['num_rows']
#                         if new_cols:
#
#                             print("\n### %s added %s columns to %s %s\n" %
#                                   (checkpoint, len(new_cols), table_name, df.shape), file=file)
#                             print(df[new_cols].dtypes, file=file)
#
#                         if dropped_cols:
#                             print("\n### %s dropped %s columns from %s %s\n" %
#                                   (checkpoint,  len(dropped_cols), table_name, df.shape),
#                                   file=file)
#                             print(dropped_cols, file=file)
#
#                         if new_rows > 0:
#                             print("\n### %s added %s rows to %s %s" %
#                                   (checkpoint, new_rows, table_name, df.shape), file=file)
#                         elif new_rows < 0:
#                             print("\n### %s dropped %s rows from %s %s" %
#                                   (checkpoint, new_rows, table_name, df.shape), file=file)
#                         else:
#                             if not new_cols and not dropped_cols:
#                                 print("\n### %s modified %s %s" %
#                                       (checkpoint, table_name, df.shape), file=file)
#
#                     tables[table_name] = {
#                         'checkpoint_name': checkpoint,
#                         'columns': df.columns.values,
#                         'num_rows': df.shape[0]
#                     }


def write_tables(output_dir):
    """
    Write pipeline tables as csv files (in output directory) as specified by output_tables list
    in settings file.

    'output_tables' can specify either a list of output tables to include or to skip
    if no output_tables list is specified, then no checkpointed tables will be written

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

    output_tables_settings_name = 'output_tables'

    output_tables_settings = setting(output_tables_settings_name)

    if output_tables_settings is None:
        logger.info("No output_tables specified in settings file. Nothing to write.")
        return

    action = output_tables_settings.get('action')
    tables = output_tables_settings.get('tables')
    prefix = output_tables_settings.get('prefix', 'final_')
    h5_store = output_tables_settings.get('h5_store', False)
    sort = output_tables_settings.get('sort', False)

    checkpointed_tables = pipeline.checkpointed_tables()
    if action == 'include':
        output_tables_list = tables
    elif action == 'skip':
        output_tables_list = [t for t in checkpointed_tables if t not in tables]
    else:
        raise "expected %s action '%s' to be either 'include' or 'skip'" % \
              (output_tables_settings_name, action)

    for table_name in output_tables_list:

        if table_name == 'checkpoints':
            df = pipeline.get_checkpoints()
        else:
            if table_name not in checkpointed_tables:
                logger.warning("Skipping '%s': Table not found." % table_name)
                continue
            df = pipeline.get_table(table_name)

            if sort:
                traceable_table_indexes = inject.get_injectable('traceable_table_indexes', {})

                if df.index.name in traceable_table_indexes:
                    df = df.sort_index()
                    logger.debug(f"write_tables sorting {table_name} on index {df.index.name}")
                else:
                    # find all registered columns we can use to sort this table
                    # (they are ordered appropriately in traceable_table_indexes)
                    sort_columns = [c for c in traceable_table_indexes if c in df.columns]
                    if len(sort_columns) > 0:
                        df = df.sort_values(by=sort_columns)
                        logger.debug(f"write_tables sorting {table_name} on columns {sort_columns}")
                    else:
                        logger.debug(f"write_tables couldn't find a column or index to sort {table_name}"
                                     f" in traceable_table_indexes: {traceable_table_indexes}")

        if h5_store:
            file_path = config.output_file_path('%soutput_tables.h5' % prefix)
            df.to_hdf(file_path, key=table_name, mode='a', format='fixed')
        else:
            file_name = "%s%s.csv" % (prefix, table_name)
            file_path = config.output_file_path(file_name)

            # include the index if it has a name or is a MultiIndex
            write_index = df.index.name is not None or isinstance(df.index, pd.MultiIndex)

            df.to_csv(file_path, index=write_index)
