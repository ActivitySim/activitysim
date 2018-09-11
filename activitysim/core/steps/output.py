# ActivitySim
# See full license in LICENSE.txt.

import logging
import os
import pandas as pd

from collections import OrderedDict

from activitysim.core import pipeline
from activitysim.core import inject
from activitysim.core import config

from activitysim.core.config import setting

logger = logging.getLogger(__name__)


def track_skim_usage(output_dir):
    """
    write statistics on skim usage (diagnostic to detect loading of un-needed skims)

    FIXME - have not yet implemented a facility to avoid loading of unused skims

    Parameters
    ----------
    output_dir: str

    """
    pd.options.display.max_columns = 500
    pd.options.display.max_rows = 100

    checkpoints = pipeline.get_checkpoints()
    tables = OrderedDict()

    skim_dict = inject.get_injectable('skim_dict')
    skim_stack = inject.get_injectable('skim_stack', None)

    with open(config.output_file_path('skim_usage.txt'), 'w') as file:

        print >> file, "\n### skim_dict usage"
        for key in skim_dict.usage:
            print>> file, key

        if skim_stack is None:
            unused_keys = {k for k in skim_dict.skim_key_to_skim_num} - {k for k in skim_dict.usage}

            print >> file, "\n### unused skim keys"
            for key in unused_keys:
                print>> file, key

        else:

            print >> file, "\n### skim_stack usage"
            for key in skim_stack.usage:
                print>> file, key

            unused = {k for k in skim_dict.skim_key_to_skim_num if not isinstance(k, tuple)} - \
                     {k for k in skim_dict.usage if not isinstance(k, tuple)}
            print >> file, "\n### unused skim str keys"
            for key in unused:
                print>> file, key

                unused = {k[0] for k in skim_dict.skim_key_to_skim_num if isinstance(k, tuple)} - \
                         {k[0] for k in skim_dict.usage if isinstance(k, tuple)} - \
                         {k for k in skim_stack.usage}
            print >> file, "\n### unused skim dim3 keys"
            for key in unused:
                print>> file, key


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

    with open(config.output_file_path('data_dict.txt'), 'w') as file:
        for table_name in output_tables:
            df = inject.get_table(table_name, None).to_frame()

            print >> file, "\n### %s %s" % (table_name, df.shape)
            print >> file, 'index:', df.index.name, df.index.dtype
            print >> file, df.dtypes


# def write_data_dictionary(output_dir):
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
#     with open(config.output_file_path('data_dict.txt'), 'w') as file:
#
#         for index, row in checkpoints.iterrows():
#
#             checkpoint = row[pipeline.CHECKPOINT_NAME]
#
#             print >> file, "\n##########################################"
#             print >> file, "# %s" % checkpoint
#             print >> file, "##########################################"
#
#             for table_name in table_names:
#
#                 if row[table_name] == '' and table_name in tables:
#                     print >> file, "\n### %s dropped %s" % (checkpoint, table_name, )
#                     del tables[table_name]
#
#                 if row[table_name] == checkpoint:
#                     df = pipeline.get_table(table_name, checkpoint)
#                     info = tables.get(table_name, None)
#                     if info is None:
#
#                         print >> file, "\n### %s created %s %s\n" % \
#                                        (checkpoint, table_name, df.shape)
#
#                         print >> file, df.dtypes
#                         print >> file, 'index:', df.index.name, df.index.dtype
#
#                     else:
#                         new_cols = [c for c in df.columns.values if c not in info['columns']]
#                         dropped_cols = [c for c in info['columns'] if c not in df.columns.values]
#                         new_rows = df.shape[0] - info['num_rows']
#                         if new_cols:
#
#                             print >> file, "\n### %s added %s columns to %s %s\n" % \
#                                            (checkpoint, len(new_cols), table_name, df.shape)
#                             print >> file, df[new_cols].dtypes
#
#                         if dropped_cols:
#                             print >> file, "\n### %s dropped %s columns from %s %s\n" % \
#                                            (checkpoint,  len(dropped_cols), table_name, df.shape)
#                             print >> file, dropped_cols
#
#                         if new_rows > 0:
#                             print >> file, "\n### %s added %s rows to %s %s" % \
#                                            (checkpoint, new_rows, table_name, df.shape)
#                         elif new_rows < 0:
#                             print >> file, "\n### %s dropped %s rows from %s %s" % \
#                                            (checkpoint, new_rows, table_name, df.shape)
#                         else:
#                             if not new_cols and not dropped_cols:
#                                 print >> file, "\n### %s modified %s %s" % \
#                                                (checkpoint, table_name, df.shape)
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

    if action not in ['include', 'skip']:
        raise "expected %s action '%s' to be either 'include' or 'skip'" % \
              (output_tables_settings_name, action)

    checkpointed_tables = pipeline.checkpointed_tables()
    if action == 'include':
        output_tables_list = tables
    elif action == 'skip':
        output_tables_list = [t for t in checkpointed_tables if t not in tables]

    for table_name in output_tables_list:

        if table_name == 'checkpoints':
            df = pipeline.get_checkpoints()
        else:
            if table_name not in checkpointed_tables:
                logger.warn("Skipping '%s': Table not found." % table_name)
                continue
            df = pipeline.get_table(table_name)

        file_name = "%s%s.csv" % (prefix, table_name)
        file_path = config.output_file_path(file_name)

        # include the index if it has a name or is a MultiIndex
        write_index = df.index.name is not None or isinstance(df.index, pd.core.index.MultiIndex)

        df.to_csv(file_path, index=write_index)
