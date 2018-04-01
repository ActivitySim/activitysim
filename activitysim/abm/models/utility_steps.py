# ActivitySim
# See full license in LICENSE.txt.

import logging
import os
import pandas as pd

from activitysim.core import pipeline
from activitysim.core import inject

from activitysim.core.config import setting

logger = logging.getLogger(__name__)


@inject.step()
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

    records = []

    # write data dictionary for all checkpointed_tables
    with open(os.path.join(output_dir, 'data_dict.txt'), 'w') as file:
        for table_name in output_tables:
            df = inject.get_table(table_name, None).to_frame()

            print >> file, "\n### %s %s" % (table_name, df.shape)
            print >> file, df.dtypes

            rows, columns = df.shape
            bytes = df.memory_usage(index=True).sum()
            records.append((table_name, rows, columns, bytes))

    df = pd.DataFrame.from_records(records, columns=['table_name', 'rows', 'columns', 'bytes'])
    df.sort_values(by='table_name', inplace=True)
    df.to_csv(os.path.join(output_dir, 'data_dict.csv'))


@inject.step()
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

    output_tables_list = pipeline.checkpointed_tables()

    if output_tables_settings is None:
        logger.info("No output_tables specified in settings file. Nothing to write.")
        return

    action = output_tables_settings.get('action')
    tables = output_tables_settings.get('tables')
    prefix = output_tables_settings.get('prefix', 'final_')

    if action not in ['include', 'skip']:
        raise "expected %s action '%s' to be either 'include' or 'skip'" % \
              (output_tables_settings_name, action)

    if action == 'include':
        output_tables_list = tables
    elif action == 'skip':
        output_tables_list = [t for t in output_tables_list if t not in tables]

    # should provide option to also write checkpoints?
    # output_tables_list.append("checkpoints.csv")

    for table_name in output_tables_list:
        table = inject.get_table(table_name, None)

        if table is None:
            logger.warn("Skipping '%s': Table not found." % table_name)
            continue

        df = table.to_frame()
        file_name = "%s%s.csv" % (prefix, table_name)
        logger.info("writing output file %s" % file_name)
        file_path = os.path.join(output_dir, file_name)
        write_index = df.index.name is not None
        df.to_csv(file_path, index=write_index)

    if (action == 'include') == ('checkpoints' in tables):
        # write checkpoints
        file_name = "%s%s.csv" % (prefix, 'checkpoints')
        pipeline.get_checkpoints().to_csv(os.path.join(output_dir, file_name))
