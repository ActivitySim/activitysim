# ActivitySim
# See full license in LICENSE.txt.

import logging
import warnings
import os

import pandas as pd

from activitysim.core import (
    inject,
    config,
    util,
)
from activitysim.core import mem

logger = logging.getLogger(__name__)


def read_input_table(tablename):
    """Reads input table name and returns cleaned DataFrame.

    Uses settings found in input_table_list in global settings file

    Parameters
    ----------
    tablename : string

    Returns
    -------
    pandas DataFrame
    """
    table_list = config.setting('input_table_list')
    assert table_list is not None, 'no input_table_list found in settings'

    table_info = None
    for info in table_list:
        if info['tablename'] == tablename:
            table_info = info

    assert table_info is not None, \
        f"could not find info for for tablename {tablename} in settings file"

    return read_from_table_info(table_info)


def read_from_table_info(table_info):
    """
    Read input text files and return cleaned up DataFrame.

    table_info is a dictionary that specifies the following input params.

    See input_table_list in settings.yaml in the example folder for a working example

    +--------------+----------------------------------------------------------+
    | key          | description                                              |
    +==============+=========================================+================+
    | tablename    | name of pipeline table in which to store dataframe       |
    +--------------+----------------------------------------------------------+
    | filename     | name of csv file to read (in data_dir)                   |
    +--------------+----------------------------------------------------------+
    | column_map   | list of input columns to rename from_name: to_name       |
    +--------------+----------------------------------------------------------+
    | index_col    | name of column to set as dataframe index column          |
    +--------------+----------------------------------------------------------+
    | drop_columns | list of column names of columns to drop                  |
    +--------------+----------------------------------------------------------+
    | h5_tablename | name of target table in HDF5 file                        |
    +--------------+----------------------------------------------------------+

    """
    input_store = config.setting('input_store', None)
    create_input_store = config.setting('create_input_store', default=False)

    tablename = table_info.get('tablename')
    data_filename = table_info.get('filename', input_store)
    h5_tablename = table_info.get('h5_tablename') or tablename
    drop_columns = table_info.get('drop_columns', None)
    column_map = table_info.get('column_map', None)
    keep_columns = table_info.get('keep_columns', None)
    rename_columns = table_info.get('rename_columns', None)
    index_col = table_info.get('index_col', None)

    assert tablename is not None, 'no tablename provided'
    assert data_filename is not None, 'no input file provided'

    data_file_path = config.data_file_path(data_filename)

    df = _read_input_file(data_file_path, h5_tablename=h5_tablename)

    # logger.debug('raw %s table columns: %s' % (tablename, df.columns.values))
    logger.debug('raw %s table size: %s' % (tablename, util.df_size(df)))

    if create_input_store:
        h5_filepath = config.output_file_path('input_data.h5')
        logger.info('writing %s to %s' % (h5_tablename, h5_filepath))
        df.to_hdf(h5_filepath, key=h5_tablename, mode='a')

        csv_dir = config.output_file_path('input_data')
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)  # make directory if needed
        df.to_csv(os.path.join(csv_dir, '%s.csv' % tablename), index=False)

    if drop_columns:
        logger.debug("dropping columns: %s" % drop_columns)
        df.drop(columns=drop_columns, inplace=True, errors='ignore')

    if column_map:
        warnings.warn("table_inf option 'column_map' renamed 'rename_columns'"
                      "Support for 'column_map' will be removed in future versions.",
                      FutureWarning)
        logger.debug("renaming columns: %s" % column_map)
        df.rename(columns=column_map, inplace=True)

    # rename columns first, so keep_columns can be a stable list of expected/required columns
    if rename_columns:
        logger.debug("renaming columns: %s" % rename_columns)
        df.rename(columns=rename_columns, inplace=True)

    # set index
    if index_col is not None:
        if index_col in df.columns:
            assert not df.duplicated(index_col).any()
            df.set_index(index_col, inplace=True)
        else:
            # FIXME not sure we want to do this. More likely they omitted index col than that they want to name it?
            # df.index.names = [index_col]
            logger.error(f"index_col '{index_col}' specified in configs but not in {tablename} table!")
            logger.error(f"{tablename} columns are: {list(df.columns)}")
            raise RuntimeError(f"index_col '{index_col}' not in {tablename} table!")

    if keep_columns:
        logger.debug("keeping columns: %s" % keep_columns)
        if not set(keep_columns).issubset(set(df.columns)):
            logger.error(f"Required columns missing from {tablename} table: "
                         f"{list(set(keep_columns).difference(set(df.columns)))}")
            logger.error(f"{tablename} table has columns: {list(df.columns)}")
            raise RuntimeError(f"Required columns missing from {tablename} table")

        df = df[keep_columns]

    if df.columns.duplicated().any():
        duplicate_column_names = df.columns[df.columns.duplicated(keep=False)].unique().to_list()
        assert not df.columns.duplicated().any(), f"duplicate columns names in {tablename}: {duplicate_column_names}"

    logger.debug('%s table columns: %s' % (tablename, df.columns.values))
    logger.debug('%s table size: %s' % (tablename, util.df_size(df)))
    logger.info('%s index name: %s' % (tablename, df.index.name))

    return df


def _read_input_file(filepath, h5_tablename=None):
    assert os.path.exists(filepath), 'input file not found: %s' % filepath

    if filepath.endswith('.csv'):
        return _read_csv_with_fallback_encoding(filepath)

    if filepath.endswith('.h5'):
        assert h5_tablename is not None, 'must provide a tablename to read HDF5 table'
        logger.info('reading %s table from %s' % (h5_tablename, filepath))
        return pd.read_hdf(filepath, h5_tablename)

    raise IOError(
        'Unsupported file type: %s. '
        'ActivitySim supports CSV and HDF5 files only' % filepath)


def _read_csv_with_fallback_encoding(filepath):
    """read a CSV to a pandas DataFrame using default utf-8 encoding,
    but try alternate Windows-compatible cp1252 if unicode fails

    """

    try:
        logger.info('Reading CSV file %s' % filepath)
        return pd.read_csv(filepath, comment='#')
    except UnicodeDecodeError:
        logger.warning(
            'Reading %s with default utf-8 encoding failed, trying cp1252 instead', filepath)
        return pd.read_csv(filepath, comment='#', encoding='cp1252')
