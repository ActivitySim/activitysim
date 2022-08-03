# ActivitySim
# See full license in LICENSE.txt.

import logging
import os
import warnings

import pandas as pd

from activitysim.core import config, inject, mem, util

logger = logging.getLogger(__name__)


def canonical_table_index_name(table_name):
    table_index_names = inject.get_injectable("canonical_table_index_names", None)
    return table_index_names and table_index_names.get(table_name, None)


def read_input_table(tablename, required=True):
    """Reads input table name and returns cleaned DataFrame.

    Uses settings found in input_table_list in global settings file

    Parameters
    ----------
    tablename : string

    Returns
    -------
    pandas DataFrame
    """
    table_list = config.setting("input_table_list")
    assert table_list is not None, "no input_table_list found in settings"

    table_info = None
    for info in table_list:
        if info["tablename"] == tablename:
            table_info = info

    if table_info is not None:
        df = read_from_table_info(table_info)
    else:
        if required:
            raise RuntimeError(
                f"could not find info for for tablename {tablename} in settings file"
            )
        df = None

    return df


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
    input_store = config.setting("input_store", None)
    create_input_store = config.setting("create_input_store", default=False)

    tablename = table_info.get("tablename")
    data_filename = table_info.get("filename", input_store)
    h5_tablename = table_info.get("h5_tablename") or tablename
    drop_columns = table_info.get("drop_columns", None)
    column_map = table_info.get("column_map", None)
    keep_columns = table_info.get("keep_columns", None)
    rename_columns = table_info.get("rename_columns", None)
    csv_dtypes = table_info.get("dtypes", {})

    # don't require a redundant index_col directive for canonical tables
    # but allow explicit disabling of assignment of index col for canonical tables, in which case, presumably,
    # the canonical index will be assigned in a subsequent initialization step (e.g. initialize_tours)
    canonical_index_col = canonical_table_index_name(tablename)

    # if there is an explicit index_col entry in table_info
    if "index_col" in table_info:
        # honor explicit index_col unless it conflicts with canonical name

        index_col = table_info["index_col"]

        if canonical_index_col:
            if index_col:
                # if there is a non-empty index_col directive, it should be for canonical_table_index_name
                assert (
                    index_col == canonical_index_col
                ), f"{tablename} index_col {table_info.get('index_col')} should be {index_col}"
            else:
                logger.info(
                    f"Not assigning canonical index_col {tablename}.{canonical_index_col} "
                    f"because settings file index_col directive is explicitly None."
                )

        #  if there is an index_col directive for a canonical table, it should be for canonical_table_index_name

    else:
        # otherwise default is to use canonical index name for known tables, and no index for unknown tables
        index_col = canonical_index_col

    assert tablename is not None, "no tablename provided"
    assert data_filename is not None, "no input file provided"

    data_file_path = config.data_file_path(data_filename)

    df = _read_input_file(
        data_file_path, h5_tablename=h5_tablename, csv_dtypes=csv_dtypes
    )

    # logger.debug('raw %s table columns: %s' % (tablename, df.columns.values))
    logger.debug("raw %s table size: %s" % (tablename, util.df_size(df)))

    if create_input_store:
        h5_filepath = config.output_file_path("input_data.h5")
        logger.info("writing %s to %s" % (h5_tablename, h5_filepath))
        df.to_hdf(h5_filepath, key=h5_tablename, mode="a")

        csv_dir = config.output_file_path("input_data")
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)  # make directory if needed
        df.to_csv(os.path.join(csv_dir, "%s.csv" % tablename), index=False)

    if drop_columns:
        logger.debug("dropping columns: %s" % drop_columns)
        df.drop(columns=drop_columns, inplace=True, errors="ignore")

    if column_map:
        warnings.warn(
            "table_inf option 'column_map' renamed 'rename_columns'"
            "Support for 'column_map' will be removed in future versions.",
            FutureWarning,
        )
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
            if canonical_index_col:
                # we expect canonical indexes to be integer-valued
                assert (
                    df[index_col] == df[index_col].astype(int)
                ).all(), f"Index col '{index_col}' has non-integer values"
                df[index_col] = df[index_col].astype(int)
            df.set_index(index_col, inplace=True)
        else:
            # FIXME not sure we want to do this. More likely they omitted index col than that they want to name it?
            # df.index.names = [index_col]
            logger.error(
                f"index_col '{index_col}' specified in configs but not in {tablename} table!"
            )
            logger.error(f"{tablename} columns are: {list(df.columns)}")
            raise RuntimeError(f"index_col '{index_col}' not in {tablename} table!")

    if keep_columns:
        logger.debug("keeping columns: %s" % keep_columns)
        if not set(keep_columns).issubset(set(df.columns)):
            logger.error(
                f"Required columns missing from {tablename} table: "
                f"{list(set(keep_columns).difference(set(df.columns)))}"
            )
            logger.error(f"{tablename} table has columns: {list(df.columns)}")
            raise RuntimeError(f"Required columns missing from {tablename} table")

        df = df[keep_columns]

    if df.columns.duplicated().any():
        duplicate_column_names = (
            df.columns[df.columns.duplicated(keep=False)].unique().to_list()
        )
        assert (
            not df.columns.duplicated().any()
        ), f"duplicate columns names in {tablename}: {duplicate_column_names}"

    logger.debug("%s table columns: %s" % (tablename, df.columns.values))
    logger.debug("%s table size: %s" % (tablename, util.df_size(df)))
    logger.debug("%s index name: %s" % (tablename, df.index.name))

    return df


def _read_input_file(filepath, h5_tablename=None, csv_dtypes=None):
    assert os.path.exists(filepath), "input file not found: %s" % filepath

    if filepath.endswith(".csv") or filepath.endswith(".csv.gz"):
        return _read_csv_with_fallback_encoding(filepath, csv_dtypes)

    if filepath.endswith(".h5"):
        assert h5_tablename is not None, "must provide a tablename to read HDF5 table"
        logger.info("reading %s table from %s" % (h5_tablename, filepath))
        return pd.read_hdf(filepath, h5_tablename)

    raise IOError(
        "Unsupported file type: %s. "
        "ActivitySim supports CSV and HDF5 files only" % filepath
    )


def _read_csv_with_fallback_encoding(filepath, dtypes=None):
    """read a CSV to a pandas DataFrame using default utf-8 encoding,
    but try alternate Windows-compatible cp1252 if unicode fails

    """

    try:
        logger.info("Reading CSV file %s" % filepath)
        df = pd.read_csv(filepath, comment="#", dtype=dtypes)
    except UnicodeDecodeError:
        logger.warning(
            "Reading %s with default utf-8 encoding failed, trying cp1252 instead",
            filepath,
        )
        df = pd.read_csv(filepath, comment="#", encoding="cp1252", dtype=dtypes)

    if dtypes:
        # although the dtype argument suppresses the DtypeWarning, it does not coerce recognized types (e.g. int)
        for c, dtype in dtypes.items():
            df[c] = df[c].astype(dtype)

    return df
