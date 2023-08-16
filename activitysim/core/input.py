# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging
import os

import pandas as pd

from activitysim.core import util, workflow
from activitysim.core.configuration import InputTable
from activitysim.core.exceptions import MissingInputTableDefinition

logger = logging.getLogger(__name__)


def canonical_table_index_name(table_name):
    from activitysim.abm.models.util import canonical_ids

    table_index_names = canonical_ids.CANONICAL_TABLE_INDEX_NAMES
    return table_index_names and table_index_names.get(table_name, None)


def read_input_table(state: workflow.State, tablename, required=True):
    """Reads input table name and returns cleaned DataFrame.

    Uses settings found in input_table_list in global settings file

    Parameters
    ----------
    tablename : string
    settings : State

    Returns
    -------
    pandas DataFrame
    """
    table_list = state.settings.input_table_list
    if required and table_list is None:
        raise AssertionError("no input_table_list found in settings")
    if not required and table_list is None:
        return None

    table_info = None
    for info in table_list:
        if info.tablename == tablename:
            table_info = info

    if table_info is not None:
        df = read_from_table_info(table_info, state)
    else:
        if required:
            raise MissingInputTableDefinition(
                f"could not find info for for tablename {tablename} in settings file"
            )
        df = None

    return df


def read_from_table_info(table_info: InputTable, state):
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
    | index_col    | name of column to set as dataframe index column          |
    +--------------+----------------------------------------------------------+
    | h5_tablename | name of target table in HDF5 file                        |
    +--------------+----------------------------------------------------------+

    """
    input_store = state.settings.input_store
    create_input_store = state.settings.create_input_store

    tablename = table_info.tablename
    data_filename = table_info.filename or input_store
    h5_tablename = table_info.h5_tablename or tablename
    keep_columns = table_info.keep_columns
    drop_columns = table_info.drop_columns
    rename_columns = table_info.rename_columns
    recode_columns = table_info.recode_columns
    csv_dtypes = table_info.dtypes or {}

    # don't require a redundant index_col directive for canonical tables
    # but allow explicit disabling of assignment of index col for canonical tables, in which case, presumably,
    # the canonical index will be assigned in a subsequent initialization step (e.g. initialize_tours)
    canonical_index_col = canonical_table_index_name(tablename)

    # if there is an explicit index_col entry in table_info
    if table_info.index_col != "NOTSET":
        # honor explicit index_col unless it conflicts with canonical name

        index_col = table_info.index_col

        if canonical_index_col:
            if index_col:
                # if there is a non-empty index_col directive, it should be for canonical_table_index_name
                assert (
                    index_col == canonical_index_col
                ), f"{tablename} index_col {table_info.index_col} should be {index_col}"
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

    data_file_path = state.filesystem.get_data_file_path(
        data_filename, alternative_suffixes=(".csv.gz", ".parquet")
    )

    df = read_input_file(
        str(data_file_path), h5_tablename=h5_tablename, csv_dtypes=csv_dtypes
    )

    # logger.debug('raw %s table columns: %s' % (tablename, df.columns.values))
    logger.debug(f"raw {tablename} table size: {util.df_size(df)}")

    if create_input_store:
        raise NotImplementedError("the input store functionality has been disabled")
        # h5_filepath = state.get_output_file_path("input_data.h5")
        # logger.info("writing %s to %s" % (h5_tablename, h5_filepath))
        # df.to_hdf(h5_filepath, key=h5_tablename, mode="a")
        #
        # csv_dir = state.get_output_file_path("input_data")
        # if not os.path.exists(csv_dir):
        #     os.makedirs(csv_dir)  # make directory if needed
        # df.to_csv(os.path.join(csv_dir, "%s.csv" % tablename), index=False)

    if drop_columns:
        logger.debug("dropping columns: %s" % drop_columns)
        df.drop(columns=drop_columns, inplace=True, errors="ignore")

    # rename columns first, so keep_columns can be a stable list of expected/required columns
    if rename_columns:
        logger.debug("renaming columns: %s" % rename_columns)
        df.rename(columns=rename_columns, inplace=True)

    # recode columns, can simplify data structure
    if recode_columns and state.settings.recode_pipeline_columns:
        for colname, recode_instruction in recode_columns.items():
            logger.info(f"recoding column {colname}: {recode_instruction}")
            if recode_instruction == "zero-based":
                if f"_original_{colname}" in df:
                    # a recoding of this column has already been completed
                    # just need to confirm it is zero-based
                    if (df[colname] != pd.RangeIndex(len(df))).any():
                        raise ValueError("column already recoded as non-zero-based")
                else:
                    remapper = {j: i for i, j in enumerate(sorted(set(df[colname])))}
                    df[f"_original_{colname}"] = df[colname]
                    df[colname] = df[colname].apply(remapper.get)
                    if keep_columns and f"_original_{colname}" not in keep_columns:
                        keep_columns.append(f"_original_{colname}")
                    if tablename == "land_use" and colname == canonical_index_col:
                        # We need to keep track if we have recoded the land_use
                        # table's index to zero-based, as we need to disable offset
                        # processing for legacy skim access.
                        state.settings.offset_preprocessing = True
            else:
                source_table, lookup_col = recode_instruction.split(".")
                parent_table = state.get_dataframe(source_table)
                try:
                    map_col = parent_table[f"_original_{lookup_col}"]
                except KeyError:
                    map_col = parent_table[lookup_col]
                remapper = dict(zip(map_col, parent_table.index))
                df[colname] = df[colname].apply(remapper.get)

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

    logger.debug(f"{tablename} table columns: {df.columns.values}")
    logger.debug(f"{tablename} table size: {util.df_size(df)}")
    logger.debug(f"{tablename} index name: {df.index.name}")

    return df


def read_input_file(filepath: str, h5_tablename: str = None, csv_dtypes=None):
    """
    Read data to a pandas DataFrame, inferring file type from filename extension.
    """

    filepath = str(filepath)
    assert os.path.exists(filepath), "input file not found: %s" % filepath

    if filepath.endswith(".csv") or filepath.endswith(".csv.gz"):
        return _read_csv_with_fallback_encoding(filepath, csv_dtypes)

    if filepath.endswith(".h5"):
        assert h5_tablename is not None, "must provide a tablename to read HDF5 table"
        logger.info(f"reading {h5_tablename} table from {filepath}")
        return pd.read_hdf(filepath, h5_tablename)

    if filepath.endswith(".parquet"):
        return pd.read_parquet(filepath)

    raise OSError(
        "Unsupported file type: %s. "
        "ActivitySim supports CSV, HDF5, and Parquet files only" % filepath
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
