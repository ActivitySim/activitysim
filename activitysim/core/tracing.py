# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging
import logging.config
import os
import time
from builtins import range

import numpy as np
import pandas as pd

# Configurations
ASIM_LOGGER = "activitysim"
CSV_FILE_TYPE = "csv"
LOGGING_CONF_FILE_NAME = "logging.yaml"

logging.INFO
logger = logging.getLogger(__name__)

timing_notes = set()


class ElapsedTimeFormatter(logging.Formatter):
    def format(self, record):
        duration_milliseconds = record.relativeCreated
        hours, rem = divmod(duration_milliseconds / 1000, 3600)
        minutes, seconds = divmod(rem, 60)
        if hours:
            record.elapsedTime = "{:0>2}:{:0>2}:{:05.2f}".format(
                int(hours), int(minutes), seconds
            )
        else:
            record.elapsedTime = "{:0>2}:{:05.2f}".format(int(minutes), seconds)
        return super(ElapsedTimeFormatter, self).format(record)


def extend_trace_label(trace_label: str = None, extension: str = None) -> str | None:
    if trace_label:
        trace_label = "%s.%s" % (trace_label, extension)
    return trace_label


def format_elapsed_time(t):
    return "%s seconds (%s minutes)" % (round(t, 3), round(t / 60.0, 1))


def print_elapsed_time(msg=None, t0=None, debug=False):
    t1 = time.time()
    if msg:
        assert t0 is not None
        t = t1 - (t0 or t1)
        msg = "Time to execute %s : %s" % (msg, format_elapsed_time(t))
        if debug:
            logger.debug(msg)
        else:
            logger.info(msg)
    return t1


def delete_output_files(whale, file_type, ignore=None, subdir=None):
    """
    Delete files in output directory of specified type.

    Parameters
    ----------
    whale : Pipeline
        The output directory is read from the Pipeline.
    file_type : str
        File extension to delete.
    ignore : list[Path-like]
        Specific files to leave alone.
    subdir : list[Path-like], optional
        Subdirectories to scrub.  If not given, the top level output directory
        plus the 'log' and 'trace' directories will be scrubbed.
    """

    output_dir = whale.filesystem.get_output_dir()

    subdir = [subdir] if subdir else None
    directories = subdir or ["", "log", "trace"]

    for subdir in directories:
        dir = output_dir.joinpath(output_dir, subdir) if subdir else output_dir

        if not dir.exists():
            continue

        if ignore:
            ignore = [os.path.realpath(p) for p in ignore]

        # logger.debug("Deleting %s files in output dir %s" % (file_type, dir))

        for the_file in os.listdir(dir):
            if the_file.endswith(file_type):
                file_path = os.path.join(dir, the_file)

                if ignore and os.path.realpath(file_path) in ignore:
                    continue

                try:
                    if os.path.isfile(file_path):
                        logger.debug("delete_output_files deleting %s" % file_path)
                        os.unlink(file_path)
                except Exception as e:
                    print(e)


def delete_trace_files(whale):
    """
    Delete CSV files in output_dir
    """
    delete_output_files(whale, CSV_FILE_TYPE, subdir="trace")
    delete_output_files(whale, CSV_FILE_TYPE, subdir="log")

    active_log_files = [
        h.baseFilename
        for h in logger.root.handlers
        if isinstance(h, logging.FileHandler)
    ]

    delete_output_files(whale, "log", ignore=active_log_files)


def print_summary(label, df, describe=False, value_counts=False):
    """
    Print summary

    Parameters
    ----------
    label: str
        tracer name
    df: pandas.DataFrame
        traced dataframe
    describe: boolean
        print describe?
    value_counts: boolean
        print value counts?

    Returns
    -------
    Nothing
    """

    if not (value_counts or describe):
        logger.error("print_summary neither value_counts nor describe")

    if value_counts:
        n = 10
        logger.info(
            "%s top %s value counts:\n%s" % (label, n, df.value_counts().nlargest(n))
        )

    if describe:
        logger.info("%s summary:\n%s" % (label, df.describe()))


def write_df_csv(
    df, file_path, index_label=None, columns=None, column_labels=None, transpose=True
):
    need_header = not os.path.isfile(file_path)

    if columns:
        df = df[columns]

    if not transpose:
        want_index = isinstance(df.index, pd.MultiIndex) or df.index.name is not None
        df.to_csv(file_path, mode="a", index=want_index, header=need_header)
        return

    df_t = df.transpose() if df.index.name in df else df.reset_index().transpose()

    if index_label:
        df_t.index.name = index_label

    if need_header:
        if column_labels is None:
            column_labels = [None, None]
        if column_labels[0] is None:
            column_labels[0] = "label"
        if column_labels[1] is None:
            column_labels[1] = "value"

        if len(df_t.columns) == len(column_labels) - 1:
            column_label_row = ",".join(column_labels)
        else:
            column_label_row = (
                column_labels[0]
                + ","
                + ",".join(
                    [
                        column_labels[1] + "_" + str(i + 1)
                        for i in range(len(df_t.columns))
                    ]
                )
            )

        with open(file_path, mode="a") as f:
            f.write(column_label_row + "\n")

    df_t.to_csv(file_path, mode="a", index=True, header=False)


def write_series_csv(
    series, file_path, index_label=None, columns=None, column_labels=None
):
    if isinstance(columns, str):
        series = series.rename(columns)
    elif isinstance(columns, list):
        if columns[0]:
            series.index.name = columns[0]
        series = series.rename(columns[1])
    if index_label and series.index.name is None:
        series.index.name = index_label

    need_header = not os.path.isfile(file_path)
    series.to_csv(file_path, mode="a", index=True, header=need_header)


# def write_csv(
#     whale: workflow.Whale,
#     df,
#     file_name,
#     index_label=None,
#     columns=None,
#     column_labels=None,
#     transpose=True,
# ):
#     """
#     Print write_csv
#
#     Parameters
#     ----------
#     df: pandas.DataFrame or pandas.Series or dict
#         traced dataframe
#     file_name: str
#         output file name
#     index_label: str
#         index name
#     columns: list
#         columns to write
#     transpose: bool
#         whether to transpose dataframe (ignored for series)
#     Returns
#     -------
#     Nothing
#     """
#
#     assert len(file_name) > 0
#
#     if not file_name.endswith(".%s" % CSV_FILE_TYPE):
#         file_name = "%s.%s" % (file_name, CSV_FILE_TYPE)
#
#     file_path = whale.filesystem.get_trace_file_path(file_name)
#
#     if os.name == "nt":
#         abs_path = os.path.abspath(file_path)
#         if len(abs_path) > 255:
#             msg = f"path length ({len(abs_path)}) may exceed Windows maximum length unless LongPathsEnabled: {abs_path}"
#             logger.warning(msg)
#
#     if os.path.isfile(file_path):
#         logger.debug("write_csv file exists %s %s" % (type(df).__name__, file_name))
#
#     if isinstance(df, pd.DataFrame):
#         # logger.debug("dumping %s dataframe to %s" % (df.shape, file_name))
#         write_df_csv(
#             df, file_path, index_label, columns, column_labels, transpose=transpose
#         )
#     elif isinstance(df, pd.Series):
#         # logger.debug("dumping %s element series to %s" % (df.shape[0], file_name))
#         write_series_csv(df, file_path, index_label, columns, column_labels)
#     elif isinstance(df, dict):
#         df = pd.Series(data=df)
#         # logger.debug("dumping %s element dict to %s" % (df.shape[0], file_name))
#         write_series_csv(df, file_path, index_label, columns, column_labels)
#     else:
#         logger.error(
#             "write_csv object for file_name '%s' of unexpected type: %s"
#             % (file_name, type(df))
#         )


def slice_ids(df, ids, column=None):
    """
    slice a dataframe to select only records with the specified ids

    Parameters
    ----------
    df: pandas.DataFrame
        traced dataframe
    ids: int or list of ints
        slice ids
    column: str
        column to slice (slice using index if None)

    Returns
    -------
    df: pandas.DataFrame
        sliced dataframe
    """

    if np.isscalar(ids):
        ids = [ids]

    try:
        if column is None:
            df = df[df.index.isin(ids)]
        else:
            df = df[df[column].isin(ids)]
    except KeyError:
        # this happens if specified slicer column is not in df
        # df = df[0:0]
        raise RuntimeError("slice_ids slicer column '%s' not in dataframe" % column)

    return df


def hh_id_for_chooser(id, choosers):
    """

    Parameters
    ----------
    id - scalar id (or list of ids) from chooser index
    choosers - pandas dataframe whose index contains ids

    Returns
    -------
        scalar household_id or series of household_ids
    """

    if choosers.index.name == "household_id":
        hh_id = id
    elif "household_id" in choosers.columns:
        hh_id = choosers.loc[id]["household_id"]
    else:
        print(": hh_id_for_chooser: nada:\n%s" % choosers.columns)
        hh_id = None

    return hh_id


def trace_id_for_chooser(id, choosers):
    """

    Parameters
    ----------
    id - scalar id (or list of ids) from chooser index
    choosers - pandas dataframe whose index contains ids

    Returns
    -------
        scalar household_id or series of household_ids
    """

    hh_id = None
    for column_name in ["household_id", "person_id"]:
        if choosers.index.name == column_name:
            hh_id = id
            break
        elif column_name in choosers.columns:
            hh_id = choosers.loc[id][column_name]
            break

    if hh_id is None:
        print(": hh_id_for_chooser: nada:\n%s" % choosers.columns)

    return hh_id, column_name


def no_results(trace_label):
    """
    standard no-op to write tracing when a model produces no results

    """
    logger.info("Skipping %s: no_results" % trace_label)
