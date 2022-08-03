# ActivitySim
# See full license in LICENSE.txt.

import logging
import logging.config
import multiprocessing  # for process name
import os
import sys
import time
from builtins import next, range
from collections import OrderedDict

import numpy as np
import pandas as pd
import yaml

from activitysim.core import inject

from . import config

# Configurations
ASIM_LOGGER = "activitysim"
CSV_FILE_TYPE = "csv"
LOGGING_CONF_FILE_NAME = "logging.yaml"


logger = logging.getLogger(__name__)


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


def extend_trace_label(trace_label, extension):
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


def log_runtime(model_name, start_time=None, timing=None):

    assert (start_time or timing) and not (start_time and timing)

    timing = timing if timing else time.time() - start_time
    seconds = round(timing, 1)
    minutes = round(timing / 60, 1)

    process_name = multiprocessing.current_process().name

    if config.setting("multiprocess", False):
        # when benchmarking, log timing for each processes in its own log
        if config.setting("benchmarking", False):
            header = "component_name,duration"
            with config.open_log_file(
                f"timing_log.{process_name}.csv", "a", header
            ) as log_file:
                print(f"{model_name},{timing}", file=log_file)
        # only continue to log runtime in global timing log for locutor
        if not inject.get_injectable("locutor", False):
            return

    header = "process_name,model_name,seconds,minutes"
    with config.open_log_file("timing_log.csv", "a", header) as log_file:
        print(f"{process_name},{model_name},{seconds},{minutes}", file=log_file)


def delete_output_files(file_type, ignore=None, subdir=None):
    """
    Delete files in output directory of specified type

    Parameters
    ----------
    output_dir: str
        Directory of trace output CSVs

    Returns
    -------
    Nothing
    """

    output_dir = inject.get_injectable("output_dir")

    subdir = [subdir] if subdir else None
    directories = subdir or ["", "log", "trace"]

    for subdir in directories:

        dir = os.path.join(output_dir, subdir) if subdir else output_dir

        if not os.path.exists(dir):
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


def delete_trace_files():
    """
    Delete CSV files in output_dir

    Returns
    -------
    Nothing
    """
    delete_output_files(CSV_FILE_TYPE, subdir="trace")
    delete_output_files(CSV_FILE_TYPE, subdir="log")

    active_log_files = [
        h.baseFilename
        for h in logger.root.handlers
        if isinstance(h, logging.FileHandler)
    ]

    delete_output_files("log", ignore=active_log_files)


def config_logger(basic=False):
    """
    Configure logger

    look for conf file in configs_dir, if not found use basicConfig

    Returns
    -------
    Nothing
    """

    # look for conf file in configs_dir
    if basic:
        log_config_file = None
    else:
        log_config_file = config.config_file_path(
            LOGGING_CONF_FILE_NAME, mandatory=False
        )

    if log_config_file:
        try:
            with open(log_config_file) as f:
                config_dict = yaml.load(f, Loader=yaml.UnsafeLoader)
        except Exception as e:
            print(f"Unable to read logging config file {log_config_file}")
            raise e

        try:
            config_dict = config_dict["logging"]
            config_dict.setdefault("version", 1)
            logging.config.dictConfig(config_dict)
        except Exception as e:
            print(f"Unable to config logging as specified in {log_config_file}")
            raise e

    else:
        logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    logger = logging.getLogger(ASIM_LOGGER)

    if log_config_file:
        logger.info("Read logging configuration from: %s" % log_config_file)
    else:
        print("Configured logging using basicConfig")
        logger.info("Configured logging using basicConfig")


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


def initialize_traceable_tables():

    traceable_table_ids = inject.get_injectable("traceable_table_ids", {})
    if len(traceable_table_ids) > 0:
        logger.debug(
            f"initialize_traceable_tables resetting table_ids for {list(traceable_table_ids.keys())}"
        )
    inject.add_injectable("traceable_table_ids", {})


def register_traceable_table(table_name, df):
    """
    Register traceable table

    Parameters
    ----------
    df: pandas.DataFrame
        traced dataframe

    Returns
    -------
    Nothing
    """

    # add index name to traceable_table_indexes

    logger.debug(f"register_traceable_table {table_name}")

    traceable_tables = inject.get_injectable("traceable_tables", [])
    if table_name not in traceable_tables:
        logger.error("table '%s' not in traceable_tables" % table_name)
        return

    idx_name = df.index.name
    if idx_name is None:
        logger.error("Can't register table '%s' without index name" % table_name)
        return

    traceable_table_ids = inject.get_injectable("traceable_table_ids", {})
    traceable_table_indexes = inject.get_injectable("traceable_table_indexes", {})

    if (
        idx_name in traceable_table_indexes
        and traceable_table_indexes[idx_name] != table_name
    ):
        logger.error(
            "table '%s' index name '%s' already registered for table '%s'"
            % (table_name, idx_name, traceable_table_indexes[idx_name])
        )
        return

    # update traceable_table_indexes with this traceable_table's idx_name
    if idx_name not in traceable_table_indexes:
        traceable_table_indexes[idx_name] = table_name
        logger.debug(
            "adding table %s.%s to traceable_table_indexes" % (table_name, idx_name)
        )
        inject.add_injectable("traceable_table_indexes", traceable_table_indexes)

    # add any new indexes associated with trace_hh_id to traceable_table_ids

    trace_hh_id = inject.get_injectable("trace_hh_id", None)
    if trace_hh_id is None:
        return

    new_traced_ids = []
    if table_name == "households":
        if trace_hh_id not in df.index:
            logger.warning("trace_hh_id %s not in dataframe" % trace_hh_id)
            new_traced_ids = []
        else:
            logger.info(
                "tracing household id %s in %s households"
                % (trace_hh_id, len(df.index))
            )
            new_traced_ids = [trace_hh_id]
    else:

        # find first already registered ref_col we can use to slice this table
        ref_col = next((c for c in traceable_table_indexes if c in df.columns), None)

        if ref_col is None:
            logger.error(
                "can't find a registered table to slice table '%s' index name '%s'"
                " in traceable_table_indexes: %s"
                % (table_name, idx_name, traceable_table_indexes)
            )
            return

        # get traceable_ids for ref_col table
        ref_col_table_name = traceable_table_indexes[ref_col]
        ref_col_traced_ids = traceable_table_ids.get(ref_col_table_name, [])

        # inject list of ids in table we are tracing
        # this allows us to slice by id without requiring presence of a household id column
        traced_df = df[df[ref_col].isin(ref_col_traced_ids)]
        new_traced_ids = traced_df.index.tolist()
        if len(new_traced_ids) == 0:
            logger.warning(
                "register %s: no rows with %s in %s."
                % (table_name, ref_col, ref_col_traced_ids)
            )

    # update the list of trace_ids for this table
    prior_traced_ids = traceable_table_ids.get(table_name, [])

    if new_traced_ids:
        assert not set(prior_traced_ids) & set(new_traced_ids)
        traceable_table_ids[table_name] = prior_traced_ids + new_traced_ids
        inject.add_injectable("traceable_table_ids", traceable_table_ids)

    logger.debug(
        "register %s: added %s new ids to %s existing trace ids"
        % (table_name, len(new_traced_ids), len(prior_traced_ids))
    )
    logger.debug(
        "register %s: tracing new ids %s in %s"
        % (table_name, new_traced_ids, table_name)
    )


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


def write_csv(
    df, file_name, index_label=None, columns=None, column_labels=None, transpose=True
):
    """
    Print write_csv

    Parameters
    ----------
    df: pandas.DataFrame or pandas.Series
        traced dataframe
    file_name: str
        output file name
    index_label: str
        index name
    columns: list
        columns to write
    transpose: bool
        whether to transpose dataframe (ignored for series)
    Returns
    -------
    Nothing
    """

    assert len(file_name) > 0

    if not file_name.endswith(".%s" % CSV_FILE_TYPE):
        file_name = "%s.%s" % (file_name, CSV_FILE_TYPE)

    file_path = config.trace_file_path(file_name)

    if os.name == "nt":
        abs_path = os.path.abspath(file_path)
        if len(abs_path) > 255:
            msg = f"path length ({len(abs_path)}) may exceed Windows maximum length unless LongPathsEnabled: {abs_path}"
            logger.warning(msg)

    if os.path.isfile(file_path):
        logger.debug("write_csv file exists %s %s" % (type(df).__name__, file_name))

    if isinstance(df, pd.DataFrame):
        # logger.debug("dumping %s dataframe to %s" % (df.shape, file_name))
        write_df_csv(
            df, file_path, index_label, columns, column_labels, transpose=transpose
        )
    elif isinstance(df, pd.Series):
        # logger.debug("dumping %s element series to %s" % (df.shape[0], file_name))
        write_series_csv(df, file_path, index_label, columns, column_labels)
    elif isinstance(df, dict):
        df = pd.Series(data=df)
        # logger.debug("dumping %s element dict to %s" % (df.shape[0], file_name))
        write_series_csv(df, file_path, index_label, columns, column_labels)
    else:
        logger.error(
            "write_csv object for file_name '%s' of unexpected type: %s"
            % (file_name, type(df))
        )


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


def get_trace_target(df, slicer, column=None):
    """
    get target ids and column or index to identify target trace rows in df

    Parameters
    ----------
    df: pandas.DataFrame
        dataframe to slice
    slicer: str
        name of column or index to use for slicing

    Returns
    -------
    (target, column) tuple

    target : int or list of ints
        id or ids that identify tracer target rows
    column : str
        name of column to search for targets or None to search index
    """

    target_ids = None  # id or ids to slice by (e.g. hh_id or person_ids or tour_ids)

    # special do-not-slice code for dumping entire df
    if slicer == "NONE":
        return target_ids, column

    if slicer is None:
        slicer = df.index.name

    if isinstance(df, pd.DataFrame):
        # always slice by household id if we can
        if "household_id" in df.columns:
            slicer = "household_id"
        if slicer in df.columns:
            column = slicer

    if column is None and df.index.name != slicer:
        raise RuntimeError(
            "bad slicer '%s' for df with index '%s'" % (slicer, df.index.name)
        )

    traceable_table_indexes = inject.get_injectable("traceable_table_indexes", {})
    traceable_table_ids = inject.get_injectable("traceable_table_ids", {})

    if df.empty:
        target_ids = None
    elif slicer in traceable_table_indexes:
        # maps 'person_id' to 'persons', etc
        table_name = traceable_table_indexes[slicer]
        target_ids = traceable_table_ids.get(table_name, [])
    elif slicer == "zone_id":
        target_ids = inject.get_injectable("trace_od", [])

    return target_ids, column


def trace_targets(df, slicer=None, column=None):

    target_ids, column = get_trace_target(df, slicer, column)

    if target_ids is None:
        targets = None
    else:

        if column is None:
            targets = df.index.isin(target_ids)
        else:
            # convert to numpy array for consistency since that is what index.isin returns
            targets = df[column].isin(target_ids).to_numpy()

    return targets


def has_trace_targets(df, slicer=None, column=None):

    target_ids, column = get_trace_target(df, slicer, column)

    if target_ids is None:
        found = False
    else:

        if column is None:
            found = df.index.isin(target_ids).any()
        else:
            found = df[column].isin(target_ids).any()

    return found


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


def dump_df(dump_switch, df, trace_label, fname):
    if dump_switch:
        trace_label = extend_trace_label(trace_label, "DUMP.%s" % fname)
        trace_df(
            df, trace_label, index_label=df.index.name, slicer="NONE", transpose=False
        )


def trace_df(
    df,
    label,
    slicer=None,
    columns=None,
    index_label=None,
    column_labels=None,
    transpose=True,
    warn_if_empty=False,
):
    """
    Slice dataframe by traced household or person id dataframe and write to CSV

    Parameters
    ----------
    df: pandas.DataFrame
        traced dataframe
    label: str
        tracer name
    slicer: Object
        slicer for subsetting
    columns: list
        columns to write
    index_label: str
        index name
    column_labels: [str, str]
        labels for columns in csv
    transpose: boolean
        whether to transpose file for legibility
    warn_if_empty: boolean
        write warning if sliced df is empty

    Returns
    -------
    Nothing
    """

    target_ids, column = get_trace_target(df, slicer)

    if target_ids is not None:
        df = slice_ids(df, target_ids, column)

    if warn_if_empty and df.shape[0] == 0 and target_ids != []:
        column_name = column or slicer
        logger.warning(
            "slice_canonically: no rows in %s with %s == %s"
            % (label, column_name, target_ids)
        )

    if df.shape[0] > 0:
        write_csv(
            df,
            file_name=label,
            index_label=(index_label or slicer),
            columns=columns,
            column_labels=column_labels,
            transpose=transpose,
        )


def interaction_trace_rows(interaction_df, choosers, sample_size=None):
    """
    Trace model design for interaction_simulate

    Parameters
    ----------
    interaction_df: pandas.DataFrame
        traced model_design dataframe
    choosers: pandas.DataFrame
        interaction_simulate choosers
        (needed to filter the model_design dataframe by traced hh or person id)
    sample_size int or None
        int for constant sample size, or None if choosers have different numbers of alternatives
    Returns
    -------
    trace_rows : numpy.ndarray
        array of booleans to flag which rows in interaction_df to trace

    trace_ids : tuple (str,  numpy.ndarray)
        column name and array of trace_ids mapping trace_rows to their target_id
        for use by trace_interaction_eval_results which needs to know target_id
        so it can create separate tables for each distinct target for readability
    """

    # slicer column name and id targets to use for chooser id added to model_design dataframe
    # currently we only ever slice by person_id, but that could change, so we check here...

    traceable_table_ids = inject.get_injectable("traceable_table_ids", {})

    if choosers.index.name == "person_id" and "persons" in traceable_table_ids:
        slicer_column_name = choosers.index.name
        targets = traceable_table_ids["persons"]
    elif "household_id" in choosers.columns and "households" in traceable_table_ids:
        slicer_column_name = "household_id"
        targets = traceable_table_ids["households"]
    elif "person_id" in choosers.columns and "persons" in traceable_table_ids:
        slicer_column_name = "person_id"
        targets = traceable_table_ids["persons"]
    else:
        print(choosers.columns)
        raise RuntimeError(
            "interaction_trace_rows don't know how to slice index '%s'"
            % choosers.index.name
        )

    if sample_size is None:
        # if sample size not constant, we count on either
        # slicer column being in itneraction_df
        # or index of interaction_df being same as choosers
        if slicer_column_name in interaction_df.columns:
            trace_rows = np.in1d(interaction_df[slicer_column_name], targets)
            trace_ids = interaction_df.loc[trace_rows, slicer_column_name].values
        else:
            assert interaction_df.index.name == choosers.index.name
            trace_rows = np.in1d(interaction_df.index, targets)
            trace_ids = interaction_df[trace_rows].index.values

    else:

        if slicer_column_name == choosers.index.name:
            trace_rows = np.in1d(choosers.index, targets)
            trace_ids = np.asanyarray(choosers[trace_rows].index)
        elif slicer_column_name == "person_id":
            trace_rows = np.in1d(choosers["person_id"], targets)
            trace_ids = np.asanyarray(choosers[trace_rows].person_id)
        elif slicer_column_name == "household_id":
            trace_rows = np.in1d(choosers["household_id"], targets)
            trace_ids = np.asanyarray(choosers[trace_rows].household_id)
        else:
            assert False

        # simply repeat if sample size is constant across choosers
        assert sample_size == len(interaction_df.index) / len(choosers.index)
        trace_rows = np.repeat(trace_rows, sample_size)
        trace_ids = np.repeat(trace_ids, sample_size)

    assert type(trace_rows) == np.ndarray
    assert type(trace_ids) == np.ndarray

    trace_ids = (slicer_column_name, trace_ids)

    return trace_rows, trace_ids


def trace_interaction_eval_results(trace_results, trace_ids, label):
    """
    Trace model design eval results for interaction_simulate

    Parameters
    ----------
    trace_results: pandas.DataFrame
        traced model_design dataframe
    trace_ids : tuple (str,  numpy.ndarray)
        column name and array of trace_ids from interaction_trace_rows()
        used to filter the trace_results dataframe by traced hh or person id
    label: str
        tracer name

    Returns
    -------
    Nothing
    """

    assert type(trace_ids[1]) == np.ndarray

    slicer_column_name = trace_ids[0]

    trace_results[slicer_column_name] = trace_ids[1]

    targets = np.unique(trace_ids[1])

    if len(trace_results.index) == 0:
        return

    # write out the raw dataframe
    file_path = config.trace_file_path("%s.raw.csv" % label)
    trace_results.to_csv(file_path, mode="a", index=True, header=True)

    # if there are multiple targets, we want them in separate tables for readability
    for target in targets:

        df_target = trace_results[trace_results[slicer_column_name] == target]

        # we want the transposed columns in predictable order
        df_target.sort_index(inplace=True)

        # # remove the slicer (person_id or hh_id) column?
        # del df_target[slicer_column_name]

        target_label = "%s.%s.%s" % (label, slicer_column_name, target)

        trace_df(
            df_target,
            label=target_label,
            slicer="NONE",
            transpose=True,
            column_labels=["expression", None],
            warn_if_empty=False,
        )


def no_results(trace_label):
    """
    standard no-op to write tracing when a model produces no results

    """
    logger.info("Skipping %s: no_results" % trace_label)


def deregister_traceable_table(table_name):
    """
    un-register traceable table

    Parameters
    ----------
    df: pandas.DataFrame
        traced dataframe

    Returns
    -------
    Nothing
    """
    traceable_tables = inject.get_injectable("traceable_tables", [])
    traceable_table_ids = inject.get_injectable("traceable_table_ids", {})
    traceable_table_indexes = inject.get_injectable("traceable_table_indexes", {})

    if table_name not in traceable_tables:
        logger.error("table '%s' not in traceable_tables" % table_name)

    else:
        traceable_table_ids = {
            k: v for k, v in traceable_table_ids.items() if k != table_name
        }
        traceable_table_indexes = OrderedDict(
            {k: v for k, v in traceable_table_indexes.items() if v != table_name}
        )

        inject.add_injectable("traceable_table_ids", traceable_table_ids)
        inject.add_injectable("traceable_table_indexes", traceable_table_indexes)

    return
