# ActivitySim
# See full license in LICENSE.txt.

import os
import psutil
import gc

import logging
import logging.config
import sys
import time
from contextlib import contextmanager

import yaml

import numpy as np
import pandas as pd
import orca


import inject_defaults


# Configurations
ASIM_LOGGER = 'simca'
CSV_FILE_TYPE = 'csv'
LOGGING_CONF_FILE_NAME = 'logging.yaml'


logger = logging.getLogger(__name__)


def get_injectable(name, default=None):

    if orca.is_injectable(name):
        return orca.get_injectable(name)
    else:
        return default


def check_for_variability():
    return get_injectable('check_for_variability', False)


def extend_trace_label(trace_label, extension):
    if trace_label:
        trace_label = "%s.%s" % (trace_label, extension)
    return trace_label


def print_elapsed_time(msg=None, t0=None, debug=False):
    t1 = time.time()
    if msg:
        t = t1 - (t0 or t1)
        msg = "Time to execute %s : %s seconds (%s minutes)" % (msg, round(t, 3), round(t/60.0))
        if debug:
            logger.debug(msg)
        else:
            logger.info(msg)
    return t1


def delete_csv_files(output_dir):
    """
    Delete CSV files

    Parameters
    ----------
    output_dir: str
        Directory of trace output CSVs

    Returns
    -------
    Nothing
    """
    for the_file in os.listdir(output_dir):
        if the_file.endswith(CSV_FILE_TYPE):
            file_path = os.path.join(output_dir, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)


def log_file_path(name):
    """
    For use in logging.yaml tag to inject log file path

    filename: !!python/object/apply:activitysim.defaults.tracing.log_file_path ['asim.log']

    Parameters
    ----------
    name: str
        output folder name

    Returns
    -------
    f: str
        output folder name
    """
    output_dir = get_injectable('output_dir')
    f = os.path.join(output_dir, name)
    return f


def config_logger(custom_config_file=None, basic=False):
    """
    Configure logger

    if log_config_file is not supplied then look for conf file in configs_dir

    if not found use basicConfig

    Parameters
    ----------
    custom_config_file: str
        custom config filename
    basic: boolean
        basic setup

    Returns
    -------
    Nothing
    """
    log_config_file = None

    if custom_config_file and os.path.isfile(custom_config_file):
        log_config_file = custom_config_file
    elif not basic:
        # look for conf file in configs_dir
        configs_dir = get_injectable('configs_dir')
        default_config_file = os.path.join(configs_dir, LOGGING_CONF_FILE_NAME)
        if os.path.isfile(default_config_file):
            log_config_file = default_config_file

    if log_config_file:
        with open(log_config_file) as f:
            config_dict = yaml.load(f)
            config_dict = config_dict['logging']
            config_dict.setdefault('version', 1)
            logging.config.dictConfig(config_dict)
    else:
        logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    logger = logging.getLogger(ASIM_LOGGER)

    if custom_config_file and not os.path.isfile(custom_config_file):
        logger.error("#\n#\n#\nconfig_logger could not find conf file '%s'" % custom_config_file)

    if log_config_file:
        logger.info("Read logging configuration from: %s" % log_config_file)
    else:
        print "Configured logging using basicConfig"
        logger.info("Configured logging using basicConfig")

    output_dir = get_injectable('output_dir')
    logger.debug("Deleting files in output_dir %s" % output_dir)
    delete_csv_files(output_dir)


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
        print "\n%s value counts:\n%s\n" % (label, df.value_counts())

    if describe:
        print "\n%s summary:\n%s\n" % (label, df.describe())


def register_households(df, trace_hh_id):
    """
    Register with orca households for tracing

    Parameters
    ----------
    df: pandas.DataFrame
        traced dataframe

    trace_hh_id: int
        household id we are tracing

    Returns
    -------
    Nothing
    """

    logger.info("tracing household id %s in %s households" % (trace_hh_id, len(df.index)))

    if trace_hh_id not in df.index:
        logger.warn("trace_hh_id %s not in dataframe" % trace_hh_id)

    # inject persons_index name of person dataframe index
    if df.index.name is None:
        df.index.names = ['household_id']
        logger.warn("households table index had no name. renamed index '%s'" % df.index.name)
    orca.add_injectable("hh_index_name", df.index.name)

    logger.debug("register_households injected hh_index_name '%s'" % df.index.name)


def register_persons(df, trace_hh_id):
    """
    Register with orca persons for tracing

    Parameters
    ----------
    df: pandas.DataFrame
        traced dataframe

    trace_hh_id: int
        household id we are tracing

    Returns
    -------
    Nothing
    """

    # inject persons_index name of person dataframe index
    if df.index.name is None:
        df.index.names = ['person_id']
        logger.warn("persons table index had no name. renamed index '%s'" % df.index.name)
    orca.add_injectable("persons_index_name", df.index.name)

    logger.debug("register_persons injected persons_index_name '%s'" % df.index.name)

    # inject list of person_ids in household we are tracing
    # this allows us to slice by person_id without requiring presence of household_id column
    traced_persons_df = df[df['household_id'] == trace_hh_id]
    trace_person_ids = traced_persons_df.index.tolist()
    if len(trace_person_ids) == 0:
        logger.warn("register_persons: trace_hh_id %s not found." % trace_hh_id)

    orca.add_injectable("trace_person_ids", trace_person_ids)
    logger.debug("register_persons injected trace_person_ids %s" % trace_person_ids)

    logger.info("tracing person_ids %s in %s persons" % (trace_person_ids, len(df.index)))


def register_tours(df, trace_hh_id):
    """
    Register with orca persons for tracing

    create an orca injectable 'trace_tour_ids' with a list of tour_ids in household we are tracing.
    This allows us to slice by tour_id without requiring presence of person_id column

    Parameters
    ----------
    df: pandas.DataFrame
        traced dataframe

    trace_hh_id: int
        household id we are tracing

    Returns
    -------
    Nothing
    """

    # get list of persons in traced household (should already have been registered)
    person_ids = get_injectable("trace_person_ids", [])

    if len(person_ids) == 0:
        # trace_hh_id not in households table or register_persons was not not called
        logger.warn("no person ids registered for trace_hh_id %s" % trace_hh_id)
        return

    # but if household_id is in households, then we may have some tours
    traced_tours_df = slice_ids(df, person_ids, column='person_id')
    trace_tour_ids = traced_tours_df.index.tolist()
    if len(trace_tour_ids) == 0:
        logger.info("register_tours: no tours found for person_ids %s." % person_ids)
    else:
        logger.info("tracing tour_ids %s in %s tours" % (trace_tour_ids, len(df.index)))

    # register_tours is called for both mandatory and non_mandatory tours
    # so there may already be some tours registered - add the new tours to the existing list
    trace_tour_ids = get_injectable("trace_tour_ids", []) + trace_tour_ids

    orca.add_injectable("trace_tour_ids", trace_tour_ids)
    logger.debug("register_tours injected trace_tour_ids %s" % trace_tour_ids)


def register_trips(df, trace_hh_id):
    """
    Register with orca persons for tracing

    create an orca injectable 'trace_tour_ids' with a list of tour_ids in household we are tracing.
    This allows us to slice by tour_id without requiring presence of person_id column

    Parameters
    ----------
    df: pandas.DataFrame
        traced dataframe

    trace_hh_id: int
        household id we are tracin

    Returns
    -------
    Nothing
    """

    # get list of persons in traced household (should already have been registered)
    tour_ids = get_injectable("trace_tour_ids", [])

    if len(tour_ids) == 0:
        # register_persons was not not called
        logger.warn("no tour ids registered for trace_hh_id %s" % trace_hh_id)
        return

    # but if household_id is in households, then we may have some trips
    traced_trips_df = slice_ids(df, tour_ids, column='tour_id')
    trace_trip_ids = traced_trips_df.index.tolist()
    if len(traced_trips_df) == 0:
        logger.info("register_trips: no trips found for tour_ids %s." % tour_ids)
    else:
        logger.info("tracing trip_ids %s in %s trips" % (trace_trip_ids, len(df.index)))

    orca.add_injectable("trace_trip_ids", trace_trip_ids)
    logger.debug("register_trips injected trace_tour_ids %s" % trace_trip_ids)


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

    trace_hh_id = get_injectable("trace_hh_id", None)

    if trace_hh_id is None:
        return

    if table_name == 'households':
        register_households(df, trace_hh_id)
    elif table_name == 'persons':
        register_persons(df, trace_hh_id)
    elif table_name == 'trips':
        register_trips(df, trace_hh_id)
    elif table_name in ["non_mandatory_tours", "mandatory_tours"]:
        register_tours(df, trace_hh_id)


def sort_for_registration(table_names):

    # names of all traceable tables ordered by dependency on household_id
    # e.g. 'persons' has to be registered AFTER 'households'
    preferred_order = ['households', 'persons', 'non_mandatory_tours', 'mandatory_tours', 'trips']

    table_names = list(table_names)

    for table_name in reversed(preferred_order):
        if table_name in table_names:
            # move it to the end of the list
            table_names.remove(table_name)
            table_names.append(table_name)

    return reversed(table_names)


def write_df_csv(df, file_path, index_label=None, columns=None, column_labels=None, transpose=True):

    mode = 'a' if os.path.isfile(file_path) else 'w'

    if columns:
        df = df[columns]

    if not transpose:
        df.to_csv(file_path, mode="a", index=True, header=True)
        return

    df_t = df.transpose()
    if df.index.name is not None:
        df_t.index.name = df.index.name
    elif index_label:
        df_t.index.name = index_label

    with open(file_path, mode=mode) as f:
        if column_labels is None:
            column_labels = [None, None]
        if column_labels[0] is None:
            column_labels[0] = 'label'
        if column_labels[1] is None:
            column_labels[1] = 'value'

        if len(df_t.columns) == len(column_labels) - 1:
            column_label_row = ','.join(column_labels)
        else:
            column_label_row = \
                column_labels[0] + ',' \
                + ','.join([column_labels[1] + '_' + str(i+1) for i in range(len(df_t.columns))])

        if mode == 'a':
            column_label_row = '# ' + column_label_row
        f.write(column_label_row + '\n')
    df_t.to_csv(file_path, mode='a', index=True, header=True)


def write_series_csv(series, file_path, index_label=None, columns=None, column_labels=None):

    if isinstance(columns, str):
        series = series.rename(columns)
    elif isinstance(columns, list):
        if columns[0]:
            series.index.name = columns[0]
        series = series.rename(columns[1])
    if index_label and series.index.name is None:
        series.index.name = index_label
    series.to_csv(file_path, mode='a', index=True, header=True)


def write_csv(df, file_name, index_label=None, columns=None, column_labels=None, transpose=True):
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

    file_path = log_file_path('%s.%s' % (file_name, CSV_FILE_TYPE))

    if os.path.isfile(file_path):
        logger.error("write_csv file exists %s %s" % (type(df).__name__, file_name))

    if isinstance(df, pd.DataFrame):
        logger.debug("dumping %s dataframe to %s" % (df.shape, file_name))
        write_df_csv(df, file_path, index_label, columns, column_labels, transpose=transpose)
    elif isinstance(df, pd.Series):
        logger.debug("dumping %s element series to %s" % (len(df.index), file_name))
        write_series_csv(df, file_path, index_label, columns, column_labels)
    else:
        logger.error("write_df_csv object '%s' of unexpected type: %s" % (file_name, type(df)))


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

    if not isinstance(ids, (list, tuple)):
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


def get_trace_target(df, slicer):
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

    if slicer is None:
        slicer = df.index.name

    target_ids = None  # id or ids to slice by (e.g. hh_id or person_ids or tour_ids)
    column = None  # column name to slice on or None to slice on index

    if len(df.index) == 0:
        target_ids = None
    elif slicer == 'PERID' or slicer == get_injectable('persons_index_name'):
        target_ids = get_injectable('trace_person_ids', [])
    elif slicer == 'HHID' or slicer == orca.get_injectable('hh_index_name'):
        target_ids = get_injectable('trace_hh_id', [])
    elif slicer == 'person_id':
        target_ids = get_injectable('trace_person_ids', [])
        column = slicer
    elif slicer == 'hh_id':
        target_ids = get_injectable('trace_hh_id', [])
        column = slicer
    elif slicer == 'tour_id':
        if isinstance(df, pd.DataFrame) and ('person_id' in df.columns):
            target_ids = get_injectable('trace_person_ids', [])
            column = 'person_id'
        else:
            target_ids = get_injectable('trace_tour_ids', [])
    elif slicer == 'trip_id':  # FIX ME
        if isinstance(df, pd.DataFrame) and ('person_id' in df.columns):
            target_ids = get_injectable('trace_person_ids', [])
            column = 'person_id'
        else:
            target_ids = get_injectable('trace_trip_ids', [])
    elif slicer == 'TAZ' or slicer == 'ZONE':
        target_ids = get_injectable('trace_od', [])
    elif slicer == 'NONE':
        target_ids = None
    else:
        raise RuntimeError("slice_canonically: bad slicer '%s'" % (slicer, ))

    if target_ids and not isinstance(target_ids, (list, tuple)):
        target_ids = [target_ids]

    return target_ids, column


def slice_canonically(df, slicer, label, warn_if_empty=False):
    """
    Slice dataframe by traced household or person id dataframe and write to CSV

    Parameters
    ----------
    df: pandas.DataFrame
        dataframe to slice
    slicer: str
        name of column or index to use for slicing
    label: str
        tracer name - only used to report bad slicer

    Returns
    -------
    sliced subset of dataframe
    """

    target_ids, column = get_trace_target(df, slicer)

    if target_ids is not None:
        df = slice_ids(df, target_ids, column)

    if warn_if_empty and len(df.index) == 0:
        column_name = column or slicer
        logger.warn("slice_canonically: no rows in %s with %s == %s"
                    % (label, column_name, target_ids))

    return df


def has_trace_targets(df, slicer=None):

    target_ids, column = get_trace_target(df, slicer)

    if target_ids is None:
        found = False
    else:

        if column is None:
            found = df.index.isin(target_ids).any()
        else:
            found = df[column].isin(target_ids).any()

    return found


def hh_id_for_chooser(id, choosers):

    if choosers.index.name == 'HHID' or choosers.index.name == get_injectable('hh_index_name'):
        hh_id = id
    elif 'household_id' in choosers.columns:
        hh_id = choosers.loc[id]['household_id']
    else:
        raise RuntimeError("don't grok chooser with index %s" % choosers.index.name)

    return hh_id


def trace_df(df, label, slicer=None, columns=None,
             index_label=None, column_labels=None, transpose=True, warn_if_empty=False):
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

    df = slice_canonically(df, slicer, label, warn_if_empty)

    if len(df.index) > 0:
        write_csv(df, file_name=label, index_label=(index_label or slicer), columns=columns,
                  column_labels=column_labels, transpose=transpose)


def interaction_trace_rows(interaction_df, choosers):
    """
    Trace model design for interaction_simulate

    Parameters
    ----------
    model_design: pandas.DataFrame
        traced model_design dataframe
    choosers: pandas.DataFrame
        interaction_simulate choosers
        (needed to filter the model_design dataframe by traced hh or person id)

    Returns
    -------
    trace_rows : numpy.ndarray
        array of booleans to select values in eval_interaction_utilities df to trace

    trace_ids : tuple (str,  numpy.ndarray)
        column name and array of trace_ids for use by


    """

    # slicer column name and id targets to use for chooser id added to model_design dataframe
    # currently we only ever slice by person_id, but that could change, so we check here...

    if choosers.index.name == 'PERID' \
            or choosers.index.name == get_injectable('persons_index_name'):
        slicer_column_name = choosers.index.name
        targets = get_injectable('trace_person_ids', [])
    elif (choosers.index.name == 'tour_id' and 'person_id' in choosers.columns):
        slicer_column_name = 'person_id'
        targets = get_injectable('trace_person_ids', [])
    else:
        raise RuntimeError("trace_interaction_model_design don't know how to slice index '%s'"
                           % choosers.index.name)

    # we can deduce the sample_size from the relative size of model_design and choosers
    # (model design rows are repeated once for each alternative)
    sample_size = len(interaction_df.index) / len(choosers.index)

    if slicer_column_name == choosers.index.name:
        trace_rows = np.in1d(choosers.index, targets)
        trace_ids = np.asanyarray(choosers[trace_rows].index)
    else:
        trace_rows = np.in1d(choosers['person_id'], targets)
        trace_ids = np.asanyarray(choosers[trace_rows].person_id)

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
    trace_ids: pandas.DataFrame
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
    file_path = log_file_path('%s.raw.csv' % label)
    trace_results.to_csv(file_path, mode="a", index=True, header=True)

    # if there are multiple targets, we want them in separate tables for readability
    for target in targets:

        df_target = trace_results[trace_results[slicer_column_name] == target]

        # we want the transposed columns in predictable order
        df_target.sort_index(inplace=True)

        # # remove the slicer (person_id or hh_id) column?
        # del df_target[slicer_column_name]

        target_label = '%s.%s.%s' % (label, slicer_column_name, target)
        trace_df(df_target, target_label,
                 slicer="NONE",
                 transpose=True,
                 column_labels=['expression', None],
                 warn_if_empty=False)
