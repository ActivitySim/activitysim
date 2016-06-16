import os
import sys
import logging
import logging.config

import yaml

import numpy as np
import pandas as pd
import orca

TRACE_LOGGER = 'activitysim.trace'
ASIM_LOGGER = 'activitysim'

CSV_FILE_TYPE = 'csv'

LOGGING_CONF_FILE_NAME = 'logging.yaml'

tracers = {}


def delete_csv_files(output_dir):

    for the_file in os.listdir(output_dir):
        if the_file.endswith(CSV_FILE_TYPE):
            file_path = os.path.join(output_dir, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)


# for use in logging.yaml tag to inject log file path
# filename: !!python/object/apply:activitysim.defaults.tracing.log_file_path ['asim.log']
def log_file_path(name):
    output_dir = orca.get_injectable('output_dir')
    f = os.path.join(output_dir, name)
    return f


def config_logger(custom_config_file=None, basic=False):

    # if log_config_file is not supplied
    # then look for conf file in configs_dir
    # if not found use basicConfig

    log_config_file = None

    if custom_config_file and os.path.isfile(custom_config_file):
        log_config_file = custom_config_file
    elif not basic:
        # look for conf file in configs_dir
        configs_dir = orca.get_injectable('configs_dir')
        default_config_file = os.path.join(configs_dir, "configs", LOGGING_CONF_FILE_NAME)
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

    output_dir = orca.get_injectable('output_dir')
    logger.info("Deleting files in output_dir %s" % output_dir)
    delete_csv_files(output_dir)


def get_tracer(name=TRACE_LOGGER):

    tracer = logging.getLogger(name)

    if (len(tracer.handlers) == 0):

        tracer.propagate = False
        tracer.setLevel(logging.INFO)

        file_path = log_file_path('%s.log' % name)
        fileHandler = logging.FileHandler(filename=file_path, mode='w')
        tracer.addHandler(fileHandler)

        logger = logging.getLogger(ASIM_LOGGER)
        logger.info("Initialized tracer %s fileHandler %s" % (name, file_path))

    return tracer


def print_summary(label, df, describe=False, value_counts=False):

    if value_counts:
        print "\n%s choices value counts:\n%s\n" % (label, df.value_counts())

    if describe:
        print "\n%s choices summary:\n%s\n" % (label, df.describe())


def register_households(df, trace_hh_id):

    tracer = get_tracer()

    if trace_hh_id is None:
        tracer.error("register_households called with null trace_hh_id")
        return

    tracer.info("tracing household id %s in sample of %s households" % (trace_hh_id, len(df.index)))

    if trace_hh_id not in df.index:
        tracer.warn("trace_hh_id %s not in dataframe")

    # inject persons_index name of person dataframe index
    if df.index.name is None:
        df.index.names = ['household_id']
        get_tracer().warn("households table index had no name. renamed index '%s'" % df.index.name)
    orca.add_injectable("hh_index_name", df.index.name)

    tracer.info("register_households injected hh_index_name '%s'" % df.index.name)


def register_persons(df, trace_hh_id):

    tracer = get_tracer()

    if trace_hh_id is None:
        tracer.warn("register_persons called with null trace_hh_id")
        return

    # inject persons_index name of person dataframe index
    if df.index.name is None:
        df.index.names = ['person_id']
        get_tracer().warn("persons table index had no name. renamed index '%s'" % df.index.name)
    orca.add_injectable("persons_index_name", df.index.name)

    tracer.debug("register_persons injected persons_index_name '%s'" % df.index.name)

    # inject list of person_ids in household we are tracing
    # this allows us to slice by person_id without requiring presence of household_id column
    traced_persons_df = df[df['household_id'] == trace_hh_id]
    trace_person_ids = traced_persons_df.index.tolist()
    if len(trace_person_ids) == 0:
        tracer.warn("register_persons: trace_hh_id %s not found." % trace_hh_id)

    orca.add_injectable("trace_person_ids", trace_person_ids)
    tracer.info("register_persons injected trace_person_ids %s" % trace_person_ids)

    tracer.info("tracing person_ids %s in sample of %s persons" % (trace_person_ids, len(df.index)))


def write_df_csv(df, file_name, index_label=None, columns=None):

    tracer = get_tracer()

    file_path = log_file_path('%s.%s' % (file_name, CSV_FILE_TYPE))
    tracer.debug("dumping %s %s to: %s" % (type(df).__name__, file_name, file_path))

    if isinstance(df, pd.DataFrame):
        if columns:
            df = df[columns]
        df_t = df.transpose()
        if df.index.name is not None:
            df_t.index.name = df.index.name
        elif index_label:
            df_t.index.name = index_label
        df_t.to_csv(file_path, index=True, header=True)
    elif isinstance(df, pd.Series):
        print
        if isinstance(columns, str):
            df = df.rename(columns)
        elif isinstance(columns, list):
            df.index.name = columns[0]
            df = df.rename(columns[1])
        if index_label and df.index.name is None:
            df.index.name = index_label
        df.to_csv(file_path, index=True, header=True)
    else:
        tracer.error("write_df_csv object '%s' of unexpected type: %s" % (file_name, type(df)))


def slice_ids(df, ids, column=None):
    if type(ids) == int:
        ids = [ids]
    try:
        if column is None:
            df = df.loc[ids]
        else:
            df = df[df[column].isin(ids)]
    except KeyError:
        df = df[0:0]
    return df


def trace_df(df, label, slicer=None, columns=None, index_label=None, warn=True):

    tracer = get_tracer()

    # print "trace_df %s %s" % (label, type(df))

    if slicer is None:
        slicer = df.index.name

    if slicer == 'PERID' or slicer == orca.get_injectable('persons_index_name'):
        df = slice_ids(df, orca.get_injectable('trace_person_ids'))
    elif slicer == 'HHID' or slicer == orca.get_injectable('hh_index_name'):
        df = slice_ids(df, orca.get_injectable('trace_hh_id'))
    elif slicer == 'person_id':
        df = slice_ids(df, orca.get_injectable('trace_person_ids'), column='person_id')
    elif slicer == 'tour_id':
        df = slice_ids(df, orca.get_injectable('trace_person_ids'), column='person_id')
    elif slicer == 'NONE':
        pass
    else:
        get_tracer().error("trace_df: bad slicer '%s' for %s " % (slicer, label))
        print df.head(3)
        df = df[0:0]

    if len(df.index) > 0:
        write_df_csv(df, file_name=label, index_label=(index_label or slicer), columns=columns)
    elif warn:
        tracer.warn("%s: no rows with id" % (label, ))
        # write the dump file because they expect it
        write_df_csv(df, file_name=label, index_label=index_label or slicer, columns=columns)


def trace_choosers(df, label):

    trace_df(df, '%s.choosers' % label, warn=False)


def trace_utilities(df, label):

    trace_df(df, '%s.utilities' % label, warn=False)


def trace_probs(df, label):

    trace_df(df, '%s.probs' % label, warn=False)


def trace_choices(df, label):

    trace_df(df, '%s.choices' % label, warn=False)


def trace_model_design(df, label):

    trace_df(df, '%s.model_design' % label, warn=False)


def trace_cdap_hh_utils(hh_utils, label):
    # hh_util : dict of pandas.Series
    #     Keys will be household IDs and values will be Series
    #     mapping alternative choices to their utility.
    hh_id = orca.get_injectable('trace_hh_id')
    s = hh_id and hh_utils.get(hh_id, None)
    if s is not None:
        trace_df(s, label, slicer='NONE', columns=['choice', 'utility'], warn=False)


def trace_cdap_ind_utils(ind_utils, label):
    trace_df(ind_utils, label, slicer='PERID', warn=False)


def trace_nan_values(df, label):
    df = slice_ids(df, orca.get_injectable('trace_person_ids'))
    if np.isnan(df).any():
        get_tracer().warn("%s NaN values in %s" % (np.isnan(df).sum(), label))
        write_df_csv(df, "%s.nan" % label)
