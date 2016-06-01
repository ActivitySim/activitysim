import os
import logging
import logging.config

import yaml

import orca

tracer = logging.getLogger(__name__)

LOGGING_CONF_FILE_NAME = 'logging.yaml'


def config_logger():

    configs_dir = orca.get_injectable('configs_dir')
    log_config_file = os.path.join(configs_dir, "configs", LOGGING_CONF_FILE_NAME)

    if not os.path.isfile(log_config_file):
        log_config_file = LOGGING_CONF_FILE_NAME

    if not os.path.isfile(log_config_file):
        log_config_file = None

    if log_config_file:
        print "reading logging configuration from: %s" % log_config_file
        with open(log_config_file) as f:
            config_dict = yaml.load(f)
            config_dict = config_dict['logging']
            config_dict.setdefault('version', 1)
            logging.config.dictConfig(config_dict)
    else:
        print "basicConfig"
        logging.basicConfig(level=logging.INFO)


def trace_logger():
    return tracer


def trace(msg):
    tracer.info("TRACE: %s" % msg)


def print_summary(label, df, describe=False, value_counts=False):

    if value_counts:
        print "\n%s choices value counts:\n%s\n" % (label, df.value_counts())

    if describe:
        print "\n%s choices summary:\n%s\n" % (label, df.describe())
