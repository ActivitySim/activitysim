import os
import sys
import logging
import logging.config

import yaml

import orca

TRACE_LOGGER = __name__
ASIM_LOGGER = 'activitysim'

tracer = logging.getLogger(TRACE_LOGGER)

LOGGING_CONF_FILE_NAME = 'logging.yaml'


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


def trace_logger():
    return tracer


def trace(msg):
    tracer.info("TRACE: %s" % msg)


def print_summary(label, df, describe=False, value_counts=False):

    if value_counts:
        print "\n%s choices value counts:\n%s\n" % (label, df.value_counts())

    if describe:
        print "\n%s choices summary:\n%s\n" % (label, df.describe())
