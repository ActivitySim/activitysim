# ActivitySim
# See full license in LICENSE.txt.

import argparse
import os
import yaml

import logging
from activitysim.core import inject

logger = logging.getLogger(__name__)


def handle_standard_args(parser=None):
    """
    Adds 'standard' activitysim arguments:
        --config : specify path to config_dir
        --output : specify path to output_dir
        --data   : specify path to data_dir

    Parameters
    ----------
    parser : argparse.ArgumentParser or None
        to  custom argument handling, pass in a parser with arguments added
        and handle them based on returned args. This method will hand the args it adds
    Returns
    -------

    args : parser.parse_args() result
    """

    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", help="path to config dir")
    parser.add_argument("-o", "--output", help="path to output dir")
    parser.add_argument("-d", "--data", help="path to data dir")
    parser.add_argument("-r", "--resume", help="resume after")
    parser.add_argument("-m", "--models", help="models run_list_name in settings")
    args = parser.parse_args()

    if args.config:
        if not os.path.exists(args.config):
            raise IOError("Could not find configs dir '%s'." % args.config)
        inject.add_injectable("configs_dir", args.config)
    if args.output:
        if not os.path.exists(args.output):
            raise IOError("Could not find output dir '%s'." % args.output)
        inject.add_injectable("output_dir", args.output)
    if args.data:
        if not os.path.exists(args.data):
            raise IOError("Could not find data dir '%s'." % args.data)
        inject.add_injectable("data_dir", args.data)
    if args.resume:
        inject.add_injectable("resume_after", args.resume)
    if args.models:
        inject.add_injectable("run_list_name", args.models)

    return args


def setting(key, default=None):

    settings = inject.get_injectable('settings')

    # explicit setting in settings file takes precedence
    s = settings.get(key, None)

    # if no setting, try injectable
    if s is None:
        s = inject.get_injectable(key, None)

    # otherwise fall back to supplied default
    if s is None:
        s = default

    return s


def read_model_settings(file_name, mandatory=False):

    configs_dir = inject.get_injectable('configs_dir')

    settings = None
    file_path = os.path.join(configs_dir,  file_name)
    if os.path.isfile(file_path):
        with open(file_path) as f:
            settings = yaml.load(f)

    if settings is None:
        settings = {}

    if mandatory and not settings:
        raise RuntimeError("Could not read settings from %s" % file_name)

    return settings


def get_model_constants(model_settings):
    """
    Read constants from model settings file

    Returns
    -------
    constants : dict
        dictionary of constants to add to locals for use by expressions in model spec
    """
    return model_settings.get('CONSTANTS', {})


def get_logit_model_settings(model_settings):
    """
    Read nest spec (for nested logit) from model settings file

    Returns
    -------
    nests : dict
        dictionary specifying nesting structure and nesting coefficients

    constants : dict
        dictionary of constants to add to locals for use by expressions in model spec
    """
    nests = None

    if model_settings is not None:

        # default to MNL
        logit_type = model_settings.get('LOGIT_TYPE', 'MNL')

        if logit_type not in ['NL', 'MNL']:
            logging.error("Unrecognized logit type '%s'" % logit_type)
            raise RuntimeError("Unrecognized logit type '%s'" % logit_type)

        if logit_type == 'NL':
            nests = model_settings.get('NESTS', None)
            if nests is None:
                logger.error("No NEST found in model spec for NL model type")
                raise RuntimeError("No NEST found in model spec for NL model type")

    return nests


def build_output_file_path(file_name, use_prefix=None):
    output_dir = inject.get_injectable('output_dir')

    if use_prefix:
        file_name = "%s-%s" % (use_prefix, file_name)

    file_path = os.path.join(output_dir, file_name)

    return file_path


def output_file_path(file_name):

    prefix = inject.get_injectable('output_file_prefix', None)
    return build_output_file_path(file_name, use_prefix=prefix)


def trace_file_path(file_name):

    output_dir = inject.get_injectable('output_dir')
    file_name = "trace.%s" % (file_name, )
    file_path = os.path.join(output_dir, file_name)
    return file_path


def log_file_path(file_name):

    prefix = inject.get_injectable('log_file_prefix', None)
    return build_output_file_path(file_name, use_prefix=prefix)


def pipeline_file_path(file_name):

    prefix = inject.get_injectable('pipeline_file_prefix', None)
    return build_output_file_path(file_name, use_prefix=prefix)
