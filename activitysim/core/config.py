# ActivitySim
# See full license in LICENSE.txt.

import argparse
import os
import yaml

import logging
from activitysim.core import inject

logger = logging.getLogger(__name__)

"""
    default injectables
"""


@inject.injectable()
def configs_dir():
    if not os.path.exists('configs'):
        raise RuntimeError("configs_dir: directory does not exist")
    return 'configs'


@inject.injectable()
def data_dir():
    if not os.path.exists('data'):
        raise RuntimeError("data_dir: directory does not exist")
    return 'data'


@inject.injectable()
def output_dir():
    if not os.path.exists('output'):
        raise RuntimeError("output_dir: directory does not exist")
    return 'output'


@inject.injectable()
def output_file_prefix():
    return ''


@inject.injectable(cache=True)
def settings():
    return read_settings_file('settings.yaml', mandatory=True)


@inject.injectable(cache=True)
def pipeline_file_name(settings):
    """
    Orca injectable to return the path to the pipeline hdf5 file based on output_dir and settings
    """
    pipeline_file_name = settings.get('pipeline', 'pipeline.h5')

    return pipeline_file_name


@inject.injectable()
def rng_base_seed():
    return 0


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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

    parser.add_argument("-c", "--config", help="path to config dir", action='append')
    parser.add_argument("-o", "--output", help="path to output dir")
    parser.add_argument("-d", "--data", help="path to data dir")
    parser.add_argument("-r", "--resume", nargs='?', const='_', type=str, help="resume after")
    parser.add_argument("-m", "--multiprocess", type=str2bool, nargs='?', const=True,
                        help="run multiprocess (boolean flag, no arg defaults to true)")

    args = parser.parse_args()

    if args.config:
        for dir in args.config:
            if not os.path.exists(dir):
                raise IOError("Could not find configs dir '%s'" % dir)
        inject.add_injectable("configs_dir", args.config)
    if args.output:
        if not os.path.exists(args.output):
            raise IOError("Could not find output dir '%s'." % args.output)
        inject.add_injectable("output_dir", args.output)
    if args.data:
        if not os.path.exists(args.data):
            raise IOError("Could not find data dir '%s'" % args.data)
        inject.add_injectable("data_dir", args.data)
    if args.resume:
        inject.add_injectable("resume_after", args.resume)
    if args.multiprocess:
        inject.add_injectable("multiprocess", args.multiprocess)

    return args


def setting(key, default=None):

    settings = inject.get_injectable('settings')

    # explicit setting in settings file takes precedence
    s = settings.get(key, None)

    # if no setting, try injectable
    if s is None:
        s = inject.get_injectable(key, None)

        if s:
            # this happens when handle_standard_args overrides a setting with an injectable
            logger.info("read setting %s from injectable" % key)

    # otherwise fall back to supplied default
    if s is None:
        s = default

    return s


def read_model_settings(file_name, mandatory=False):
    """

    Parameters
    ----------
    file_name : str
        yaml file name
    mandatory : bool
        throw error if file empty or not found
    Returns
    -------

    """

    if not file_name.lower().endswith('.yaml'):
        file_name = '%s.yaml' % (file_name, )

    model_settings = read_settings_file(file_name, mandatory=mandatory)

    return model_settings


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


def config_file_path(file_name, mandatory=True):

    configs_dir = inject.get_injectable('configs_dir')

    if isinstance(configs_dir, str):
        configs_dir = [configs_dir]

    assert isinstance(configs_dir, list)

    file_path = None
    for dir in configs_dir:
        p = os.path.join(dir, file_name)
        if os.path.exists(p):
            file_path = p
            break

    if mandatory and not file_path:
        raise RuntimeError("config_file_path: file '%s' not in %s" % (file_path, configs_dir))

    return file_path


def read_settings_file(file_name, mandatory=True):

    def backfill_settings(settings, backfill):
        new_settings = backfill.copy()
        new_settings.update(settings)
        return new_settings

    configs_dir = inject.get_injectable('configs_dir')

    if isinstance(configs_dir, str):
        configs_dir = [configs_dir]
    assert isinstance(configs_dir, list)

    settings = {}
    for dir in configs_dir:
        file_path = os.path.join(dir, file_name)
        if os.path.exists(file_path):
            if settings:
                logger.debug("inherit settings for %s from %s" % (file_name, file_path))

            with open(file_path) as f:
                s = yaml.load(f)
            settings = backfill_settings(settings, s)

            if s.get('inherit_settings', False):
                logger.debug("inherit_settings flag set for %s in %s" % (file_name, file_path))
                continue
            else:
                break

    if mandatory and not settings:
        raise RuntimeError("read_settings_file: no settings for '%s' in %s" %
                           (file_name, configs_dir))

    return settings
