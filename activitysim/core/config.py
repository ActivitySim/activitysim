# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402

import argparse
import os
import yaml
import sys

import logging
from activitysim.core import inject

logger = logging.getLogger(__name__)

"""
    default injectables
"""


@inject.injectable(cache=True)
def locutor():
    # when multiprocessing, sometimes you only want one process to write trace files
    # mp_tasks overrides this definition to designate a single sub-process as locutor
    return True


@inject.injectable(cache=True)
def configs_dir():
    if not os.path.exists('configs'):
        raise RuntimeError("configs_dir: directory does not exist")
    return 'configs'


@inject.injectable(cache=True)
def data_dir():
    if not os.path.exists('data'):
        raise RuntimeError("data_dir: directory does not exist")
    return 'data'


@inject.injectable(cache=True)
def output_dir():
    if not os.path.exists('output'):
        raise RuntimeError("output_dir: directory does not exist")
    return 'output'


@inject.injectable()
def output_file_prefix():
    return ''


@inject.injectable(cache=True)
def pipeline_file_name(settings):

    pipeline_file_name = settings.get('pipeline_file_name', 'pipeline.h5')

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


@inject.injectable(cache=True)
def settings():
    settings_dict = read_settings_file('settings.yaml', mandatory=True)

    return settings_dict


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

    injectables - array of injectables altered by args
    """

    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", help="path to config dir", action='append')
    parser.add_argument("-o", "--output", help="path to output dir")
    parser.add_argument("-d", "--data", help="path to data dir", action='append')
    parser.add_argument("-r", "--resume", nargs='?', const='_', type=str, help="resume after")
    parser.add_argument("-m", "--multiprocess", type=str2bool, nargs='?', const=True,
                        help="run multiprocess (boolean flag, no arg defaults to true)")

    parser.add_argument("-p", "--pipeline", help="pipeline file name")

    args = parser.parse_args()

    injectables = []

    def override_injectable(name, value):
        inject.add_injectable(name, value)
        injectables.append(name)

    def override_setting(key, value):
        new_settings = inject.get_injectable('settings')
        new_settings[key] = value
        inject.add_injectable('settings', new_settings)

    if args.config:
        for dir in args.config:
            if not os.path.exists(dir):
                raise IOError("Could not find configs dir '%s'" % dir)
        override_injectable("configs_dir", args.config)

    if args.data:
        for dir in args.data:
            if not os.path.exists(dir):
                raise IOError("Could not find data dir '%s'" % dir)
        override_injectable("data_dir", args.data)

    if args.output:
        if not os.path.exists(args.output):
            raise IOError("Could not find output dir '%s'." % args.output)
        override_injectable("output_dir", args.output)

    if args.pipeline:
        override_injectable("pipeline_file_name", args.pipeline)

    # - do these after potentially overriding configs_dir
    # FIXME we don't currently pass settings as an injectable to mp_tasks.run_multiprocess
    # these two make it through as they are incorporated into the run_list by parent
    # but if a more extensible capability is desired, settings could be passed in injectables
    if args.resume is not None:
        override_setting('resume_after', args.resume)
    if args.multiprocess is not None:
        override_setting('multiprocess', args.multiprocess)

    return injectables


def setting(key, default=None):

    return inject.get_injectable('settings').get(key, default)


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


def cascading_input_file_path(file_name, dir_list_injectable_name, mandatory=True):

    dir_list = inject.get_injectable(dir_list_injectable_name)

    if isinstance(dir_list, str):
        dir_list = [dir_list]

    assert isinstance(dir_list, list)

    file_path = None
    for dir in dir_list:
        p = os.path.join(dir, file_name)
        if os.path.isfile(p):
            file_path = p
            break

    if mandatory and not file_path:
        raise RuntimeError("file_path %s: file '%s' not in %s" %
                           (dir_list_injectable_name, file_path, dir_list))

    return file_path


def data_file_path(file_name, mandatory=True):

    return cascading_input_file_path(file_name, 'data_dir', mandatory)


def config_file_path(file_name, mandatory=True):

    return cascading_input_file_path(file_name, 'configs_dir', mandatory)


def output_file_path(file_name):

    prefix = inject.get_injectable('output_file_prefix', None)
    return build_output_file_path(file_name, use_prefix=prefix)


def trace_file_path(file_name):

    output_dir = inject.get_injectable('output_dir')

    # - check for optional trace subfolder
    if os.path.exists(os.path.join(output_dir, 'trace')):
        output_dir = os.path.join(output_dir, 'trace')
    else:
        file_name = "trace.%s" % (file_name,)

    file_path = os.path.join(output_dir, file_name)
    return file_path


def log_file_path(file_name):

    output_dir = inject.get_injectable('output_dir')

    # - check for optional log subfolder
    if os.path.exists(os.path.join(output_dir, 'log')):
        output_dir = os.path.join(output_dir, 'log')

    # - check for optional process name prefix
    prefix = inject.get_injectable('log_file_prefix', None)
    if prefix:
        file_name = "%s-%s" % (prefix, file_name)

    file_path = os.path.join(output_dir, file_name)

    return file_path


def open_log_file(file_name, mode):
    mode = mode + 'b' if sys.version_info < (3,) else mode
    return open(log_file_path(file_name), mode)


def pipeline_file_path(file_name):

    prefix = inject.get_injectable('pipeline_file_prefix', None)
    return build_output_file_path(file_name, use_prefix=prefix)


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
                logger.debug("read settings for %s from %s" % (file_name, file_path))

            with open(file_path) as f:
                s = yaml.load(f, Loader=yaml.SafeLoader)
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


def filter_warnings():
    """
    set warning filter to 'strict' if specified in settings
    """

    if setting('strict', False):  # noqa: E402
        import warnings
        warnings.filterwarnings('error', category=Warning)
        warnings.filterwarnings('default', category=PendingDeprecationWarning, module='future')
        warnings.filterwarnings('default', category=FutureWarning, module='pandas')
        warnings.filterwarnings('default', category=RuntimeWarning, module='numpy')
