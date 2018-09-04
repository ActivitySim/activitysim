# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import pandas as pd
import yaml

from activitysim.core import inject

logger = logging.getLogger(__name__)


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
def settings(configs_dir):
    with open(os.path.join(configs_dir, 'settings.yaml')) as f:
        return yaml.load(f)


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
