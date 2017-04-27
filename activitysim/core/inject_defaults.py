# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import orca
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


@orca.injectable()
def configs_dir():
    if not os.path.exists('configs'):
        raise RuntimeError("configs_dir: directory does not exist")
    return 'configs'


@orca.injectable()
def data_dir():
    if not os.path.exists('data'):
        raise RuntimeError("data_dir: directory does not exist")
    return 'data'


@orca.injectable()
def output_dir():
    if not os.path.exists('output'):
        raise RuntimeError("output_dir: directory does not exist")
    return 'output'


@orca.injectable()
def extensions_dir():
    if not os.path.exists('extensions'):
        raise RuntimeError("output_dir: directory does not exist")
    return 'extensions'


@orca.injectable()
def settings(configs_dir):
    with open(os.path.join(configs_dir, 'settings.yaml')) as f:
        return yaml.load(f)


@orca.injectable(cache=True)
def pipeline_path(output_dir, settings):
    """
    Orca injectable to return the path to the pipeline hdf5 file based on output_dir and settings
    """
    pipeline_file_name = settings.get('pipeline', 'pipeline.h5')
    pipeline_file_path = os.path.join(output_dir, pipeline_file_name)
    return pipeline_file_path
