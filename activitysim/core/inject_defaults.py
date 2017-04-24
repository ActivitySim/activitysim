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
        raise extensions_dir("output_dir: directory does not exist")
    return 'extensions'


@orca.injectable()
def settings(configs_dir):
    with open(os.path.join(configs_dir, 'settings.yaml')) as f:
        return yaml.load(f)


@orca.injectable(cache=True)
def store(data_dir, settings):
    if 'store' not in settings:
        logger.error("store file name not specified in settings")
        raise RuntimeError("store file name not specified in settings")
    fname = os.path.join(data_dir, settings["store"])
    if not os.path.exists(fname):
        logger.error("store file not found: %s" % fname)
        raise RuntimeError("store file not found: %s" % fname)
    return pd.HDFStore(fname, mode='r')
