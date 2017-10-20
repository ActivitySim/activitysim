# ActivitySim
# See full license in LICENSE.txt.

import os
import warnings
import logging

import numpy as np
import pandas as pd
import yaml

from activitysim.core import pipeline
from activitysim.core import inject

# FIXME - really?
warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)
pd.options.mode.chained_assignment = None

logger = logging.getLogger(__name__)


@inject.injectable(cache=True)
def store(data_dir, settings):
    if 'store' not in settings:
        logger.error("store file name not specified in settings")
        raise RuntimeError("store file name not specified in settings")
    fname = os.path.join(data_dir, settings["store"])
    if not os.path.exists(fname):
        logger.error("store file not found: %s" % fname)
        raise RuntimeError("store file not found: %s" % fname)

    file = pd.HDFStore(fname, mode='r')
    pipeline.close_on_exit(file, fname)

    return file


@inject.injectable(cache=True)
def cache_skim_key_values(settings):
    return settings['skim_time_periods']['labels']


@inject.injectable(cache=True)
def households_sample_size(settings):
    return settings.get('households_sample_size', 0)


@inject.injectable(cache=True)
def chunk_size(settings):
    return int(settings.get('chunk_size', 0))


@inject.injectable(cache=True)
def check_for_variability(settings):
    return bool(settings.get('check_for_variability', False))


@inject.injectable(cache=True)
def trace_hh_id(settings):

    id = settings.get('trace_hh_id', None)

    if id and not isinstance(id, int):
        logger.warn("setting trace_hh_id is wrong type, should be an int, but was %s" % type(id))
        id = None

    return id


@inject.injectable()
def trace_person_ids():
    # overridden by register_persons if trace_hh_id is defined
    return []


@inject.injectable()
def trace_tour_ids():
    # overridden by register_tours if trace_hh_id is defined
    return []


@inject.injectable(cache=True)
def hh_index_name(settings):
    # overridden by register_households if trace_hh_id is defined
    return None


@inject.injectable(cache=True)
def persons_index_name(settings):
    # overridden by register_persons if trace_hh_id is defined
    return None


@inject.injectable(cache=True)
def trace_od(settings):

    od = settings.get('trace_od', None)

    if od and not (isinstance(od, list) and len(od) == 2 and all(isinstance(x, int) for x in od)):
        logger.warn("setting trace_od is wrong type, should be a list of length 2, but was %s" % od)
        od = None

    return od


@inject.injectable(cache=True)
def enable_trace_log(trace_hh_id, trace_od):
    return (trace_hh_id or trace_od)
