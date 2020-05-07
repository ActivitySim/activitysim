# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import numpy as np
import openmatrix as omx

from activitysim.core import skim as askim
from activitysim.core import config
from activitysim.core import inject


logger = logging.getLogger('activitysim')

"""
Read in the omx files and create the skim objects
"""


@inject.injectable(cache=True)
def cache_skim_key_values(settings):
    return settings['skim_time_periods']['labels']


def add_to_skim_dict(skim_dict, omx_file, cache_skim_key_values, offset_int=None):

    if offset_int is None:

        if 'default_mapping' not in omx_file.listMappings():

            raise RuntimeError("Could not find 'default_mapping' in omx file."
                               "\nMaybe rerun import_data.py to rebuild the skims files.")

        offset_map = omx_file.mapentries('default_mapping')
        skim_dict.offset_mapper.set_offset_list(offset_map)
    else:
        skim_dict.offset_mapper.set_offset_int(offset_int)

    skims_in_omx = omx_file.listMatrices()
    for skim_name in skims_in_omx:
        key, sep, key2 = skim_name.partition('__')
        skim_data = omx_file[skim_name]

        if not sep:
            # no separator - this is a simple 2d skim - we load them all
            skim_dict.set(key, skim_data)
        else:
            # there may be more time periods in the skim than are used by the model
            # cache_skim_key_values is a list of time periods (from settings) that are used
            # FIXME - assumes that the only types of key2 are skim_time_periods
            if key2 in cache_skim_key_values:
                skim_dict.set((key, key2), skim_data)


@inject.injectable(cache=True)
def taz_skim_dict(data_dir, settings):

    logger.info("loading taz_skim_dict")

    skims_file = config.data_file_path(settings["taz_skims_file"])
    cache_skim_key_values = settings['skim_time_periods']['labels']

    skim_dict = askim.SkimDict()

    with omx.open_file(skims_file) as omx_file:
        add_to_skim_dict(skim_dict, omx_file, cache_skim_key_values)

    return skim_dict


@inject.injectable(cache=True)
def tap_skim_dict(data_dir, settings):

    logger.info("loading tap_skim_dict")

    cache_skim_key_values = settings['skim_time_periods']['labels']
    skim_dict = askim.SkimDict()

    for skims_file in settings["tap_skims_files"]:
        skims_file_path = config.data_file_path(skims_file)
        with omx.open_file(skims_file_path) as omx_file:
            add_to_skim_dict(skim_dict, omx_file, cache_skim_key_values)

    return skim_dict
