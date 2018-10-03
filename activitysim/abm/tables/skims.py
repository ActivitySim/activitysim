# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

from collections import OrderedDict
import multiprocessing as mp

import numpy as np

import openmatrix as omx

from activitysim.core import skim
from activitysim.core import inject
from activitysim.core import util

logger = logging.getLogger(__name__)

"""
Read in the omx files and create the skim objects
"""


def skims_to_load(omx_file_path, tags_to_load=None):

    # select the skims to load
    with omx.open_file(omx_file_path) as omx_file:

        omx_shape = tuple(map(int, omx_file.shape()))  # sometimes omx shape are floats!
        skim_keys = OrderedDict()

        for skim_name in omx_file.listMatrices():
            key1, sep, key2 = skim_name.partition('__')

            # ignore composite tags not in tags_to_load
            if tags_to_load and sep and key2 not in tags_to_load:
                continue

            key = (key1, key2) if sep else key1
            skim_keys[skim_name] = key

    num_skims = len(skim_keys.keys())

    skim_data_shape = omx_shape + (num_skims, )
    skim_dtype = np.float32

    logger.debug("skims_to_load from %s" % (omx_file_path, ))
    logger.debug("skims_to_load skim_data_shape %s skim_dtype %s" % (skim_data_shape, skim_dtype))

    return skim_keys, skim_data_shape, skim_dtype


def shared_buffer_for_skims(skims_shape, skim_dtype, shared=False):

    buffer_size = int(np.prod(skims_shape))

    if np.issubdtype(skim_dtype, np.float64):
        typecode = 'd'
    elif np.issubdtype(skim_dtype, np.float32):
        typecode = 'f'
    else:
        raise RuntimeError("shared_buffer_for_skims unrecognized dtype %s" % skim_dtype)

    logger.info("allocating shared buffer of size %s (%s)" % (buffer_size, skims_shape, ))

    skim_buffer = mp.RawArray(typecode, buffer_size)

    return skim_buffer


def load_skims(omx_file_path, skim_keys, skim_data):

    # read skims into skim_data
    with omx.open_file(omx_file_path) as omx_file:
        n = 0
        for skim_name, key in skim_keys.iteritems():

            logger.debug("load_skims skim_name %s key %s" % (skim_name, key))

            omx_data = omx_file[skim_name]
            assert np.issubdtype(omx_data.dtype, np.floating)

            # this will trigger omx readslice to read and copy data to skim_data's buffer
            a = skim_data[:, :, n]
            a[:] = omx_data[:]
            n += 1

    logger.info("load_skims loaded %s skims from %s" % (n, omx_file_path))


@inject.injectable(cache=True)
def skim_dict(data_dir, settings):

    omx_file_path = os.path.join(data_dir, settings["skims_file"])
    tags_to_load = settings['skim_time_periods']['labels']

    logger.info("loading skim_dict from %s" % (omx_file_path, ))

    # select the skims to load
    skim_keys, skims_shape, skim_dtype = skims_to_load(omx_file_path, tags_to_load)

    logger.debug("skim_data_shape %s skim_dtype %s" % (skims_shape, skim_dtype))

    skim_buffer = inject.get_injectable('skim_buffer', None)
    if skim_buffer:
        logger.info('Using existing skim_buffer for skims')
        skim_data = np.frombuffer(skim_buffer, dtype=skim_dtype).reshape(skims_shape)
    else:
        skim_data = np.zeros(skims_shape, dtype=skim_dtype)
        load_skims(omx_file_path, skim_keys, skim_data)

    logger.info("skim_data dtype %s shape %s bytes %s (%s)" %
                (skim_dtype, skims_shape, skim_data.nbytes, util.GB(skim_data.nbytes)))

    # create skim dict
    skim_dict = skim.SkimDict(skim_data, skim_keys.values())
    skim_dict.offset_mapper.set_offset_int(-1)

    return skim_dict


@inject.injectable(cache=True)
def skim_stack(skim_dict):

    logger.debug("loading skim_stack injectable")
    return skim.SkimStack(skim_dict)
