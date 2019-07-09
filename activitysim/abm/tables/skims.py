# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402
from builtins import range
from builtins import int

from future.utils import iteritems

import sys
import os
import logging
import multiprocessing

from collections import OrderedDict

import numpy as np
import openmatrix as omx

from activitysim.core import skim
from activitysim.core import inject
from activitysim.core import util
from activitysim.core import config

logger = logging.getLogger(__name__)

"""
Read in the omx files and create the skim objects
"""


def get_skim_info(omx_file_path, tags_to_load=None):

    # this is sys.maxint for p2.7 but no limit for p3
    # windows sys.maxint =  2147483647
    MAX_BLOCK_BYTES = sys.maxint - 1 if sys.version_info < (3,) else sys.maxsize - 1

    # Note: we load all skims except those with key2 not in tags_to_load
    # Note: we require all skims to be of same dtype so they can share buffer - is that ok?
    # fixme is it ok to require skims be all the same type? if so, is this the right choice?
    skim_dtype = np.float32
    omx_name = os.path.splitext(os.path.basename(omx_file_path))[0]

    with omx.open_file(omx_file_path) as omx_file:
        # omx_shape = tuple(map(int, tuple(omx_file.shape())))  # sometimes omx shape are floats!

        # fixme call to omx_file.shape() failing in windows p3.5
        omx_shape = omx_file.shape()
        omx_shape = (int(omx_shape[0]), int(omx_shape[1]))  # sometimes omx shape are floats!

        omx_skim_names = omx_file.listMatrices()

    # - omx_keys dict maps skim key to omx_key
    # DISTWALK: DISTWALK
    # ('DRV_COM_WLK_BOARDS', 'AM'): DRV_COM_WLK_BOARDS__AM, ...
    omx_keys = OrderedDict()
    for skim_name in omx_skim_names:
        key1, sep, key2 = skim_name.partition('__')

        # - ignore composite tags not in tags_to_load
        if tags_to_load and sep and key2 not in tags_to_load:
            continue

        skim_key = (key1, key2) if sep else key1
        omx_keys[skim_key] = skim_name

    num_skims = len(omx_keys)
    skim_data_shape = omx_shape + (num_skims, )

    # - key1_subkeys dict maps key1 to dict of subkeys with that key1
    # DIST: {'DIST': 0}
    # DRV_COM_WLK_BOARDS: {'MD': 1, 'AM': 0, 'PM': 2}, ...
    key1_subkeys = OrderedDict()
    for skim_key, omx_key in iteritems(omx_keys):
        if isinstance(skim_key, tuple):
            key1, key2 = skim_key
        else:
            key1 = key2 = skim_key
        key2_dict = key1_subkeys.setdefault(key1, {})
        key2_dict[key2] = len(key2_dict)

    # - blocks dict maps block name to blocksize (number of subkey skims in block)
    # skims_0: 198,
    # skims_1: 198, ...
    # - key1_block_offsets dict maps key1 to (block, offset) of first skim with that key1
    # DISTWALK: (0, 2),
    # DRV_COM_WLK_BOARDS: (0, 3), ...

    if MAX_BLOCK_BYTES:
        max_block_items = MAX_BLOCK_BYTES // np.dtype(skim_dtype).itemsize
        max_skims_per_block = max_block_items // np.prod(omx_shape)
    else:
        max_skims_per_block = num_skims

    def block_name(block):
        return "skim_%s_%s" % (omx_name, block)

    key1_block_offsets = OrderedDict()
    blocks = OrderedDict()
    block = offset = 0
    for key1, v in iteritems(key1_subkeys):
        num_subkeys = len(v)
        if offset + num_subkeys > max_skims_per_block:  # next block
            blocks[block_name(block)] = offset
            block += 1
            offset = 0
        key1_block_offsets[key1] = (block, offset)
        offset += num_subkeys
    blocks[block_name(block)] = offset  # last block

    # - block_offsets dict maps skim_key to (block, offset) of omx matrix
    # DIST: (0, 0),
    # ('DRV_COM_WLK_BOARDS', 'AM'): (0, 3),
    # ('DRV_COM_WLK_BOARDS', 'MD') (0, 4), ...
    block_offsets = OrderedDict()
    for skim_key in omx_keys:

        if isinstance(skim_key, tuple):
            key1, key2 = skim_key
        else:
            key1 = key2 = skim_key

        block, key1_offset = key1_block_offsets[key1]

        key2_relative_offset = key1_subkeys.get(key1).get(key2)

        block_offsets[skim_key] = (block, key1_offset + key2_relative_offset)

    logger.debug("get_skim_info from %s" % (omx_file_path, ))
    logger.debug("get_skim_info skim_dtype %s omx_shape %s num_skims %s num_blocks %s" %
                 (skim_dtype, omx_shape, num_skims, len(blocks)))

    skim_info = {
        'omx_name': omx_name,
        'omx_shape': omx_shape,
        'num_skims': num_skims,
        'dtype': skim_dtype,
        'omx_keys': omx_keys,
        'key1_block_offsets': key1_block_offsets,
        'block_offsets': block_offsets,
        'blocks': blocks,
    }

    return skim_info


def buffers_for_skims(skim_info, shared=False):

    skim_dtype = skim_info['dtype']
    omx_shape = [np.float64(x) for x in skim_info['omx_shape']]
    blocks = skim_info['blocks']

    skim_buffers = {}
    for block_name, block_size in iteritems(blocks):

        # buffer_size must be int (or p2.7 long), not np.int64
        buffer_size = int(np.prod(omx_shape) * block_size)

        csz = buffer_size * np.dtype(skim_dtype).itemsize
        logger.info("allocating shared buffer %s for %s (%s) matrices (%s)" %
                    (block_name, buffer_size, omx_shape, util.GB(csz)))

        if shared:
            if np.issubdtype(skim_dtype, np.float64):
                typecode = 'd'
            elif np.issubdtype(skim_dtype, np.float32):
                typecode = 'f'
            else:
                raise RuntimeError("buffers_for_skims unrecognized dtype %s" % skim_dtype)

            buffer = multiprocessing.RawArray(typecode, buffer_size)
        else:
            buffer = np.zeros(buffer_size, dtype=skim_dtype)

        skim_buffers[block_name] = buffer

    return skim_buffers


def skim_data_from_buffers(skim_buffers, skim_info):

    assert type(skim_buffers) == dict

    omx_shape = skim_info['omx_shape']
    skim_dtype = skim_info['dtype']
    blocks = skim_info['blocks']

    skim_data = []
    for block_name, block_size in iteritems(blocks):
        skims_shape = omx_shape + (block_size,)
        block_buffer = skim_buffers[block_name]
        assert len(block_buffer) == int(np.prod(skims_shape))
        block_data = np.frombuffer(block_buffer, dtype=skim_dtype).reshape(skims_shape)
        skim_data.append(block_data)

    return skim_data


def load_skims(omx_file_path, skim_info, skim_buffers):

    skim_data = skim_data_from_buffers(skim_buffers, skim_info)

    block_offsets = skim_info['block_offsets']
    omx_keys = skim_info['omx_keys']

    # read skims into skim_data
    with omx.open_file(omx_file_path) as omx_file:
        for skim_key, omx_key in iteritems(omx_keys):

            omx_data = omx_file[omx_key]
            assert np.issubdtype(omx_data.dtype, np.floating)

            block, offset = block_offsets[skim_key]
            block_data = skim_data[block]

            logger.debug("load_skims load omx_key %s skim_key %s to block %s offset %s" %
                         (omx_key, skim_key, block, offset))

            # this will trigger omx readslice to read and copy data to skim_data's buffer
            a = block_data[:, :, offset]
            a[:] = omx_data[:]

    logger.info("load_skims loaded skims from %s" % (omx_file_path, ))


@inject.injectable(cache=True)
def skim_dict(data_dir, settings):

    omx_file_path = config.data_file_path(settings["skims_file"])
    tags_to_load = settings['skim_time_periods']['labels']

    logger.info("loading skim_dict from %s" % (omx_file_path, ))

    # select the skims to load
    skim_info = get_skim_info(omx_file_path, tags_to_load)

    logger.debug("omx_shape %s skim_dtype %s" % (skim_info['omx_shape'], skim_info['dtype']))

    skim_buffers = inject.get_injectable('data_buffers', None)
    if skim_buffers:
        logger.info('Using existing skim_buffers for skims')
    else:
        skim_buffers = buffers_for_skims(skim_info, shared=False)
        load_skims(omx_file_path, skim_info, skim_buffers)

    skim_data = skim_data_from_buffers(skim_buffers, skim_info)

    block_names = list(skim_info['blocks'].keys())
    for i in range(len(skim_data)):
        block_name = block_names[i]
        block_data = skim_data[i]
        logger.info("block_name %s bytes %s (%s)" %
                    (block_name, block_data.nbytes, util.GB(block_data.nbytes)))

    # create skim dict
    skim_dict = skim.SkimDict(skim_data, skim_info)
    skim_dict.offset_mapper.set_offset_int(-1)

    return skim_dict


@inject.injectable(cache=True)
def skim_stack(skim_dict):

    logger.debug("loading skim_stack injectable")
    return skim.SkimStack(skim_dict)
