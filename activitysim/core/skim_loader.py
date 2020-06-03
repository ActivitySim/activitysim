# ActivitySim
# See full license in LICENSE.txt.
from builtins import range
from builtins import int

import sys
import os
import logging
import multiprocessing

from collections import OrderedDict
from functools import reduce
from operator import mul

import numpy as np
import openmatrix as omx

from activitysim.core import skim
from activitysim.core import skim_loader
from activitysim.core import inject
from activitysim.core import util
from activitysim.core import config

logger = logging.getLogger(__name__)


def get_skim_info(omx_tag):
    omx_files = config.setting(omx_tag)
    omx_files = [omx_files] if isinstance(omx_files, str) else omx_files  # want a list,not a single file name

    tags_to_load = config.setting('skim_time_periods')['labels']

    # Note: we load all skims except those with key2 not in tags_to_load
    # Note: we require all skims to be of same dtype so they can share buffer - is that ok?
    # fixme is it ok to require skims be all the same type? if so, is this the right choice?
    skim_dtype = np.float32

    omx_shape = offset_map = offset_map_name = None
    omx_manifest = {}  # dict mapping { omx_key: skim_name }

    for omx_file_name in omx_files:

        omx_file_path = config.data_file_path(omx_file_name)

        logger.debug("get_skim_info reading %s" % (omx_file_path,))

        with omx.open_file(omx_file_path) as omx_file:
            # omx_shape = tuple(map(int, tuple(omx_file.shape())))  # sometimes omx shape are floats!

            # fixme call to omx_file.shape() failing in windows p3.5
            if omx_shape is None:
                omx_shape = tuple(int(i) for i in omx_file.shape())  # sometimes omx shape are floats!
            else:
                assert (omx_shape == tuple(int(i) for i in omx_file.shape()))

            for skim_name in omx_file.listMatrices():
                assert skim_name not in omx_manifest, \
                    f"duplicate skim '{skim_name}' found in {omx_manifest[skim_name]} and {omx_file}"
                omx_manifest[skim_name] = omx_file_name

            for m in omx_file.listMappings():
                if offset_map is None:
                    offset_map_name = m
                    offset_map = omx_file.mapentries(offset_map_name)
                    assert len(offset_map) == omx_shape[0]

                    logger.debug(f"get_skim_info omx_tag {omx_tag} using offset_map {m}")
                else:
                    # don't really expect more than one, but ok if they are all the same
                    if not (offset_map == omx_file.mapentries(m)):
                        raise RuntimeError(f"Multiple different mappings in omx file: {offset_map_name} != {m}")

    # - omx_keys dict maps skim key to omx_key
    # DISTWALK: DISTWALK
    # ('DRV_COM_WLK_BOARDS', 'AM'): DRV_COM_WLK_BOARDS__AM, ...
    omx_keys = OrderedDict()
    for skim_name in omx_manifest.keys():
        key1, sep, key2 = skim_name.partition('__')

        # - ignore composite tags not in tags_to_load
        if tags_to_load and sep and key2 not in tags_to_load:
            continue

        skim_key = (key1, key2) if sep else key1
        omx_keys[skim_key] = skim_name

    num_skims = len(omx_keys)

    # - key1_subkeys dict maps key1 to dict of subkeys with that key1
    # DIST: {'DIST': 0}
    # DRV_COM_WLK_BOARDS: {'MD': 1, 'AM': 0, 'PM': 2}, ...
    key1_subkeys = OrderedDict()
    for skim_key, omx_key in omx_keys.items():
        if isinstance(skim_key, tuple):
            key1, key2 = skim_key
        else:
            key1 = key2 = skim_key
        key2_dict = key1_subkeys.setdefault(key1, {})
        key2_dict[key2] = len(key2_dict)

    key1_block_offsets = OrderedDict()
    offset = 0
    for key1, v in key1_subkeys.items():
        num_subkeys = len(v)
        key1_block_offsets[key1] = offset
        offset += num_subkeys

    # - block_offsets dict maps skim_key to offset of omx matrix
    # DIST: 0,
    # ('DRV_COM_WLK_BOARDS', 'AM'): 3,
    # ('DRV_COM_WLK_BOARDS', 'MD') 4, ...
    block_offsets = OrderedDict()
    for skim_key in omx_keys:

        if isinstance(skim_key, tuple):
            key1, key2 = skim_key
        else:
            key1 = key2 = skim_key

        key1_offset = key1_block_offsets[key1]
        key2_relative_offset = key1_subkeys.get(key1).get(key2)
        block_offsets[skim_key] = key1_offset + key2_relative_offset

    logger.debug("get_skim_info skim_dtype %s omx_shape %s num_skims %s" %
                 (skim_dtype, omx_shape, num_skims,))

    skim_info = {
        'omx_tag': omx_tag,
        'omx_files': omx_files,
        'omx_manifest': omx_manifest,
        'omx_shape': omx_shape,
        'num_skims': num_skims,
        'dtype': skim_dtype,
        'offset_map_name': offset_map_name,
        'offset_map': offset_map,
        'omx_keys': omx_keys,
        'key1_block_offsets': key1_block_offsets,
        'block_offsets': block_offsets,
        'block_name': omx_tag,
    }

    return skim_info


def multiply_large_numbers(list_of_numbers):
    return reduce(mul, list_of_numbers)


def buffers_for_skims(skim_info, shared=False):
    skim_dtype = skim_info['dtype']
    omx_shape = skim_info['omx_shape']
    num_skims = skim_info['num_skims']
    block_name = skim_info['block_name']

    skim_buffers = {}

    # buffer_size must be int, not np.int64
    buffer_size = int(multiply_large_numbers(omx_shape) * num_skims)

    itemsize = np.dtype(skim_dtype).itemsize
    csz = buffer_size * itemsize
    logger.info("allocating shared buffer %s for %s skims (skim size: %s * %s bytes = %s) total size: %s (%s)" %
                (block_name, num_skims, omx_shape, itemsize, buffer_size, csz, util.GB(csz)))

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
    num_skims = skim_info['num_skims']
    block_name = skim_info['block_name']

    skims_shape = omx_shape + (num_skims,)
    block_buffer = skim_buffers[block_name]
    assert len(block_buffer) == int(multiply_large_numbers(skims_shape))
    skim_data = np.frombuffer(block_buffer, dtype=skim_dtype).reshape(skims_shape)

    return skim_data


def default_skim_cache_dir():
    return inject.get_injectable('output_dir')


def build_skim_cache_file_name(omx_tag):
    return f"cached_{omx_tag}.mmap"


def read_skim_cache(skim_info, skim_data):
    """
        read cached memmapped skim data from canonically named cache file(s) in output directory into skim_data
    """

    skim_cache_dir = config.setting('skim_cache_dir', default_skim_cache_dir())
    logger.info(f"load_skims reading skims data from cache directory {skim_cache_dir}")

    omx_tag = skim_info['omx_tag']
    dtype = np.dtype(skim_info['dtype'])
    block_name = skim_info['block_name']

    skim_cache_file_name = build_skim_cache_file_name(omx_tag)
    skim_cache_path = os.path.join(skim_cache_dir, skim_cache_file_name)

    assert os.path.isfile(skim_cache_path), \
        "read_skim_cache could not find skim_cache_path: %s" % (skim_cache_path,)

    logger.info(f"load_skims reading block_name {block_name} {skim_data.shape} from {skim_cache_file_name}")

    data = np.memmap(skim_cache_path, shape=skim_data.shape, dtype=dtype, mode='r')
    assert data.shape == skim_data.shape

    skim_data[::] = data[::]


def write_skim_cache(skim_info, skim_data):
    """
        write skim data from skim_data to canonically named cache file(s) in output directory
    """

    skim_cache_dir = config.setting('skim_cache_dir', default_skim_cache_dir())
    logger.info(f"load_skims writing skims data to cache directory {skim_cache_dir}")

    omx_tag = skim_info['omx_tag']
    dtype = np.dtype(skim_info['dtype'])
    block_name = skim_info['block_name']

    skim_cache_file_name = build_skim_cache_file_name(omx_tag)
    skim_cache_path = os.path.join(skim_cache_dir, skim_cache_file_name)

    logger.info(f"load_skims writing block_name {block_name} {skim_data.shape} to {skim_cache_file_name}")

    data = np.memmap(skim_cache_path, shape=skim_data.shape, dtype=dtype, mode='w+')
    data[::] = skim_data


def read_skims_from_omx(skim_info, skim_data):
    """
    read skims from omx file into skim_data
    """

    block_offsets = skim_info['block_offsets']
    omx_keys = skim_info['omx_keys']
    omx_files = skim_info['omx_files']
    omx_manifest = skim_info['omx_manifest']

    for omx_file_name in omx_files:

        omx_file_path = config.data_file_path(omx_file_name)
        num_skims_loaded = 0

        logger.info(f"read_skims_from_omx {omx_file_path}")

        # read skims into skim_data
        with omx.open_file(omx_file_path) as omx_file:
            for skim_key, omx_key in omx_keys.items():

                if omx_manifest[omx_key] == omx_file_name:
                    omx_data = omx_file[omx_key]
                    assert np.issubdtype(omx_data.dtype, np.floating)

                    offset = block_offsets[skim_key]

                    logger.debug("read_skims_from_omx load omx_key %s skim_key %s to offset %s" %
                                 (omx_key, skim_key, offset))

                    # this will trigger omx readslice to read and copy data to skim_data's buffer
                    a = skim_data[:, :, offset]
                    a[:] = omx_data[:]

                    num_skims_loaded += 1

        logger.info(f"read_skims_from_omx loaded {num_skims_loaded} skims from {omx_file_name}")


def load_skims(skim_info, skim_buffers):

    read_cache = config.setting('read_skim_cache')
    write_cache = config.setting('write_skim_cache')
    assert not (read_cache and write_cache), \
        "read_skim_cache and write_skim_cache are both True in settings file. I am assuming this is a mistake"

    skim_data = skim_data_from_buffers(skim_buffers, skim_info)

    if read_cache:
        read_skim_cache(skim_info, skim_data)
    else:
        read_skims_from_omx(skim_info, skim_data)

    if write_cache:
        write_skim_cache(skim_info, skim_data)


def create_skim_dict(omx_tag):
    logger.info(f"create_skim_dict loading skim dict {omx_tag}")

    # select the skims to load
    skim_info = get_skim_info(omx_tag)
    block_name = skim_info['block_name']

    logger.debug(f"create_skim_dict {omx_tag} omx_shape {skim_info['omx_shape']} skim_dtype {skim_info['dtype']}")

    skim_buffers = inject.get_injectable('data_buffers', None)
    if skim_buffers:
        logger.info('create_skim_dict {omx_tag} using existing skim_buffers for skims')
    else:
        skim_buffers = buffers_for_skims(skim_info, shared=False)
        load_skims(skim_info, skim_buffers)

    skim_data = skim_data_from_buffers(skim_buffers, skim_info)

    logger.info("create_skim_dict {omx_tag} block_name %s bytes %s (%s)" %
                (block_name, skim_data.nbytes, util.GB(skim_data.nbytes)))

    # create skim dict
    skim_dict = skim.SkimDict(skim_data, skim_info)

    offset_map = skim_info['offset_map']
    if offset_map is not None:
        skim_dict.offset_mapper.set_offset_list(offset_map)
        logger.debug(
            f"create_skim_dict {omx_tag} using offset map {skim_info['offset_map_name']} from omx file: {offset_map}")
    else:
        # assume this is a one-based skim map
        skim_dict.offset_mapper.set_offset_int(-1)

    return skim_dict
