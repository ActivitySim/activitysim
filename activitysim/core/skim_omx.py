# ActivitySim
# See full license in LICENSE.txt.
# from builtins import int

import os
import logging
import numpy as np
import openmatrix as omx
from abc import ABC, abstractmethod

from activitysim.core import config
from activitysim.core import skim

logger = logging.getLogger(__name__)


class AbstractSkimFactory(ABC):

    def __init__(self, network_los):
        self.network_los = network_los

    @property
    def name(self):
        return type(self).__name__

    @property
    def share_data_for_multiprocessing(self):
        """
        Does subclass support shareable data for multiprocessing

        Returns
        -------
        boolean
        """
        return False

    def allocate_skim_buffer(self, skim_info, shared=False):
        assert False, "Not supported"

    def skim_data_from_buffer(self, skim_info, skim_buffer):
        assert False, "Not supported"

    def cached_skim_info_path(self, skim_tag):
        return os.path.join(self.network_los.get_cache_dir(), f"cached_{skim_tag}.yaml")

    def memmap_skim_data_path(self, skim_tag):
        return os.path.join(self.network_los.get_cache_dir(), f"cached_{skim_tag}.mmap")

    def load_skim_info(self, skim_tag):
        """
        Read omx files for skim <skim_tag> (e.g. 'TAZ') and build skim_info dict

        Parameters
        ----------
        skim_tag
        omx_file_names
        skim_time_periods

        Returns
        -------
            skim_info: dict {
                skim_tag:           str                         (e.g. 'TAZ')
                omx_file_names:     list                        e.g. ['file1.omx', 'file2.omx']
                omx_manifest:       dict                        # dict mapping { omx_key: omx_file_name }
                omx_shape:          tuple
                num_skims:          int
                dtype:              type                        (e.g. np.float32)
                offset_map:         NoneType
                omx_keys:           OrderedDict
                base_keys:          list
                block_offsets:      OrderedDict                 # dict mapping skim key tuple to offset

            }
        """

        omx_file_names = self.network_los.omx_file_names(skim_tag)
        skim_time_periods = self.network_los.skim_time_periods

        # if no skim_time_periods are specified
        dim3_tags_to_load = skim_time_periods and skim_time_periods['labels']

        # Note: we load all skims except those with key2 not in dim3_tags_to_load
        # Note: we require all skims to be of same dtype so they can share buffer - is that ok?
        # fixme is it ok to require skims be all the same type? if so, is this the right choice?
        skim_dtype_name = self.network_los.skim_dtype_name

        omx_shape = offset_map = offset_map_name = None
        omx_manifest = {}  # dict mapping { omx_key: skim_name }

        for omx_file_name in omx_file_names:

            omx_file_path = config.data_file_path(omx_file_name)

            logger.debug(f"load_skim_info {skim_tag} reading {omx_file_path}")

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
                    else:
                        # don't really expect more than one, but ok if they are all the same
                        if not (offset_map == omx_file.mapentries(m)):
                            raise RuntimeError(f"Multiple different mappings in omx file: {offset_map_name} != {m}")

        # - omx_keys dict maps skim key to omx_key
        # DISTWALK: DISTWALK
        # ('DRV_COM_WLK_BOARDS', 'AM'): DRV_COM_WLK_BOARDS__AM, ...
        omx_keys = dict()
        for skim_name in omx_manifest.keys():
            key1, sep, key2 = skim_name.partition('__')

            # - ignore composite tags not in dim3_tags_to_load
            if dim3_tags_to_load and sep and key2 not in dim3_tags_to_load:
                continue

            skim_key = (key1, key2) if sep else key1

            omx_keys[skim_key] = skim_name

        num_skims = len(omx_keys)

        # - key1_subkeys dict maps key1 to dict of subkeys with that key1
        # DIST: {'DIST': 0}
        # DRV_COM_WLK_BOARDS: {'MD': 1, 'AM': 0, 'PM': 2}, ...
        key1_subkeys = dict()
        for skim_key, omx_key in omx_keys.items():
            if isinstance(skim_key, tuple):
                key1, key2 = skim_key
            else:
                key1 = key2 = skim_key
            key2_dict = key1_subkeys.setdefault(key1, {})
            key2_dict[key2] = len(key2_dict)

        key1_block_offsets = dict()
        offset = 0
        for key1, v in key1_subkeys.items():
            num_subkeys = len(v)
            key1_block_offsets[key1] = offset
            offset += num_subkeys

        # - block_offsets dict maps skim_key to offset of omx matrix
        # DIST: 0,
        # ('DRV_COM_WLK_BOARDS', 'AM'): 3,
        # ('DRV_COM_WLK_BOARDS', 'MD') 4, ...
        block_offsets = dict()
        for skim_key in omx_keys:

            if isinstance(skim_key, tuple):
                key1, key2 = skim_key
            else:
                key1 = key2 = skim_key

            key1_offset = key1_block_offsets[key1]
            key2_relative_offset = key1_subkeys.get(key1).get(key2)
            block_offsets[skim_key] = key1_offset + key2_relative_offset

        if skim.ROW_MAJOR_LAYOUT:
            skim_data_shape = (num_skims,) + omx_shape
        else:
            skim_data_shape = omx_shape + (num_skims,)

        skim_info = {
            'skim_tag': skim_tag,
            'omx_file_names': omx_file_names,  # list of omx_file_names
            'omx_manifest': omx_manifest,  # dict mapping { omx_key: omx_file_name }
            'omx_shape': omx_shape,
            'num_skims': num_skims,
            'skim_data_shape': skim_data_shape,
            'dtype_name': skim_dtype_name,
            'offset_map': offset_map,  # array from omx_file.mapentries or None
            'offset_map_name': offset_map_name,
            'omx_keys': omx_keys,  # dict mapping skim key tuple to omx_key
            'base_keys': list(key1_block_offsets.keys()),  # list of base (key1) keys
            'block_offsets': block_offsets,  # dict mapping skim key tuple to offset
        }

        #bug
        for key in ['omx_shape', 'num_skims', 'skim_data_shape', 'dtype_name', 'offset_map_name']:
            logger.debug(f"load_skim_info '{skim_tag}' {key}: {skim_info[key]}")

        # for key in skim_info.keys():
        #    print(f"{key} {type(skim_info[key])}\n{skim_info[key]}\n")

        return skim_info

    def read_skims_from_omx(self, skim_info, skim_data):
        """
        read skims from omx file into skim_data
        """

        block_offsets = skim_info['block_offsets']
        omx_keys = skim_info['omx_keys']
        omx_file_names = skim_info['omx_file_names']
        omx_manifest = skim_info['omx_manifest']  # dict mapping { omx_key: skim_name }

        for omx_file_name in omx_file_names:

            omx_file_path = config.data_file_path(omx_file_name)
            num_skims_loaded = 0

            logger.info(f"read_skims_from_omx {omx_file_path}")

            # read skims into skim_data
            with omx.open_file(omx_file_path) as omx_file:
                for skim_key, omx_key in omx_keys.items():

                    if omx_manifest[omx_key] == omx_file_name:

                        offset = block_offsets[skim_key]
                        logger.debug(f"read_skims_from_omx file {omx_file_name} omx_key {omx_key} "
                                     f"skim_key {skim_key} to offset {offset}")

                        if skim.ROW_MAJOR_LAYOUT:
                            a = skim_data[offset, :, :]  # LAYOUT
                        else:
                            a = skim_data[:, :, offset]  #LAYOUT

                        # this will trigger omx readslice to read and copy data to skim_data's buffer
                        omx_data = omx_file[omx_key]
                        a[:] = omx_data[:]

                        num_skims_loaded += 1

            logger.info(f"read_skims_from_omx loaded {num_skims_loaded} skims from {omx_file_name}")

    def open_existing_readonly_memmap_skim_cache(self, skim_info):
        """
            read cached memmapped skim data from canonically named cache file(s) in output directory into skim_data
            return True if it was there and we read it, return False if not found
        """

        skim_tag = skim_info['skim_tag']
        dtype = np.dtype(skim_info['dtype_name'])
        skim_data_shape = skim_info['skim_data_shape']

        skim_cache_path = self.memmap_skim_data_path(skim_tag)

        if not os.path.isfile(skim_cache_path):
            logger.warning(f"read_skim_cache file not found: {skim_cache_path}")
            return None

        logger.info(f"reading skim cache {skim_tag} {skim_data_shape} from {skim_cache_path}")

        data = np.memmap(skim_cache_path, shape=skim_data_shape, dtype=dtype, mode='r')

        return data

    def create_empty_writable_memmap_skim_cache(self, skim_info):
        """
            write skim data from skim_data to canonically named cache file(s) in output directory
        """

        skim_tag = skim_info['skim_tag']
        dtype = np.dtype(skim_info['dtype_name'])
        skim_data_shape = skim_info['skim_data_shape']

        skim_cache_path = self.memmap_skim_data_path(skim_tag)

        logger.info(f"writing skim cache {skim_tag} {skim_data_shape} to {skim_cache_path}")

        data = np.memmap(skim_cache_path, shape=skim_data_shape, dtype=dtype, mode='w+')

        return data

    def copy_omx_to_mmap_file(self, skim_info):

        skim_data = self.create_empty_writable_memmap_skim_cache(skim_info)
        self.read_skims_from_omx(skim_info, skim_data)
        skim_data._mmap.close()
        del skim_data
