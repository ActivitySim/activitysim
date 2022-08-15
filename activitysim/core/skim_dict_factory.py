# ActivitySim
# See full license in LICENSE.txt.
# from builtins import int

import logging
import multiprocessing
import os
from abc import ABC, abstractmethod

import numpy as np
import openmatrix as omx

from activitysim.core import config, inject, skim_dictionary, tracing, util

logger = logging.getLogger(__name__)


class SkimData(object):
    """
    A facade for 3D skim data exposing numpy indexing and shape
    The primary purpose is to document and police the api used to access skim data
    Subclasses using a different backing store to perform additional/alternative
    only need to implement the methods exposed here.

    For instance, to open/close memmapped files just in time, or to access backing data via an alternate api
    """

    def __init__(self, skim_data):
        """
        skim_data is an np.ndarray or anything that implements the methods/properties of this class

        Parameters
        ----------
        skim_data : np.ndarray or quack-alike
        """
        self._skim_data = skim_data

    def __getitem__(self, indexes):
        if len(indexes) != 3:
            raise ValueError(f"number of indexes ({len(indexes)}) should be 3")
        return self._skim_data[indexes]

    @property
    def shape(self):
        """
        Returns
        -------
        list-like shape tuple as returned by numpy.shape
        """
        return self._skim_data.shape


class SkimInfo(object):
    def __init__(self, skim_tag, network_los):
        """

        skim_tag:           str             (e.g. 'TAZ')
        dtype_name:         str             (e.g. 'float32')
        omx_manifest:       dict            dict mapping { omx_key: omx_file_name }
        omx_shape:          2D tuple        shape of omx matrix: (<number_of_zones>, <number_of_zones>)
        num_skims:          int             total number of individual skim matrices in omx files
        skim_data_shape:    3D tuple        (num_skims, omx_shape[0], omx_shape[1]) if ROW_MAJOR_LAYOUT
        offset_map:         dict or None    1D ndarray as returned by omx_file.mapentries, if omx file has mappings
        offset_map_name:    str             name of offset_map in omx_filecorresponding to offset_map, if there was one
        omx_keys:       dict            dict mapping skim key (str or tuple) to skim key in omx file
                                            {DISTWALK: DISTWALK,
                                            ('DRV_COM_WLK_BOARDS', 'AM'): DRV_COM_WLK_BOARDS__AM, ...}
        base_keys:          list of str     e.g. 'BIKEDIST' or 'SOVTOLL_VTOLL' (base key of 3d skim)
        block_offsets:      dict            dict mapping skim key tuple to offset

        Parameters
        ----------
        skim_tag
        """

        self.network_los = network_los
        self.skim_tag = skim_tag
        self.dtype_name = network_los.skim_dtype_name

        self.omx_manifest = None
        self.omx_shape = None
        self.num_skims = None
        self.skim_data_shape = None
        self.offset_map_name = None
        self.offset_map = None
        self.omx_keys = None
        self.base_keys = None
        self.block_offsets = None

        self.load_skim_info(skim_tag)

    def load_skim_info(self, skim_tag):
        """
        Read omx files for skim <skim_tag> (e.g. 'TAZ') and build skim_info dict

        Parameters
        ----------
        skim_tag: str

        """

        omx_file_names = self.network_los.omx_file_names(skim_tag)

        self.omx_file_paths = config.expand_input_file_list(omx_file_names)

        # ignore any 3D skims not in skim_time_periods
        # specifically, load all skims except those with key2 not in dim3_tags_to_load
        skim_time_periods = self.network_los.skim_time_periods
        dim3_tags_to_load = skim_time_periods and skim_time_periods["labels"]

        self.omx_manifest = {}  # dict mapping { omx_key: skim_name }

        for omx_file_path in self.omx_file_paths:

            logger.debug(f"load_skim_info {skim_tag} reading {omx_file_path}")

            with omx.open_file(omx_file_path) as omx_file:

                # fixme call to omx_file.shape() failing in windows p3.5
                if self.omx_shape is None:
                    self.omx_shape = tuple(
                        int(i) for i in omx_file.shape()
                    )  # sometimes omx shape are floats!
                else:
                    assert self.omx_shape == tuple(
                        int(i) for i in omx_file.shape()
                    ), f"Mismatch shape {self.omx_shape} != {omx_file.shape()}"

                for skim_name in omx_file.listMatrices():
                    assert (
                        skim_name not in self.omx_manifest
                    ), f"duplicate skim '{skim_name}' found in {self.omx_manifest[skim_name]} and {omx_file}"
                    self.omx_manifest[skim_name] = omx_file_path

                for m in omx_file.listMappings():
                    if self.offset_map is None:
                        self.offset_map_name = m
                        self.offset_map = omx_file.mapentries(self.offset_map_name)
                        assert len(self.offset_map) == self.omx_shape[0]
                    else:
                        # don't really expect more than one, but ok if they are all the same
                        if not (self.offset_map == omx_file.mapentries(m)):
                            raise RuntimeError(
                                f"Multiple mappings in omx file: {self.offset_map_name} != {m}"
                            )

        # - omx_keys dict maps skim key to omx_key
        # DISTWALK: DISTWALK
        # ('DRV_COM_WLK_BOARDS', 'AM'): DRV_COM_WLK_BOARDS__AM, ...
        self.omx_keys = dict()
        for skim_name in self.omx_manifest.keys():
            key1, sep, key2 = skim_name.partition("__")

            # - ignore composite tags not in dim3_tags_to_load
            if dim3_tags_to_load and sep and key2 not in dim3_tags_to_load:
                continue

            skim_key = (key1, key2) if sep else key1

            self.omx_keys[skim_key] = skim_name

        self.num_skims = len(self.omx_keys)

        # - key1_subkeys dict maps key1 to dict of subkeys with that key1
        # DIST: {'DIST': 0}
        # DRV_COM_WLK_BOARDS: {'MD': 1, 'AM': 0, 'PM': 2}, ...
        key1_subkeys = dict()
        for skim_key, omx_key in self.omx_keys.items():
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
        self.block_offsets = dict()
        for skim_key in self.omx_keys:

            if isinstance(skim_key, tuple):
                key1, key2 = skim_key
            else:
                key1 = key2 = skim_key

            key1_offset = key1_block_offsets[key1]
            key2_relative_offset = key1_subkeys.get(key1).get(key2)
            self.block_offsets[skim_key] = key1_offset + key2_relative_offset

        if skim_dictionary.ROW_MAJOR_LAYOUT:
            self.skim_data_shape = (
                self.num_skims,
                self.omx_shape[0],
                self.omx_shape[1],
            )
        else:
            self.skim_data_shape = self.omx_shape + (self.num_skims,)

        # list of base keys (keys
        self.base_keys = tuple(k for k in key1_block_offsets.keys())

    def print(self):
        print(f"SkimInfo for {self.skim_tag}")
        print(f"omx_shape {self.omx_shape}")
        print(f"num_skims {self.num_skims}")
        print(f"skim_data_shape {self.skim_data_shape}")
        print(f"offset_map_name {self.offset_map_name}")
        # print(f"omx_manifest {self.omx_manifest}")
        # print(f"offset_map {self.offset_map}")
        # print(f"omx_keys {self.omx_keys}")
        # print(f"base_keys {self.base_keys}")
        # print(f"block_offsets {self.block_offsets}")


class AbstractSkimFactory(ABC):
    """
    Provide access to skim data from store.

    load_skim_info(skim_tag: str): dict
        Read omx files for skim <skim_tag> (e.g. 'TAZ') and build skim_info dict

    get_skim_data(skim_tag: str, skim_info: dict): SkimData
        Read skim data from backing store and return it as a 3D ndarray quack-alike SkimData object

    allocate_skim_buffer(skim_info, shared: bool): 1D array buffer sized for 3D SkimData
        Allocate a ram skim buffer (ndarray or multiprocessing.Array) to use as frombuffer for SkimData

    """

    def __init__(self, network_los):
        self.network_los = network_los

    @property
    def supports_shared_data_for_multiprocessing(self):
        """
        Does subclass support shareable data for multiprocessing

        Returns
        -------
        boolean
        """
        return False

    def allocate_skim_buffer(self, skim_info, shared=False):
        """
        For multiprocessing
        """
        assert False, "Not supported"

    def _skim_data_from_buffer(self, skim_info, skim_buffer):
        assert False, "Not supported"

    def _memmap_skim_data_path(self, skim_tag):
        return os.path.join(config.get_cache_dir(), f"cached_{skim_tag}.mmap")

    def load_skim_info(self, skim_tag):
        return SkimInfo(skim_tag, self.network_los)

    def _read_skims_from_omx(self, skim_info, skim_data):
        """
        read skims from omx file into skim_data
        """

        skim_tag = skim_info.skim_tag
        omx_keys = skim_info.omx_keys
        omx_manifest = skim_info.omx_manifest  # dict mapping { omx_key: skim_name }

        for omx_file_path in skim_info.omx_file_paths:

            num_skims_loaded = 0

            logger.info(f"_read_skims_from_omx {omx_file_path}")

            # read skims into skim_data
            with omx.open_file(omx_file_path) as omx_file:
                for skim_key, omx_key in omx_keys.items():

                    if omx_manifest[omx_key] == omx_file_path:

                        offset = skim_info.block_offsets[skim_key]
                        logger.debug(
                            f"_read_skims_from_omx file {omx_file_path} omx_key {omx_key} "
                            f"skim_key {skim_key} to offset {offset}"
                        )

                        if skim_dictionary.ROW_MAJOR_LAYOUT:
                            a = skim_data[offset, :, :]
                        else:
                            a = skim_data[:, :, offset]

                        # this will trigger omx readslice to read and copy data to skim_data's buffer
                        omx_data = omx_file[omx_key]
                        a[:] = omx_data[:]

                        num_skims_loaded += 1

            logger.info(
                f"_read_skims_from_omx loaded {num_skims_loaded} skims from {omx_file_path}"
            )

    def _open_existing_readonly_memmap_skim_cache(self, skim_info):
        """
        read cached memmapped skim data from canonically named cache file(s) in output directory into skim_data
        return True if it was there and we read it, return False if not found
        """

        dtype = np.dtype(skim_info.dtype_name)

        skim_cache_path = self._memmap_skim_data_path(skim_info.skim_tag)

        if not os.path.isfile(skim_cache_path):
            logger.warning(f"read_skim_cache file not found: {skim_cache_path}")
            return None

        logger.info(
            f"reading skim cache {skim_info.skim_tag} {skim_info.skim_data_shape} from {skim_cache_path}"
        )

        try:
            data = np.memmap(
                skim_cache_path, shape=skim_info.skim_data_shape, dtype=dtype, mode="r"
            )
        except Exception as e:
            logger.warning(
                f"{type(e).__name__} reading {skim_info.skim_tag} skim_cache {skim_cache_path}:  {str(e)}"
            )
            logger.warning(
                f"ignoring incompatible {skim_info.skim_tag} skim_cache {skim_cache_path}"
            )
            return None

        return data

    def _create_empty_writable_memmap_skim_cache(self, skim_info):
        """
        write skim data from skim_data to canonically named cache file(s) in output directory
        """

        dtype = np.dtype(skim_info.dtype_name)

        skim_cache_path = self._memmap_skim_data_path(skim_info.skim_tag)

        logger.info(
            f"writing skim cache {skim_info.skim_tag} {skim_info.skim_data_shape} to {skim_cache_path}"
        )

        data = np.memmap(
            skim_cache_path, shape=skim_info.skim_data_shape, dtype=dtype, mode="w+"
        )

        return data

    def copy_omx_to_mmap_file(self, skim_info):

        skim_data = self._create_empty_writable_memmap_skim_cache(skim_info)
        self._read_skims_from_omx(skim_info, skim_data)
        skim_data._mmap.close()
        del skim_data


class NumpyArraySkimFactory(AbstractSkimFactory):
    def __init__(self, network_los):
        super().__init__(network_los)

    @property
    def supports_shared_data_for_multiprocessing(self):
        return True

    def allocate_skim_buffer(self, skim_info, shared=False):
        """
        Allocate a ram skim buffer to use as frombuffer for SkimData
        If shared is True, return a shareable multiprocessing.RawArray, otherwise a numpy.ndarray

        Parameters
        ----------
        skim_info: dict
        shared: boolean

        Returns
        -------
        multiprocessing.RawArray or numpy.ndarray
        """

        assert (
            shared == self.network_los.multiprocess()
        ), f"NumpyArraySkimFactory.allocate_skim_buffer shared {shared} multiprocess {not shared}"

        dtype_name = skim_info.dtype_name
        dtype = np.dtype(dtype_name)

        # multiprocessing.RawArray argument buffer_size must be int, not np.int64
        buffer_size = util.iprod(skim_info.skim_data_shape)

        csz = buffer_size * dtype.itemsize
        logger.info(
            f"allocate_skim_buffer shared {shared} {skim_info.skim_tag} shape {skim_info.skim_data_shape} "
            f"total size: {util.INT(csz)} ({util.GB(csz)})"
        )

        if shared:
            if dtype_name == "float64":
                typecode = "d"
            elif dtype_name == "float32":
                typecode = "f"
            else:
                raise RuntimeError(
                    "allocate_skim_buffer unrecognized dtype %s" % dtype_name
                )

            buffer = multiprocessing.RawArray(typecode, buffer_size)
        else:
            buffer = np.zeros(buffer_size, dtype=dtype)

        return buffer

    def _skim_data_from_buffer(self, skim_info, skim_buffer):
        """
        return a numpy ndarray using skim_buffer as backing store

        Parameters
        ----------
        skim_info
        skim_buffer

        Returns
        -------

        """

        dtype = np.dtype(skim_info.dtype_name)
        assert len(skim_buffer) == util.iprod(skim_info.skim_data_shape)
        skim_data = np.frombuffer(skim_buffer, dtype=dtype).reshape(
            skim_info.skim_data_shape
        )
        return skim_data

    def load_skims_to_buffer(self, skim_info, skim_buffer):
        """
        Load skims from disk store (omx or cache) into ram skim buffer (multiprocessing.RawArray or numpy.ndarray)

        Parameters
        ----------
        skim_info: doct
        skim_buffer: 1D buffer sized to hold all skims (multiprocessing.RawArray or numpy.ndarray)
        """

        read_cache = self.network_los.setting("read_skim_cache", False)
        write_cache = self.network_los.setting("write_skim_cache", False)

        skim_data = self._skim_data_from_buffer(skim_info, skim_buffer)
        assert skim_data.shape == skim_info.skim_data_shape

        if read_cache:
            # returns None if cache file not found
            cache_data = self._open_existing_readonly_memmap_skim_cache(skim_info)

            # copy memmapped cache to RAM numpy ndarray
            if cache_data is not None:
                assert cache_data.shape == skim_data.shape
                np.copyto(skim_data, cache_data)
                cache_data._mmap.close()
                del cache_data
                return

        # read omx skims into skim_buffer (np array)
        self._read_skims_from_omx(skim_info, skim_data)

        if write_cache:
            cache_data = self._create_empty_writable_memmap_skim_cache(skim_info)
            np.copyto(cache_data, skim_data)
            cache_data._mmap.close()
            del cache_data

            # bug - do we need to close it?

        logger.info(
            f"load_skims_to_buffer {skim_info.skim_tag} shape {skim_data.shape}"
        )

    def get_skim_data(self, skim_tag, skim_info):
        """
        Read skim data from backing store and return it as a 3D ndarray quack-alike SkimData object

        Parameters
        ----------
        skim_tag: str
        skim_info: string

        Returns
        -------
        SkimData
        """

        data_buffers = inject.get_injectable("data_buffers", None)
        if data_buffers:
            # we assume any existing skim buffers will already have skim data loaded into them
            logger.info(
                f"get_skim_data {skim_tag} using existing shared skim_buffers for skims"
            )
            skim_buffer = data_buffers[skim_tag]
        else:
            skim_buffer = self.allocate_skim_buffer(skim_info, shared=False)
            self.load_skims_to_buffer(skim_info, skim_buffer)

        skim_data = SkimData(self._skim_data_from_buffer(skim_info, skim_buffer))

        logger.info(
            f"get_skim_data {skim_tag} {type(skim_data).__name__} shape {skim_data.shape}"
        )

        return skim_data


class JitMemMapSkimData(SkimData):
    """
    SkimData subclass for just-in-time memmap.

    Since opening a memmap is fast, open the memmap read the data on demand and immediately close it.
    This essentially eliminates RAM usage, but it means we are loading the data every time we access the skim,
    which may be significantly slower, depending on patterns of usage.
    """

    def __init__(self, skim_cache_path, skim_info):
        super().__init__(skim_info)
        self.skim_cache_path = skim_cache_path
        self.dtype = np.dtype(skim_info.dtype_name)
        self._shape = skim_info.skim_data_shape

    def __getitem__(self, indexes):
        assert len(indexes) == 3, f"number of indexes ({len(indexes)}) should be 3"
        # open memmap
        data = np.memmap(
            self.skim_cache_path, shape=self._shape, dtype=self.dtype, mode="r"
        )
        # dereference skim values
        result = data[indexes]
        # closing memmap's underlying mmap frees data read into (not really needed as we are exiting scope)
        data._mmap.close()
        return result

    @property
    def shape(self):
        return self._shape


class MemMapSkimFactory(AbstractSkimFactory):
    """
    The numpy.memmap docs states: The memmap object can be used anywhere an ndarray is accepted.
    You might think that since memmap duck-types ndarray, we could simply wrap it in a SkimData object.

    But, as the numpy.memmap docs also say: "Memory-mapped files are used for accessing
    small segments of large files on disk, without reading the entire file into memory."

    The words "small segments" are not accidental, because, as you gradually access all the parts
    of the memmapped array, memory usage increases as all the memory is loaded into RAM.

    Under this scenario, the MemMapSkimFactory operates as a just-in-time loader, with no net savings
    in RAM footprint (other than potentially avoiding loading any unused skims).

    Alternatively, since opening a memmap is fast, you could just open the memmap read the data on demand,
    and immediately close it. This essentially eliminates RAM usage, but it means you are loading the data
    every time you access the skim, which, depending on you patterns of usage, may or may not be acceptable.

    """

    def __init__(self, network_los):
        super().__init__(network_los)

    def get_skim_data(self, skim_tag, skim_info):
        """
        Read skim data from backing store and return it as a 3D ndarray quack-alike SkimData object
        (either a JitMemMapSkimData or a memmap backed SkimData object)

        Parameters
        ----------
        skim_tag: str
        skim_info: string

        Returns
        -------
        SkimData or subclass
        """

        # don't expect legacy shared memory buffers
        assert not inject.get_injectable("data_buffers", {}).get(skim_tag)

        skim_cache_path = self._memmap_skim_data_path(skim_tag)
        if not os.path.isfile(skim_cache_path):
            self.copy_omx_to_mmap_file(skim_info)

        JIT = True  # FIXME - this should be a network_los setting, along with selection of the factory?
        if JIT:
            skim_data = JitMemMapSkimData(skim_cache_path, skim_info)
        else:
            # WARNING: memmap gobbles ram up to skim size - see note above
            skim_data = self._open_existing_readonly_memmap_skim_cache(skim_info)
            skim_data = SkimData(skim_data)

        logger.info(
            f"get_skim_data {skim_tag} {type(skim_data).__name__} shape {skim_data.shape}"
        )

        return skim_data
