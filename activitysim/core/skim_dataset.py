from __future__ import annotations

import glob
import logging
import os
import time
from functools import partial
from pathlib import Path

import numpy as np
import openmatrix
import pandas as pd
import sharrow as sh
import xarray as xr

from activitysim.core import config
from activitysim.core import flow as __flow  # noqa: 401
from activitysim.core import workflow
from activitysim.core.input import read_input_file

logger = logging.getLogger(__name__)

POSITIONS_AS_DICT = True


class SkimDataset:
    """
    A wrapper around xarray.Dataset containing skim data, with time period management.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.time_map = {
            j: i for i, j in enumerate(self.dataset.indexes["time_period"])
        }
        self.time_label_dtype = pd.api.types.CategoricalDtype(
            self.dataset.indexes["time_period"],
            ordered=True,
        )
        self.usage = set()  # track keys of skims looked up

    @property
    def odim(self):
        if "omaz" in self.dataset.dims:
            return "omaz"
        else:
            return "otaz"

    @property
    def ddim(self):
        if "dmaz" in self.dataset.dims:
            return "dmaz"
        else:
            return "dtaz"

    def get_skim_usage(self):
        """
        return set of keys of skims looked up. e.g. {'DIST', 'SOV'}

        Returns
        -------
        set:
        """
        return self.usage

    def wrap(self, orig_key, dest_key):
        """
        Get a wrapper for the given keys.

        Parameters
        ----------
        orig_key, dest_key : str

        Returns
        -------
        DatasetWrapper
        """
        return DatasetWrapper(self.dataset, orig_key, dest_key, time_map=self.time_map)

    def wrap_3d(self, orig_key, dest_key, dim3_key):
        """
        Get a 3d wrapper for the given keys.

        Parameters
        ----------
        orig_key, dest_key : str

        Returns
        -------
        DatasetWrapper
        """
        return DatasetWrapper(
            self.dataset, orig_key, dest_key, dim3_key, time_map=self.time_map
        )

    def lookup(self, orig, dest, key):
        """
        Return list of skim values of skims(s) at orig/dest in skim with the specified key (e.g. 'DIST')

        Parameters
        ----------
        orig: list of orig zone_ids
        dest: list of dest zone_ids
        key: str

        Returns
        -------
        Numpy.ndarray: list of skim values for od pairs
        """

        self.usage.add(key)
        use_index = None

        # orig or dest might be a list instead of a series, in which case `index`
        # is a builtin method instead of an array of coordinates, we don't want that.
        if use_index is None and hasattr(orig, "index") and not isinstance(orig, list):
            use_index = orig.index
        if use_index is None and hasattr(dest, "index") and not isinstance(dest, list):
            use_index = dest.index

        orig = np.asanyarray(orig).astype(int)
        dest = np.asanyarray(dest).astype(int)

        some_missing = (orig.min() < 0) or (dest.min() < 0)

        # TODO offset mapper if required
        positions = {self.odim: orig, self.ddim: dest}

        # When asking for a particular time period
        if isinstance(key, tuple) and len(key) == 2:
            main_key, time_key = key
            if time_key in self.time_map:
                positions["time_period"] = np.full_like(orig, self.time_map[time_key])
                key = main_key
            else:
                raise KeyError(key)

        result = self.dataset.iat(
            **positions, _name=key
        )  # Dataset.iat as implemented by sharrow strips data encoding

        if some_missing:
            result[(orig < 0) | (dest < 0)] = np.nan
        result = result.to_series()

        if use_index is not None:
            result.index = use_index
        return result

    def map_time_periods_from_series(self, time_period_labels):
        logger.info(f"vectorize lookup for time_period={time_period_labels.name}")
        time_period_idxs = pd.Series(
            np.vectorize(self.time_map.get)(time_period_labels),
            index=time_period_labels.index,
        )
        return time_period_idxs


class DatasetWrapper:
    """
    Mimics the SkimWrapper interface to allow legacy code to access data.

    Parameters
    ----------
    dataset: Dataset
    orig_key, dest_key: str
        name of columns in target dataframe to use as origin and destination
        lookups, respectively
    time_key : str, optional
    time_map : Mapping, optional
        A mapping from time period index numbers to (more aggregate) time
        period names.
    """

    def __init__(self, dataset, orig_key, dest_key, time_key=None, *, time_map=None):
        """
        Mimics the SkimWrapper interface to allow legacy code to access data.

        """
        self.dataset = dataset
        self.orig_key = orig_key
        self.dest_key = dest_key
        self.time_key = time_key
        self.df = None
        if time_map is None:
            self.time_map = {
                j: i for i, j in enumerate(self.dataset.indexes["time_period"])
            }
        else:
            self.time_map = time_map
        self.time_label_dtype = pd.api.types.CategoricalDtype(
            self.dataset.indexes["time_period"],
            ordered=True,
        )

    @property
    def odim(self):
        if "omaz" in self.dataset.dims:
            return "omaz"
        else:
            return "otaz"

    @property
    def ddim(self):
        if "dmaz" in self.dataset.dims:
            return "dmaz"
        else:
            return "dtaz"

    def map_time_periods(self, df):
        if self.time_key:
            logger.info(f"vectorize lookup for time_period={self.time_key}")
            time_period_idxs = pd.Series(
                np.vectorize(self.time_map.get)(df[self.time_key]),
                index=df.index,
            )
            return time_period_idxs

    def set_df(self, df):
        """
        Set the dataframe

        Parameters
        ----------
        df : DataFrame
            The dataframe which contains the origin and destination ids

        Returns
        -------
        self (to facilitate chaining)
        """
        assert (
            self.orig_key in df
        ), f"orig_key '{self.orig_key}' not in df columns: {list(df.columns)}"
        assert (
            self.dest_key in df
        ), f"dest_key '{self.dest_key}' not in df columns: {list(df.columns)}"
        if self.time_key:
            assert (
                self.time_key in df
            ), f"time_key '{self.time_key}' not in df columns: {list(df.columns)}"
        self.df = df

        # TODO allow offsets if needed
        positions = {
            self.odim: df[self.orig_key],
            self.ddim: df[self.dest_key],
        }
        if self.time_key:
            if (
                not df[self.time_key].dtype == "category"
                and np.issubdtype(df[self.time_key].dtype, np.integer)
                and df[self.time_key].max() < self.dataset.dims["time_period"]
            ):
                logger.info(f"natural use for time_period={self.time_key}")
                positions["time_period"] = df[self.time_key]
            elif (
                df[self.time_key].dtype == "category"
                and df[self.time_key].dtype == self.time_label_dtype
            ):
                positions["time_period"] = df[self.time_key].cat.codes
            else:
                logger.info(f"vectorize lookup for time_period={self.time_key}")
                positions["time_period"] = pd.Series(
                    np.vectorize(self.time_map.get, "I")(df[self.time_key], 0),
                    index=df.index,
                )

        if POSITIONS_AS_DICT:
            self.positions = {}
            for k, v in positions.items():
                try:
                    is_int = np.issubdtype(v.dtype, np.integer)
                except Exception:
                    is_int = False
                if is_int:
                    self.positions[k] = v
                else:
                    try:
                        self.positions[k] = v.astype(int)
                    except TypeError:
                        # possibly some missing values that are not relevant,
                        # fill with zeros to continue.
                        self.positions[k] = v.fillna(0).astype(int)
        else:
            self.positions = pd.DataFrame(positions).astype(int)

        return self

    def lookup(self, key, reverse=False):
        """
        Generally not called by the user - use __getitem__ instead

        Parameters
        ----------
        key : hashable
             The key (identifier) for this skim object

        od : bool (optional)
            od=True means lookup standard origin-destination skim value
            od=False means lookup destination-origin skim value

        Returns
        -------
        impedances: pd.Series
            A Series of impedances which are elements of the Skim object and
            with the same index as df
        """

        assert self.df is not None, "Call set_df first"
        if reverse:
            if isinstance(self.positions, dict):
                x = self.positions.copy()
                x.update(
                    {
                        self.odim: self.positions[self.ddim],
                        self.ddim: self.positions[self.odim],
                    }
                )
            else:
                x = self.positions.rename(
                    columns={self.odim: self.ddim, self.ddim: self.odim}
                )
        else:
            if isinstance(self.positions, dict):
                x = self.positions.copy()
            else:
                x = self.positions

        # When asking for a particular time period
        if isinstance(key, tuple) and len(key) == 2:
            main_key, time_key = key
            if time_key in self.time_map:
                if isinstance(x, dict):
                    # np.broadcast_to saves memory over np.full_like, since we
                    # don't ever write to this array.
                    x["time_period"] = np.broadcast_to(
                        self.time_map[time_key], x[self.odim].shape
                    )
                else:
                    x = x.assign(time_period=self.time_map[time_key])
                key = main_key
            else:
                raise KeyError(key)

        result = self.dataset.iat(**x, _name=key)  # iat strips data encoding
        # if 'digital_encoding' in self.dataset[key].attrs:
        #     result = array_decode(result, self.dataset[key].attrs['digital_encoding'])

        # Return a series, consistent with ActivitySim SkimWrapper
        out = result.to_series()
        out.index = self.df.index
        return out

    def reverse(self, key):
        """
        return skim value in reverse (d-o) direction
        """
        return self.lookup(key, reverse=True)

    def max(self, key):
        """
        return max skim value in either o-d or d-o direction
        """
        assert self.df is not None, "Call set_df first"

        s = np.maximum(
            self.lookup(key),
            self.lookup(key, True),
        )

        return pd.Series(s, index=self.df.index)

    def __getitem__(self, key):
        """
        Get the lookup for an available skim object (df and orig/dest and column names implicit)

        Parameters
        ----------
        key : hashable
             The key (identifier) for the skim object

        Returns
        -------
        impedances: pd.Series with the same index as df
            A Series of impedances values from the single Skim with specified key, indexed byt orig/dest pair
        """
        return self.lookup(key)


def _should_invalidate_cache_file(cache_filename, *source_filenames):
    """
    Check if a cache file should be invalidated.

    It should be invalidated if any source file has a modification time
    more recent than the cache file modification time.

    Parameters
    ----------
    cache_filename : Path-like
    source_filenames : Collection[Path-like]

    Returns
    -------
    bool
    """
    from stat import ST_MTIME

    try:
        stat0 = os.stat(cache_filename)
    except FileNotFoundError:
        # cache file does not even exist
        return True
    for i in source_filenames:
        stat1 = os.stat(i)
        if stat0[ST_MTIME] < stat1[ST_MTIME]:
            return True
    return False


def _use_existing_backing_if_valid(backing, omx_file_paths, skim_tag):
    """
    Open an xarray dataset from a backing store if possible.

    Parameters
    ----------
    backing : str
        What kind of memory backing to use.  Memmaps always start
        with "memmap:" and then have a file system location, so
        if this pattern does not apply the backing is not a memmap,
        and instead the backing string is used as the key to find
        the data in ephemeral shared memory.
    omx_file_paths : Collection[Path-like]
        These are the original source files.  If the file modification
        time for any of these files is more recent than the memmap,
        the memmap files are invalid and will be deleted so they
        can be rebuilt.
    skim_tag : str
        For error message reporting only

    Returns
    -------
    xarray.Dataset or None
    """
    out = None
    if backing.startswith("memmap:"):
        # when working with a memmap, check if the memmap file on disk
        # needs to be invalidated, because the source skims have been
        # modified more recently.
        if not _should_invalidate_cache_file(backing[7:], *omx_file_paths):
            try:
                out = sh.Dataset.shm.from_shared_memory(backing, mode="r")
            except FileNotFoundError as err:
                logger.info(f"skim dataset {skim_tag!r} not found {err!s}")
                logger.info(f"loading skim dataset {skim_tag!r} from original sources")
                out = None
            else:
                logger.info("using skim_dataset from shared memory")
        else:
            sh.Dataset.shm.delete_shared_memory_files(backing)
    else:
        # when working in ephemeral shared memory, assume that if that data
        # is loaded then it is good to use without further checks.
        try:
            out = sh.Dataset.shm.from_shared_memory(backing, mode="r")
        except FileNotFoundError as err:
            logger.info(f"skim dataset {skim_tag!r} not found {err!s}")
            logger.info(f"loading skim dataset {skim_tag!r} from original sources")
            out = None
    return out


def _dedupe_time_periods(network_los_preload):
    raw_time_periods = network_los_preload.los_settings.skim_time_periods.labels
    # deduplicate time period names
    time_periods = []
    for t in raw_time_periods:
        if t not in time_periods:
            time_periods.append(t)
    return time_periods


def _apply_digital_encoding(dataset, digital_encodings):
    """
    Apply digital encoding to compress skims with minimal information loss.

    Parameters
    ----------
    dataset : xarray.Dataset
    digital_encodings : Collection[Dict]
        A collection of digital encoding instructions.  To apply the same
        encoding for multiple variables, the Dict should have a 'regex' key
        that gives a regular expressing to match.  Otherwise, see
        sharrow's digital_encoding for other details.

    Returns
    -------
    dataset : xarray.Dataset
        As modified
    """
    if digital_encodings:
        import re

        # apply once, before saving to zarr, will stick around in cache
        for encoding in digital_encodings:
            logger.info(f"applying zarr digital-encoding: {encoding}")
            regex = encoding.pop("regex", None)
            joint_dict = encoding.pop("joint_dict", None)
            if joint_dict:
                joins = []
                for k in dataset.variables:
                    assert isinstance(k, str)  # variable names should be strings
                    if re.match(regex, k):
                        joins.append(k)
                dataset = dataset.digital_encoding.set(
                    joins, joint_dict=joint_dict, **encoding
                )
            elif regex:
                if "name" in encoding:
                    raise ValueError(
                        "cannot give both name and regex for digital_encoding"
                    )
                for k in dataset.variables:
                    assert isinstance(k, str)  # variable names should be strings
                    if re.match(regex, k):
                        dataset = dataset.digital_encoding.set(k, **encoding)
            else:
                dataset = dataset.digital_encoding.set(**encoding)
    return dataset


def _scan_for_unused_names(state, tokens):
    """
    Scan all spec files to find unused skim variable names.

    Parameters
    ----------
    state : State
    tokens : Collection[str]

    Returns
    -------
    Set[str]
    """
    configs_dir_list = state.filesystem.get_configs_dir()
    configs_dir_list = (
        [configs_dir_list]
        if isinstance(configs_dir_list, (str, Path))
        else configs_dir_list
    )
    if isinstance(configs_dir_list, tuple):
        configs_dir_list = list(configs_dir_list)
    assert isinstance(configs_dir_list, list)

    for directory in configs_dir_list:
        logger.debug(f"scanning for unused skims in {directory}")
        filenames = glob.glob(os.path.join(directory, "*.csv"))
        for filename in filenames:
            with open(filename, "rt") as f:
                content = f.read()
            missing_tokens = set()
            for t in tokens:
                if t not in content:
                    missing_tokens.add(t)
            tokens = missing_tokens
            if not tokens:
                return tokens
    return tokens


def _drop_unused_names(state, dataset):
    logger.info("scanning for unused skims")
    tokens = set(dataset.variables.keys()) - set(dataset.coords.keys())
    unused_tokens = _scan_for_unused_names(state, tokens)
    if unused_tokens:
        baggage = dataset.digital_encoding.baggage(None)
        unused_tokens -= baggage
        # retain sparse matrix tables
        unused_tokens = set(i for i in unused_tokens if not i.startswith("_s_"))
        # retain lookup tables
        unused_tokens = set(i for i in unused_tokens if not i.startswith("_digitized_"))
        logger.info(f"dropping unused skims: {unused_tokens}")
        dataset = dataset.drop_vars(unused_tokens)
    else:
        logger.info("no unused skims found")
    return dataset


def load_sparse_maz_skims(
    dataset,
    land_use_index,
    remapper,
    zone_system,
    maz2taz_file_name,
    maz_to_maz_tables=(),
    max_blend_distance=None,
    data_file_resolver=None,
):
    """
    Load sparse MAZ data on top of TAZ skim data.

    Parameters
    ----------
    dataset : xarray.Dataset
        The existing dataset at TAZ resolution only.
    land_use_index : pandas.Index
        The index of the land use table.  For two and three zone systems,
        these index values should be MAZ identifiers.
    remapper : dict, optional
        A dictionary mapping where the keys are the original (nominal) zone
        id's, and the values are the recoded (typically zero-based contiguous)
        zone id's.  Recoding improves runtime efficiency.
    zone_system : int
        Currently 1, 2 and 3 are supported.
    maz2taz_file_name : str
    maz_to_maz_tables : Collection[]
    max_blend_distance : optional
    data_file_resolver : function or Callable

    Returns
    -------
    xarray.Dataset
    """
    from ..core.los import THREE_ZONE, TWO_ZONE

    if data_file_resolver is None:
        raise ValueError("missing file resolver")

    if zone_system in [TWO_ZONE, THREE_ZONE]:
        # maz
        maz_filename = data_file_resolver(maz2taz_file_name, mandatory=True)
        maz_taz = read_input_file(maz_filename)
        maz_taz = maz_taz[["MAZ", "TAZ"]].set_index("MAZ").sort_index()

        # MAZ alignment is ensured here, so no re-alignment check is
        # needed below for TWO_ZONE or THREE_ZONE systems
        try:
            pd.testing.assert_index_equal(
                maz_taz.index, land_use_index, check_names=False
            )
        except AssertionError:
            if remapper is not None:
                maz_taz.index = maz_taz.index.map(remapper.get)
                maz_taz = maz_taz.sort_index()
                assert maz_taz.index.equals(
                    land_use_index.sort_values()
                ), "maz-taz lookup index does not match index of land_use table"
            else:
                raise

        dataset.redirection.set(
            maz_taz,
            map_to="otaz",
            name="omaz",
            map_also={"dtaz": "dmaz"},
        )

        maz_to_maz_tables = (
            [maz_to_maz_tables]
            if isinstance(maz_to_maz_tables, str)
            else maz_to_maz_tables
        )

        if max_blend_distance is None:
            max_blend_distance = {}
        if isinstance(max_blend_distance, int):
            max_blend_distance = {"DEFAULT": max_blend_distance}

        for file_name in maz_to_maz_tables:
            df = read_input_file(data_file_resolver(file_name, mandatory=True))
            if remapper is not None:
                df.OMAZ = df.OMAZ.map(remapper.get)
                df.DMAZ = df.DMAZ.map(remapper.get)
            for colname in df.columns:
                if colname in ["OMAZ", "DMAZ"]:
                    continue
                max_blend_distance_i = max_blend_distance.get("DEFAULT", None)
                max_blend_distance_i = max_blend_distance.get(
                    colname, max_blend_distance_i
                )
                dataset.redirection.sparse_blender(
                    colname,
                    df.OMAZ,
                    df.DMAZ,
                    df[colname],
                    max_blend_distance=max_blend_distance_i,
                    index=land_use_index,
                )

    return dataset


def load_skim_dataset_to_shared_memory(state, skim_tag="taz") -> xr.Dataset:
    """
    Load skims from disk into shared memory.

    Parameters
    ----------
    state : State
    skim_tag : str, default "taz"

    Returns
    -------
    xarray.Dataset
    """
    from activitysim.core.los import ONE_ZONE

    # TODO:SHARROW: taz and maz are the same
    network_los_preload = state.get_injectable("network_los_preload")
    if network_los_preload is None:
        raise ValueError("missing network_los_preload")

    # find which OMX files are to be used.
    omx_file_paths = state.filesystem.expand_input_file_list(
        network_los_preload.omx_file_names(skim_tag),
    )
    omx_file_handles = []
    zarr_file = network_los_preload.zarr_file_name(skim_tag)

    if state.settings.disable_zarr:
        # we can disable the zarr optimizations by setting the `disable_zarr`
        # flag in the master config file to True
        zarr_file = None

    if zarr_file is not None:
        zarr_file = os.path.join(state.filesystem.get_cache_dir(), zarr_file)

    max_float_precision = network_los_preload.skim_max_float_precision(skim_tag)

    skim_digital_encoding = network_los_preload.skim_digital_encoding(skim_tag)
    zarr_digital_encoding = network_los_preload.zarr_pre_encoding(skim_tag)

    # The backing can be plain shared_memory, or a memmap
    backing = network_los_preload.skim_backing_store(skim_tag)
    if backing == "memmap":
        # if memmap is given without a path, create a cache file
        mmap_file = os.path.join(
            config.get_cache_dir(), f"sharrow_dataset_{skim_tag}.mmap"
        )
        backing = f"memmap:{mmap_file}"

    land_use = state.get_dataframe("land_use")

    if f"_original_{land_use.index.name}" in land_use:
        land_use_zone_ids = land_use[f"_original_{land_use.index.name}"]
        remapper = dict(zip(land_use_zone_ids, land_use_zone_ids.index))
    else:
        remapper = None

    if state.settings.store_skims_in_shm:
        d = _use_existing_backing_if_valid(backing, omx_file_paths, skim_tag)
    else:
        d = None  # skims are not stored in shared memory, so we need to load them
    do_not_save_zarr = False

    if d is None:
        time_periods = _dedupe_time_periods(network_los_preload)
        if zarr_file:
            logger.info(f"looking for zarr skims at {zarr_file}")
        if zarr_file and os.path.exists(zarr_file):
            from .util import latest_file_modification_time

            logger.info("found zarr skims, loading them")
            d = sh.dataset.from_zarr_with_attr(zarr_file)
            zarr_write_time = d.attrs.get("ZARR_WRITE_TIME", 0)
            if zarr_write_time < latest_file_modification_time(omx_file_paths):
                logger.warning("zarr skims older than omx, not using them")
                do_not_save_zarr = True
                d = None
            else:
                d = d.max_float_precision(max_float_precision)
        if d is None:
            if zarr_file and not do_not_save_zarr:
                logger.info("did not find zarr skims, loading omx")
            omx_file_handles = [
                openmatrix.open_file(f, mode="r") for f in omx_file_paths
            ]
            d = sh.dataset.from_omx_3d(
                omx_file_handles,
                index_names=(
                    ("otap", "dtap", "time_period")
                    if skim_tag == "tap"
                    else ("otaz", "dtaz", "time_period")
                ),
                time_periods=time_periods,
                max_float_precision=max_float_precision,
                ignore=state.settings.omx_ignore_patterns,
            )

            if zarr_file:
                try:
                    import zarr  # noqa

                    # ensure zarr is available before we do all this work
                except ModuleNotFoundError:
                    logger.warning(
                        "the 'zarr' package is not installed, "
                        "cannot cache skims to zarr"
                    )
                else:
                    if zarr_digital_encoding:
                        d = _apply_digital_encoding(d, zarr_digital_encoding)
                    logger.info(f"writing zarr skims to {zarr_file}")
                    d.attrs["ZARR_WRITE_TIME"] = time.time()
                    if not do_not_save_zarr:
                        d.to_zarr_with_attr(zarr_file)

        if skim_tag in ("taz", "maz"):
            # load sparse MAZ skims, if any
            # these are processed after the ZARR stuff as the GCXS sparse array
            # is not yet compatible with ZARR directly.
            # see https://github.com/pydata/sparse/issues/222
            #  or https://github.com/zarr-developers/zarr-python/issues/424
            maz2taz_file_name = network_los_preload.setting("maz", None)
            if maz2taz_file_name:
                d = load_sparse_maz_skims(
                    d,
                    land_use.index,
                    remapper,
                    zone_system=network_los_preload.zone_system,
                    maz2taz_file_name=network_los_preload.setting("maz"),
                    maz_to_maz_tables=network_los_preload.setting("maz_to_maz.tables"),
                    max_blend_distance=network_los_preload.setting(
                        "maz_to_maz.max_blend_distance", default={}
                    ),
                    data_file_resolver=partial(
                        state.filesystem.get_data_file_path,
                        alternative_suffixes=(".csv.gz", ".parquet"),
                    ),
                )

        d = _drop_unused_names(state, d)
        # apply non-zarr dependent digital encoding
        d = _apply_digital_encoding(d, skim_digital_encoding)

    if skim_tag in ("taz", "maz"):
        # check alignment of TAZs that it matches land_use table
        logger.info("checking skims alignment with land_use")
        try:
            land_use_zone_id = land_use[f"_original_{land_use.index.name}"]
        except KeyError:
            land_use_zone_id = land_use.index
    else:
        land_use_zone_id = None

    dask_required = False
    if network_los_preload.zone_system == ONE_ZONE:
        # check TAZ alignment for ONE_ZONE system.
        # other systems use MAZ for most lookups, which dynamically
        # resolves to TAZ inside the Dataset code.
        if d["otaz"].attrs.get("preprocessed") != "zero-based-contiguous":
            try:
                np.testing.assert_array_equal(land_use_zone_id, d.otaz)
            except AssertionError as err:
                logger.info(f"otaz realignment required\n{err}")
                d = d.reindex(otaz=land_use_zone_id)
                dask_required = True
            else:
                logger.info("otaz alignment ok")
            d["otaz"] = land_use.index.to_numpy()
            d["otaz"].attrs["preprocessed"] = "zero-based-contiguous"
        else:
            np.testing.assert_array_equal(land_use.index, d.otaz)

        if d["dtaz"].attrs.get("preprocessed") != "zero-based-contiguous":
            try:
                np.testing.assert_array_equal(land_use_zone_id, d.dtaz)
            except AssertionError as err:
                logger.info(f"dtaz realignment required\n{err}")
                d = d.reindex(dtaz=land_use_zone_id)
                dask_required = True
            else:
                logger.info("dtaz alignment ok")
            d["dtaz"] = land_use.index.to_numpy()
            d["dtaz"].attrs["preprocessed"] = "zero-based-contiguous"
        else:
            np.testing.assert_array_equal(land_use.index, d.dtaz)

    if d.shm.is_shared_memory:
        for f in omx_file_handles:
            f.close()
        return d
    elif not state.settings.store_skims_in_shm:
        logger.info(
            "store_skims_in_shm is False, keeping skims in process-local memory"
        )
        return d
    else:
        logger.info("writing skims to shared memory")
        if dask_required:
            # setting `load` to True uses dask to load the data into memory
            d_shared_mem = d.shm.to_shared_memory(backing, mode="r", load=True)
        else:
            # setting `load` to false then calling `reload_from_omx_3d` avoids
            # using dask to load the data into memory, which is not performant
            # on Windows for large datasets, but this only works if the data
            # requires no realignment (i.e. the land use table and skims match
            # exactly in order and length).
            d_shared_mem = d.shm.to_shared_memory(backing, mode="r", load=False)
            sh.dataset.reload_from_omx_3d(
                d_shared_mem,
                [str(i) for i in omx_file_paths],
                ignore=state.settings.omx_ignore_patterns,
            )
        for f in omx_file_handles:
            f.close()
        return d_shared_mem


@workflow.cached_object
def skim_dataset(state: workflow.State) -> xr.Dataset:
    return load_skim_dataset_to_shared_memory(state)


@workflow.cached_object
def tap_dataset(state: workflow.State) -> xr.Dataset:
    return load_skim_dataset_to_shared_memory(state, "tap")
