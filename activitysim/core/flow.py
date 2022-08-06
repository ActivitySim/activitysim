import contextlib
import glob
import hashlib
import logging
import os
import re
import time
from datetime import timedelta
from numbers import Number
from stat import ST_MTIME
from typing import Mapping

import numpy as np
import openmatrix
import pandas as pd
from orca import orca

from .. import __version__
from ..core import tracing
from . import config, inject
from .simulate_consts import SPEC_EXPRESSION_NAME, SPEC_LABEL_NAME

try:
    import sharrow as sh
except ModuleNotFoundError:
    sh = None


logger = logging.getLogger(__name__)

_FLOWS = {}

if os.environ.get("TRAVIS") == "true":
    # The multithreaded dask scheduler causes problems on travis.
    # Here, we detect if this code is running on Travis, and if so, we
    # change the default scheduler to single-threaded.  This should not
    # be particularly problematic, as only tiny test cases are run on Travis.
    import dask

    dask.config.set(
        scheduler="single-threaded"
    )  # overwrite default with threaded scheduler


@contextlib.contextmanager
def logtime(tag, tag2=""):
    logger.info(f"begin {tag} {tag2}")
    t0 = time.time()
    try:
        yield
    except Exception:
        logger.error(f"error in {tag} after {timedelta(seconds=time.time()-t0)} {tag2}")
        raise
    else:
        logger.info(f"completed {tag} in {timedelta(seconds=time.time()-t0)} {tag2}")


class TimeLogger:

    aggregate_timing = {}

    def __init__(self, tag1):
        self._time_point = self._time_start = time.time()
        self._time_log = []
        self._tag1 = tag1

    def mark(self, tag, ping=True, logger=None, suffix=""):
        if ping:
            now = time.time()
            elapsed = now - self._time_point
            self._time_log.append((tag, timedelta(seconds=elapsed)))
            self._time_point = now
            if logger is not None:
                logger.info(
                    "elapsed time {0} {1} {2}".format(
                        tag,
                        timedelta(seconds=elapsed),
                        suffix,
                    )
                )
        else:
            self._time_log.append((tag, "skipped"))
            elapsed = 0
        if self._tag1:
            tag = f"{self._tag1}.{tag}"
        if tag not in self.aggregate_timing:
            self.aggregate_timing[tag] = elapsed
        else:
            self.aggregate_timing[tag] += elapsed

    def summary(self, logger, tag, level=20, suffix=None):
        gross_elaspsed = time.time() - self._time_start
        if suffix:
            msg = f"{tag} in {timedelta(seconds=gross_elaspsed)}: ({suffix})\n"
        else:
            msg = f"{tag} in {timedelta(seconds=gross_elaspsed)}: \n"
        msgs = []
        for i in self._time_log:
            j = timedelta(seconds=self.aggregate_timing[f"{self._tag1}.{i[0]}"])
            msgs.append("   - {0:24s} {1} [{2}]".format(*i, j))
        msg += "\n".join(msgs)
        logger.log(level=level, msg=msg)

    @classmethod
    def aggregate_summary(
        cls, logger, heading="Aggregate Flow Timing Summary", level=20
    ):
        msg = f"{heading}\n"
        msgs = []
        for tag, elapsed in cls.aggregate_timing.items():
            msgs.append("   - {0:48s} {1}".format(tag, timedelta(seconds=elapsed)))
        msg += "\n".join(msgs)
        logger.log(level=level, msg=msg)


def only_simple(x, exclude_keys=()):
    """
    All the values in a dict that are plain numbers, strings, or lists or tuples thereof.
    """
    y = {}
    for k, v in x.items():
        if k not in exclude_keys:
            if isinstance(v, (Number, str)):
                y[k] = v
            elif isinstance(v, (list, tuple)):
                if all(isinstance(j, (Number, str)) for j in v):
                    y[k] = v
    return y


def get_flow(spec, local_d, trace_label=None, choosers=None, interacts=None):
    global _FLOWS
    extra_vars = only_simple(local_d)
    orig_col_name = local_d.get("orig_col_name", None)
    dest_col_name = local_d.get("dest_col_name", None)
    stop_col_name = None
    parking_col_name = None
    timeframe = local_d.get("timeframe", "tour")
    if timeframe == "trip":
        orig_col_name = local_d.get("ORIGIN", orig_col_name)
        dest_col_name = local_d.get("DESTINATION", dest_col_name)
        parking_col_name = local_d.get("PARKING", parking_col_name)
        if orig_col_name is None and "od_skims" in local_d:
            orig_col_name = local_d["od_skims"].orig_key
        if dest_col_name is None and "od_skims" in local_d:
            dest_col_name = local_d["od_skims"].dest_key
        if stop_col_name is None and "dp_skims" in local_d:
            stop_col_name = local_d["dp_skims"].dest_key
    local_d = size_terms_on_flow(local_d)
    size_term_mapping = local_d.get("size_array", {})
    flow = new_flow(
        spec,
        extra_vars,
        orig_col_name,
        dest_col_name,
        trace_label,
        timeframe=timeframe,
        choosers=choosers,
        stop_col_name=stop_col_name,
        parking_col_name=parking_col_name,
        size_term_mapping=size_term_mapping,
        interacts=interacts,
    )
    return flow


def should_invalidate_cache_file(cache_filename, *source_filenames):
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


@inject.injectable(cache=True)
def skim_dataset():
    from ..core.los import ONE_ZONE, THREE_ZONE, TWO_ZONE

    # TODO:SHARROW: taz and maz are the same
    skim_tag = "taz"
    network_los_preload = inject.get_injectable("network_los_preload", None)
    if network_los_preload is None:
        raise ValueError("missing network_los_preload")

    # find which OMX files are to be used.
    omx_file_paths = config.expand_input_file_list(
        network_los_preload.omx_file_names(skim_tag),
    )
    zarr_file = network_los_preload.zarr_file_name(skim_tag)

    if config.setting("disable_zarr", False):
        # we can disable the zarr optimizations by setting the `disable_zarr`
        # flag in the master config file to True
        zarr_file = None

    if zarr_file is not None:
        zarr_file = os.path.join(config.get_cache_dir(), zarr_file)

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

    with logtime("loading skims as dataset"):

        land_use = inject.get_table("land_use")

        if f"_original_{land_use.index.name}" in land_use.to_frame():
            land_use_zone_ids = land_use.to_frame()[f"_original_{land_use.index.name}"]
            remapper = dict(zip(land_use_zone_ids, land_use_zone_ids.index))
        else:
            remapper = None

        d = None
        if backing.startswith("memmap:"):
            # when working with a memmap, check if the memmap file on disk
            # needs to be invalidated, because the source skims have been
            # modified more recently.
            if not should_invalidate_cache_file(backing[7:], *omx_file_paths):
                try:
                    d = sh.Dataset.shm.from_shared_memory(backing, mode="r")
                except FileNotFoundError as err:
                    logger.info(f"skim dataset {skim_tag!r} not found {err!s}")
                    logger.info(f"loading skim dataset {skim_tag!r} from disk")
                    d = None
                else:
                    logger.info(f"using skim_dataset from shared memory")
            else:
                sh.Dataset.shm.delete_shared_memory_files(backing)
        else:
            # when working in ephemeral shared memory, assume that if that data
            # is loaded then it is good to use without further checks.
            try:
                d = sh.Dataset.shm.from_shared_memory(backing, mode="r")
            except FileNotFoundError as err:
                logger.info(f"skim dataset {skim_tag!r} not found {err!s}")
                logger.info(f"loading skim dataset {skim_tag!r} from disk")
                d = None

        if d is None:
            time_periods_ = network_los_preload.los_settings["skim_time_periods"][
                "labels"
            ]
            # deduplicate time period names
            time_periods = []
            for t in time_periods_:
                if t not in time_periods:
                    time_periods.append(t)
            if zarr_file:
                logger.info(f"looking for zarr skims at {zarr_file}")
            if zarr_file and os.path.exists(zarr_file):
                # TODO: check if the OMX skims are modified more recently than
                #       the cached ZARR versions; if so do not use the ZARR
                # load skims from zarr.zip
                logger.info(f"found zarr skims, loading them")
                d = sh.dataset.from_zarr_with_attr(zarr_file).max_float_precision(
                    max_float_precision
                )
            else:
                if zarr_file:
                    logger.info(f"did not find zarr skims, loading omx")
                d = sh.dataset.from_omx_3d(
                    [openmatrix.open_file(f, mode="r") for f in omx_file_paths],
                    time_periods=time_periods,
                    max_float_precision=max_float_precision,
                )
                # load sparse MAZ skims, if any
                if network_los_preload.zone_system in [TWO_ZONE, THREE_ZONE]:

                    # maz
                    maz2taz_file_name = network_los_preload.setting("maz")
                    maz_taz = pd.read_csv(
                        config.data_file_path(maz2taz_file_name, mandatory=True)
                    )
                    maz_taz = maz_taz[["MAZ", "TAZ"]].set_index("MAZ").sort_index()

                    # MAZ alignment is ensured here, so no re-alignment check is
                    # needed below for TWO_ZONE or THREE_ZONE systems
                    try:
                        pd.testing.assert_index_equal(
                            maz_taz.index, land_use.index, check_names=False
                        )
                    except AssertionError:
                        if remapper is not None:
                            maz_taz.index = maz_taz.index.map(remapper.get)
                            maz_taz = maz_taz.sort_index()
                            assert maz_taz.index.equals(
                                land_use.to_frame().sort_index().index
                            ), f"maz-taz lookup index does not match index of land_use table"
                        else:
                            raise

                    d.redirection.set(
                        maz_taz,
                        map_to="otaz",
                        name="omaz",
                        map_also={"dtaz": "dmaz"},
                    )

                    maz_to_maz_tables = network_los_preload.setting("maz_to_maz.tables")
                    maz_to_maz_tables = (
                        [maz_to_maz_tables]
                        if isinstance(maz_to_maz_tables, str)
                        else maz_to_maz_tables
                    )

                    max_blend_distance = network_los_preload.setting(
                        "maz_to_maz.max_blend_distance", default={}
                    )
                    if isinstance(max_blend_distance, int):
                        max_blend_distance = {"DEFAULT": max_blend_distance}

                    for file_name in maz_to_maz_tables:

                        df = pd.read_csv(
                            config.data_file_path(file_name, mandatory=True)
                        )
                        if remapper is not None:
                            df.OMAZ = df.OMAZ.map(remapper.get)
                            df.DMAZ = df.DMAZ.map(remapper.get)
                        for colname in df.columns:
                            if colname in ["OMAZ", "DMAZ"]:
                                continue
                            max_blend_distance_i = max_blend_distance.get(
                                "DEFAULT", None
                            )
                            max_blend_distance_i = max_blend_distance.get(
                                colname, max_blend_distance_i
                            )
                            d.redirection.sparse_blender(
                                colname,
                                df.OMAZ,
                                df.DMAZ,
                                df[colname],
                                max_blend_distance=max_blend_distance_i,
                                index=land_use.index,
                            )

                if zarr_file:
                    if zarr_digital_encoding:
                        import zarr  # ensure zarr is available before we do all this work.

                        # apply once, before saving to zarr, will stick around in cache
                        for encoding in zarr_digital_encoding:
                            logger.info(f"applying zarr digital-encoding: {encoding}")
                            regex = encoding.pop("regex", None)
                            joint_dict = encoding.pop("joint_dict", None)
                            if joint_dict:
                                joins = []
                                for k in d.variables:
                                    if re.match(regex, k):
                                        joins.append(k)
                                d = d.digital_encoding.set(
                                    joins, joint_dict=joint_dict, **encoding
                                )
                            elif regex:
                                if "name" in encoding:
                                    raise ValueError(
                                        "cannot give both name and regex for digital_encoding"
                                    )
                                for k in d.variables:
                                    if re.match(regex, k):
                                        d = d.digital_encoding.set(k, **encoding)
                            else:
                                d = d.digital_encoding.set(**encoding)

                    logger.info(f"writing zarr skims to {zarr_file}")
                    # save skims to zarr
                    try:
                        d.to_zarr_with_attr(zarr_file)
                    except ModuleNotFoundError:
                        logger.warning("the 'zarr' package is not installed")
            logger.info(f"scanning for unused skims")
            tokens = set(d.variables.keys()) - set(d.coords.keys())
            unused_tokens = scan_for_unused_names(tokens)
            if unused_tokens:
                baggage = d.digital_encoding.baggage(None)
                unused_tokens -= baggage
                # retain sparse matrix tables
                unused_tokens = set(i for i in unused_tokens if not i.startswith("_s_"))
                # retain lookup tables
                unused_tokens = set(
                    i for i in unused_tokens if not i.startswith("_digitized_")
                )
                logger.info(f"dropping unused skims: {unused_tokens}")
                d = d.drop_vars(unused_tokens)
            else:
                logger.info(f"no unused skims found")
            # apply digital encoding
            if skim_digital_encoding:
                for encoding in skim_digital_encoding:
                    regex = encoding.pop("regex", None)
                    if regex:
                        if "name" in encoding:
                            raise ValueError(
                                "cannot give both name and regex for digital_encoding"
                            )
                        for k in d.variables:
                            if re.match(regex, k):
                                d = d.digital_encoding.set(k, **encoding)
                    else:
                        d = d.digital_encoding.set(**encoding)

        # check alignment of TAZs that it matches land_use table
        logger.info(f"checking skims alignment with land_use")
        try:
            land_use_zone_id = land_use[f"_original_{land_use.index.name}"]
        except KeyError:
            land_use_zone_id = land_use.index

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
                else:
                    logger.info(f"otaz alignment ok")
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
                else:
                    logger.info(f"dtaz alignment ok")
                d["dtaz"] = land_use.index.to_numpy()
                d["dtaz"].attrs["preprocessed"] = "zero-based-contiguous"
            else:
                np.testing.assert_array_equal(land_use.index, d.dtaz)

        if d.shm.is_shared_memory:
            return d
        else:
            logger.info(f"writing skims to shared memory")
            return d.shm.to_shared_memory(backing, mode="r")


def scan_for_unused_names(tokens):
    """
    Scan all spec files to find unused skim variable names.

    Parameters
    ----------
    tokens : Collection[str]

    Returns
    -------
    Set[str]
    """
    configs_dir_list = inject.get_injectable("configs_dir")
    configs_dir_list = (
        [configs_dir_list] if isinstance(configs_dir_list, str) else configs_dir_list
    )
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


@inject.injectable(cache=True)
def skim_dataset_dict(skim_dataset):
    from .skim_dataset import SkimDataset

    return SkimDataset(skim_dataset)


def skims_mapping(
    orig_col_name,
    dest_col_name,
    timeframe="tour",
    stop_col_name=None,
    parking_col_name=None,
):
    logger.info(f"loading skims_mapping")
    logger.info(f"- orig_col_name: {orig_col_name}")
    logger.info(f"- dest_col_name: {dest_col_name}")
    logger.info(f"- stop_col_name: {stop_col_name}")
    skim_dataset = inject.get_injectable("skim_dataset")
    odim = "omaz" if "omaz" in skim_dataset.dims else "otaz"
    ddim = "dmaz" if "dmaz" in skim_dataset.dims else "dtaz"
    if (
        orig_col_name is not None
        and dest_col_name is not None
        and stop_col_name is None
        and parking_col_name is None
    ):
        if timeframe == "timeless":
            return dict(
                skims=skim_dataset,
                relationships=(
                    f"df._orig_col_name -> skims.{odim}",
                    f"df._dest_col_name -> skims.{ddim}",
                ),
            )
        if timeframe == "timeless_directional":
            return dict(
                od_skims=skim_dataset,
                do_skims=skim_dataset,
                relationships=(
                    f"df._orig_col_name -> od_skims.{odim}",
                    f"df._dest_col_name -> od_skims.{ddim}",
                    f"df._dest_col_name -> do_skims.{odim}",
                    f"df._orig_col_name -> do_skims.{ddim}",
                ),
            )
        elif timeframe == "trip":
            return dict(
                odt_skims=skim_dataset,
                dot_skims=skim_dataset,
                od_skims=skim_dataset,
                relationships=(
                    f"df._orig_col_name -> odt_skims.{odim}",
                    f"df._dest_col_name -> odt_skims.{ddim}",
                    f"df.trip_period -> odt_skims.time_period",
                    f"df._dest_col_name -> dot_skims.{odim}",
                    f"df._orig_col_name -> dot_skims.{ddim}",
                    f"df.trip_period -> dot_skims.time_period",
                    f"df._orig_col_name -> od_skims.{odim}",
                    f"df._dest_col_name -> od_skims.{ddim}",
                ),
            )
        else:
            return dict(
                # TODO:SHARROW: organize dimensions.
                odt_skims=skim_dataset,
                dot_skims=skim_dataset,
                odr_skims=skim_dataset,
                dor_skims=skim_dataset,
                od_skims=skim_dataset,
                relationships=(
                    f"df._orig_col_name -> odt_skims.{odim}",
                    f"df._dest_col_name -> odt_skims.{ddim}",
                    f"df.out_period      @  odt_skims.time_period",
                    f"df._dest_col_name -> dot_skims.{odim}",
                    f"df._orig_col_name -> dot_skims.{ddim}",
                    f"df.in_period       @  dot_skims.time_period",
                    f"df._orig_col_name -> odr_skims.{odim}",
                    f"df._dest_col_name -> odr_skims.{ddim}",
                    f"df.in_period       @  odr_skims.time_period",
                    f"df._dest_col_name -> dor_skims.{odim}",
                    f"df._orig_col_name -> dor_skims.{ddim}",
                    f"df.out_period      @  dor_skims.time_period",
                    f"df._orig_col_name -> od_skims.{odim}",
                    f"df._dest_col_name -> od_skims.{ddim}",
                ),
            )
    elif stop_col_name is not None:  # trip_destination
        return dict(
            od_skims=skim_dataset,
            dp_skims=skim_dataset,
            odt_skims=skim_dataset,
            dot_skims=skim_dataset,
            dpt_skims=skim_dataset,
            pdt_skims=skim_dataset,
            relationships=(
                f"df._orig_col_name -> od_skims.{odim}",
                f"df._dest_col_name -> od_skims.{ddim}",
                f"df._dest_col_name -> dp_skims.{odim}",
                f"df._stop_col_name -> dp_skims.{ddim}",
                f"df._orig_col_name -> odt_skims.{odim}",
                f"df._dest_col_name -> odt_skims.{ddim}",
                f"df.trip_period     -> odt_skims.time_period",
                f"df._dest_col_name -> dot_skims.{odim}",
                f"df._orig_col_name -> dot_skims.{ddim}",
                f"df.trip_period     -> dot_skims.time_period",
                f"df._dest_col_name -> dpt_skims.{odim}",
                f"df._stop_col_name  -> dpt_skims.{ddim}",
                f"df.trip_period     -> dpt_skims.time_period",
                f"df._stop_col_name    -> pdt_skims.{odim}",
                f"df._dest_col_name -> pdt_skims.{ddim}",
                f"df.trip_period     -> pdt_skims.time_period",
            ),
        )
    elif parking_col_name is not None:  # parking location
        return dict(
            od_skims=skim_dataset,
            do_skims=skim_dataset,
            op_skims=skim_dataset,
            pd_skims=skim_dataset,
            odt_skims=skim_dataset,
            dot_skims=skim_dataset,
            opt_skims=skim_dataset,
            pdt_skims=skim_dataset,
            relationships=(
                f"df._orig_col_name -> od_skims.{odim}",
                f"df._dest_col_name -> od_skims.{ddim}",
                f"df._dest_col_name -> do_skims.{odim}",
                f"df._orig_col_name -> do_skims.{ddim}",
                f"df._orig_col_name -> op_skims.{odim}",
                f"df._park_col_name -> op_skims.{ddim}",
                f"df._park_col_name -> pd_skims.{odim}",
                f"df._dest_col_name -> pd_skims.{ddim}",
                f"df._orig_col_name -> odt_skims.{odim}",
                f"df._dest_col_name -> odt_skims.{ddim}",
                f"df.trip_period    -> odt_skims.time_period",
                f"df._dest_col_name -> dot_skims.{odim}",
                f"df._orig_col_name -> dot_skims.{ddim}",
                f"df.trip_period    -> dot_skims.time_period",
                f"df._orig_col_name -> opt_skims.{odim}",
                f"df._park_col_name -> opt_skims.{ddim}",
                f"df.trip_period    -> opt_skims.time_period",
                f"df._park_col_name -> pdt_skims.{odim}",
                f"df._dest_col_name -> pdt_skims.{ddim}",
                f"df.trip_period    -> pdt_skims.time_period",
            ),
        )
    else:
        return {}


def new_flow(
    spec,
    extra_vars,
    orig_col_name,
    dest_col_name,
    trace_label=None,
    timeframe="tour",
    choosers=None,
    stop_col_name=None,
    parking_col_name=None,
    size_term_mapping=None,
    interacts=None,
):

    with logtime(f"setting up flow {trace_label}"):
        if choosers is None:
            chooser_cols = []
        else:
            chooser_cols = list(choosers.columns)

        cache_dir = os.path.join(
            config.get_cache_dir(),
            "__sharrowcache__",
        )
        os.makedirs(cache_dir, exist_ok=True)
        logger.debug(f"flow.cache_dir: {cache_dir}")
        skims_mapping_ = skims_mapping(
            orig_col_name,
            dest_col_name,
            timeframe,
            stop_col_name,
            parking_col_name=parking_col_name,
        )
        if size_term_mapping is None:
            size_term_mapping = {}

        if interacts is None:
            if choosers is None:
                logger.info(f"empty flow on {trace_label}")
            else:
                logger.info(f"{len(choosers)} chooser rows on {trace_label}")
            flow_tree = sh.DataTree(df=[] if choosers is None else choosers)
            idx_name = choosers.index.name or "index"
            rename_dataset_cols = {
                idx_name: "chooserindex",
            }
            if orig_col_name is not None:
                rename_dataset_cols[orig_col_name] = "_orig_col_name"
            if dest_col_name is not None:
                rename_dataset_cols[dest_col_name] = "_dest_col_name"
            if stop_col_name is not None:
                rename_dataset_cols[stop_col_name] = "_stop_col_name"
            if parking_col_name is not None:
                rename_dataset_cols[parking_col_name] = "_park_col_name"

            def _apply_filter(_dataset, renames: dict):
                ds = _dataset.rename(renames).ensure_integer(renames.values())
                for _k, _v in renames.items():
                    ds[_k] = ds[_v]
                return ds

            from functools import partial

            flow_tree.replacement_filters[flow_tree.root_node_name] = partial(
                _apply_filter, renames=rename_dataset_cols
            )
            flow_tree.root_dataset = flow_tree.root_dataset  # apply the filter
        else:
            logger.info(
                f"{len(choosers)} chooser rows and {len(interacts)} interact rows on {trace_label}"
            )
            top = sh.dataset.from_named_objects(
                pd.RangeIndex(len(choosers), name="chooserindex"),
                pd.RangeIndex(len(interacts), name="interactindex"),
            )
            flow_tree = sh.DataTree(start=top)
            rename_dataset_cols = {
                orig_col_name: "_orig_col_name",
                dest_col_name: "_dest_col_name",
            }
            if stop_col_name is not None:
                rename_dataset_cols[stop_col_name] = "_stop_col_name"
            if parking_col_name is not None:
                rename_dataset_cols[parking_col_name] = "_park_col_name"
            choosers_ = (
                sh.dataset.construct(choosers)
                .rename_or_ignore(rename_dataset_cols)
                .ensure_integer(
                    [
                        "_orig_col_name",
                        "_dest_col_name",
                        "_stop_col_name",
                        "_park_col_name",
                    ]
                )
            )
            for _k, _v in rename_dataset_cols.items():
                if _v in choosers_:
                    choosers_[_k] = choosers_[_v]
            flow_tree.add_dataset(
                "df",
                choosers_,
                f"start.chooserindex -> df.{next(iter(choosers_.dims))}",
            )
            interacts_ = sh.dataset.construct(interacts).rename_or_ignore(
                rename_dataset_cols
            )
            flow_tree.add_dataset(
                "interact_table",
                interacts_,
                f"start.interactindex -> interact_table.{next(iter(interacts_.dims))}",
            )
            flow_tree.subspace_fallbacks["df"] = ["interact_table"]

        flow_tree.add_items(skims_mapping_)
        flow_tree.add_items(size_term_mapping)
        flow_tree.extra_vars = extra_vars

        # logger.info(f"initializing sharrow shared data {trace_label}")
        # pool = sh.SharedData(
        #     chooser_cols,
        #     **skims_mapping_,
        #     **size_term_mapping,
        #     extra_vars=extra_vars,
        #     alias_main="df",
        # )

        # - eval spec expressions
        if isinstance(spec.index, pd.MultiIndex):
            # spec MultiIndex with expression and label
            exprs = spec.index.get_level_values(SPEC_EXPRESSION_NAME)
            labels = spec.index.get_level_values(SPEC_LABEL_NAME)
        else:
            exprs = spec.index
            labels = exprs

        defs = {}
        for (expr, label) in zip(exprs, labels):
            if expr[0] == "@":
                if label == expr:
                    if expr[1:].isidentifier():
                        defs[expr[1:] + "_"] = expr[1:]
                    else:
                        defs[expr[1:]] = expr[1:]
                else:
                    defs[label] = expr[1:]
            elif expr[0] == "_" and "@" in expr:
                # - allow temps of form _od_DIST@od_skim['DIST']
                target = expr[: expr.index("@")]
                rhs = expr[expr.index("@") + 1 :]
                defs[target] = rhs
            else:
                if label == expr and expr.isidentifier():
                    defs[expr + "_"] = expr
                else:
                    defs[label] = expr

        readme = f"""
        activitysim version: {__version__}
        trace label: {trace_label}
        orig_col_name: {orig_col_name}
        dest_col_name: {dest_col_name}
        expressions:"""
        for (expr, label) in zip(exprs, labels):
            readme += f"\n            - {label}: {expr}"
        if extra_vars:
            readme += f"\n        extra_vars:"
            for i, v in extra_vars.items():
                readme += f"\n            - {i}: {v}"

        logger.info(f"setting up sharrow flow {trace_label}")
        return flow_tree.setup_flow(
            defs,
            cache_dir=cache_dir,
            readme=readme[1:],  # remove leading newline
            flow_library=_FLOWS,
            # extra_hash_data=(orig_col_name, dest_col_name),
            hashing_level=0,
        )


def size_terms_on_flow(locals_d):
    if "size_terms_array" in locals_d:
        # skim_dataset = inject.get_injectable('skim_dataset')
        dest_col_name = locals_d["od_skims"].dest_key
        a = sh.Dataset(
            {
                "arry": sh.DataArray(
                    locals_d["size_terms_array"],
                    dims=["stoptaz", "purpose_index"],
                    coords={
                        "stoptaz": np.arange(
                            locals_d["size_terms_array"].shape[0]
                        ),  # TODO: this assumes zero-based array of choices, is this always right?
                    },
                )
            }
        )
        # a = a.reindex(stoptaz=skim_dataset.coords['dtaz'].values) # TODO {ddim}?
        locals_d["size_array"] = dict(
            size_terms=a,
            relationships=(
                f"df._dest_col_name -> size_terms.stoptaz",
                f"df.purpose_index_num -> size_terms.purpose_index",
            ),
        )
    return locals_d


def apply_flow(
    spec, choosers, locals_d=None, trace_label=None, required=False, interacts=None
):
    if sh is None:
        return None, None
    if locals_d is None:
        locals_d = {}
    with logtime("apply_flow"):
        try:
            flow = get_flow(
                spec, locals_d, trace_label, choosers=choosers, interacts=interacts
            )
        except ValueError as err:
            if "unable to rewrite" in str(err):
                logger.error(f"error in apply_flow: {err!s}")
                if required:
                    raise
                return None, None
            else:
                raise
        with logtime("flow.load", trace_label or ""):
            try:
                flow_result = flow.dot(
                    coefficients=spec.values.astype(np.float32),
                    dtype=np.float32,
                    compile_watch=True,
                )
                # TODO: are there remaining internal arrays in dot that need to be
                #  passed out to be seen by the dynamic chunker before they are freed?
            except ValueError as err:
                if "could not convert" in str(err):
                    logger.error(f"error in apply_flow: {err!s}")
                    if required:
                        raise
                    return None, flow
                raise
            except Exception as err:
                logger.error(f"error in apply_flow: {err!s}")
                # index_keys = self.shared_data.meta_match_names_idx.keys()
                # logger.debug(f"Flow._get_indexes: {index_keys}")
                raise
            if flow.compiled_recently:
                tracing.timing_notes.add(f"compiled:{flow.name}")
            return flow_result, flow
