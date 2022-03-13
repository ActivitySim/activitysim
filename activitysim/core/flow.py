import os
import hashlib
import re
import glob
import numpy as np
import pandas as pd
import time
import logging
import openmatrix
import contextlib
from orca import orca
from numbers import Number
from typing import Mapping
from datetime import timedelta
from stat import ST_MTIME

try:
    import sharrow as sh
except ModuleNotFoundError:
    sh = None

from .simulate_consts import SPEC_EXPRESSION_NAME, SPEC_LABEL_NAME
from . import inject, config
from .. import __version__

logger = logging.getLogger(__name__)

_FLOWS = {}


@contextlib.contextmanager
def logtime(tag, tag2=''):
    logger.info(f"begin {tag} {tag2}")
    t0 = time.time()
    try:
        yield
    except:
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
                logger.info("elapsed time {0} {1} {2}".format(
                    tag, timedelta(seconds=elapsed), suffix,
                ))
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
    def aggregate_summary(cls, logger, heading="Aggregate Flow Timing Summary", level=20):
        msg = f"{heading}\n"
        msgs = []
        for tag, elapsed in cls.aggregate_timing.items():
            msgs.append("   - {0:48s} {1}".format(tag, timedelta(seconds=elapsed)))
        msg += "\n".join(msgs)
        logger.log(level=level, msg=msg)


def only_numbers(x, exclude_keys=()):
    """
    All the values in a dict that are plain numbers.
    """
    y = {}
    for k, v in x.items():
        if k not in exclude_keys:
            if isinstance(v, Number):
                y[k] = v
            # elif isinstance(v, np.ndarray):
            #     y[k] = v
    return y


def get_flow(spec, local_d, trace_label=None, choosers=None, interacts=None):
    global _FLOWS
    extra_vars = only_numbers(local_d)
    orig_col_name = local_d.get('orig_col_name', None)
    dest_col_name = local_d.get('dest_col_name', None)
    stop_col_name = None
    timeframe = local_d.get('timeframe', 'tour')
    if timeframe == 'trip':
        orig_col_name = local_d.get('ORIGIN', orig_col_name)
        dest_col_name = local_d.get('DESTINATION', dest_col_name)
        if orig_col_name is None and 'od_skims' in local_d:
            orig_col_name = local_d['od_skims'].orig_key
        if dest_col_name is None and 'od_skims' in local_d:
            dest_col_name = local_d['od_skims'].dest_key
        if stop_col_name is None and 'dp_skims' in local_d:
            stop_col_name = local_d['dp_skims'].dest_key
    local_d = size_terms_on_flow(local_d)
    size_term_mapping = local_d.get('size_array', {})
    flow = new_flow(
        spec, extra_vars,
        orig_col_name,
        dest_col_name,
        trace_label,
        timeframe=timeframe,
        choosers=choosers,
        stop_col_name=stop_col_name,
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
    skim_tag = 'taz'
    network_los_preload = inject.get_injectable('network_los_preload', None)
    if network_los_preload is None:
        raise ValueError("missing network_los_preload")

    # find which OMX files are to be used.
    omx_file_paths = config.expand_input_file_list(
        network_los_preload.omx_file_names(skim_tag),
    )
    zarr_file = config.data_file_path(
        network_los_preload.zarr_file_name(skim_tag),
        mandatory=False,
        allow_glob=False,
    )
    max_float_precision = network_los_preload.skim_max_float_precision(skim_tag)

    skim_digital_encoding = network_los_preload.skim_digital_encoding(skim_tag)

    # The backing can be plain shared_memory, or a memmap
    backing = network_los_preload.skim_backing_store(skim_tag)
    if backing == "memmap":
        # if memmap is given without a path, create a cache file
        mmap_file = os.path.join(config.get_cache_dir(), f"sharrow_dataset_{skim_tag}.mmap")
        backing = f"memmap:{mmap_file}"

    with logtime("loading skims as dataset"):

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
            time_periods_ = network_los_preload.los_settings['skim_time_periods']['labels']
            # deduplicate time period names
            time_periods = []
            for t in time_periods_:
                if t not in time_periods:
                    time_periods.append(t)
            if zarr_file:
                logger.info(f"looking for zarr skims at {zarr_file}")
            if zarr_file and os.path.exists(zarr_file):
                # load skims from zarr.zip
                logger.info(f"found zarr skims, loading them")
                d = sh.dataset.from_zarr(zarr_file).max_float_precision(max_float_precision)
            else:
                d = sh.dataset.from_omx_3d(
                    [openmatrix.open_file(f) for f in omx_file_paths],
                    time_periods=time_periods,
                    max_float_precision=max_float_precision,
                )
                if zarr_file:
                    logger.info(f"writing zarr skims to {zarr_file}")
                    # save skims to zarr
                    try:
                        d.to_zarr(zarr_file)
                    except ModuleNotFoundError:
                        logger.warning("the 'zarr' package is not installed")
            logger.info(f"scanning for unused skims")
            tokens = set(d.variables.keys()) - set(d.coords.keys())
            unused_tokens = scan_for_unused_names(tokens)
            if unused_tokens:
                logger.info(f"dropping unused skims: {unused_tokens}")
                d = d.drop_vars(unused_tokens)
            else:
                logger.info(f"no unused skims found")
            # apply digital encoding
            if skim_digital_encoding:
                for encoding in skim_digital_encoding:
                    regex = encoding.pop('regex', None)
                    if regex:
                        if 'name' in encoding:
                            raise ValueError("cannot give both name and regex for digital_encoding")
                        for k in d.variables:
                            if re.match(regex, k):
                                d = d.set_digital_encoding(k, **encoding)
                    else:
                        d = d.set_digital_encoding(**encoding)

        # check alignment of TAZs that it matches land_use table
        logger.info(f"checking skims alignment with land_use")
        land_use = inject.get_table('land_use')
        try:
            land_use_zone_id = land_use[f'_original_{land_use.index.name}']
        except KeyError:
            land_use_zone_id = land_use.index

        if d['otaz'].attrs.get('preprocessed') != 'zero-based-contiguous':
            try:
                np.testing.assert_array_equal(land_use_zone_id, d.otaz)
            except AssertionError as err:
                logger.info(f"otaz realignment required\n{err}")
                d = d.reindex(otaz=land_use_zone_id)
            else:
                logger.info(f"otaz alignment ok")
            d['otaz'] = land_use.index.to_numpy()
            d['otaz'].attrs['preprocessed'] = ('zero-based-contiguous')
        else:
            np.testing.assert_array_equal(land_use.index, d.otaz)

        if d['dtaz'].attrs.get('preprocessed') != 'zero-based-contiguous':
            try:
                np.testing.assert_array_equal(land_use_zone_id, d.dtaz)
            except AssertionError as err:
                logger.info(f"dtaz realignment required\n{err}")
                d = d.reindex(dtaz=land_use_zone_id)
            else:
                logger.info(f"dtaz alignment ok")
            d['dtaz'] = land_use.index.to_numpy()
            d['dtaz'].attrs['preprocessed'] = ('zero-based-contiguous')
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
    configs_dir_list = inject.get_injectable('configs_dir')
    configs_dir_list = [configs_dir_list] if isinstance(configs_dir_list, str) else configs_dir_list
    assert isinstance(configs_dir_list, list)

    for directory in configs_dir_list:
        logger.debug(f"scanning for unused skims in {directory}")
        filenames = glob.glob(os.path.join(directory, "*.csv"))
        for filename in filenames:
            with open(filename, 'rt') as f:
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


def skims_mapping(orig_col_name, dest_col_name, timeframe='tour', stop_col_name=None):
    logger.info(f"loading skims_mapping")
    logger.info(f"- orig_col_name: {orig_col_name}")
    logger.info(f"- dest_col_name: {dest_col_name}")
    logger.info(f"- stop_col_name: {stop_col_name}")
    skim_dataset = inject.get_injectable('skim_dataset')
    if orig_col_name is not None and dest_col_name is not None and stop_col_name is None:
        if timeframe == 'timeless':
            return dict(
                skims=skim_dataset,
                relationships=(
                    f"df._orig_col_name -> skims.otaz",
                    f"df._dest_col_name -> skims.dtaz",
                ),
            )
        elif timeframe == 'trip':
            return dict(
                odt_skims=skim_dataset.rename_dims_and_coords({'otaz': 'ptaz', 'dtaz': 'ataz'}),
                dot_skims=skim_dataset.rename_dims_and_coords({'otaz': 'ataz', 'dtaz': 'ptaz'}),
                od_skims=skim_dataset.drop_dims('time_period').rename_dims_and_coords({'otaz': 'ptaz', 'dtaz': 'ataz'}),
                relationships=(
                    f"df._orig_col_name -> odt_skims.ptaz",
                    f"df._dest_col_name -> odt_skims.ataz",
                    f"df.trip_period -> odt_skims.time_period",
                    f"df._dest_col_name -> dot_skims.ataz",
                    f"df._orig_col_name -> dot_skims.ptaz",
                    f"df.trip_period -> dot_skims.time_period",
                    f"df._orig_col_name -> od_skims.ptaz",
                    f"df._dest_col_name -> od_skims.ataz",
                ),
            )
        else:
            return dict(
                odt_skims=skim_dataset.rename_dims_and_coords({'otaz': 'ptaz', 'dtaz': 'ataz', 'time_period': 'out_period'}),
                dot_skims=skim_dataset.rename_dims_and_coords({'otaz': 'ataz', 'dtaz': 'ptaz', 'time_period': 'in_period'}),
                odr_skims=skim_dataset.rename_dims_and_coords({'otaz': 'ptaz', 'dtaz': 'ataz', 'time_period': 'in_period'}),
                dor_skims=skim_dataset.rename_dims_and_coords({'otaz': 'ataz', 'dtaz': 'ptaz', 'time_period': 'out_period'}),
                od_skims=skim_dataset.drop_dims('time_period').rename_dims_and_coords({'otaz': 'ptaz', 'dtaz': 'ataz'}),
                relationships=(
                    f"df._orig_col_name -> odt_skims.ptaz",
                    f"df._dest_col_name -> odt_skims.ataz",
                    f"df.out_period      @  odt_skims.out_period",
                    f"df._dest_col_name -> dot_skims.ataz",
                    f"df._orig_col_name -> dot_skims.ptaz",
                    f"df.in_period       @  dot_skims.in_period",
                    f"df._orig_col_name -> odr_skims.ptaz",
                    f"df._dest_col_name -> odr_skims.ataz",
                    f"df.in_period       @  odr_skims.in_period",
                    f"df._dest_col_name -> dor_skims.ataz",
                    f"df._orig_col_name -> dor_skims.ptaz",
                    f"df.out_period      @  dor_skims.out_period",
                    f"df._orig_col_name -> od_skims.ptaz",
                    f"df._dest_col_name -> od_skims.ataz",
                ),
            )
    elif stop_col_name is not None: # trip_destination
        return dict(
            od_skims=skim_dataset.drop_dims('time_period').rename_dims_and_coords({'otaz': 'ptaz', 'dtaz': 'ataz'}),
            dp_skims=skim_dataset.drop_dims('time_period').rename_dims_and_coords({'otaz': 'ataz', 'dtaz': 'staz'}),
            odt_skims=skim_dataset.rename_dims_and_coords({'otaz': 'ptaz', 'dtaz': 'ataz'}),
            dot_skims=skim_dataset.rename_dims_and_coords({'otaz': 'ataz', 'dtaz': 'ptaz'}),
            dpt_skims=skim_dataset.rename_dims_and_coords({'otaz': 'ataz', 'dtaz': 'staz'}),
            pdt_skims=skim_dataset.rename_dims_and_coords({'otaz': 'staz', 'dtaz': 'ataz'}),
            relationships=(
                f"df._orig_col_name -> od_skims.ptaz",
                f"df._dest_col_name -> od_skims.ataz",

                f"df._dest_col_name -> dp_skims.ataz",
                f"df._stop_col_name    -> dp_skims.staz",

                f"df._orig_col_name -> odt_skims.ptaz",
                f"df._dest_col_name -> odt_skims.ataz",
                f"df.trip_period     -> odt_skims.time_period",

                f"df._dest_col_name -> dot_skims.ataz",
                f"df._orig_col_name -> dot_skims.ptaz",
                f"df.trip_period     -> dot_skims.time_period",

                f"df._dest_col_name -> dpt_skims.ataz",
                f"df._stop_col_name    -> dpt_skims.staz",
                f"df.trip_period     -> dpt_skims.time_period",

                f"df._stop_col_name    -> pdt_skims.staz",
                f"df._dest_col_name -> pdt_skims.ataz",
                f"df.trip_period     -> pdt_skims.time_period",
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
        timeframe='tour',
        choosers=None,
        stop_col_name=None,
        size_term_mapping=None,
        interacts=None
):

    with logtime(f"setting up flow {trace_label}"):
        if choosers is None:
            chooser_cols = []
        else:
            chooser_cols = list(choosers.columns)

        cache_dir = os.path.join(
            orca.get_injectable('output_dir'),
            "cache",
            "__sharrowcache__",
        )
        os.makedirs(cache_dir, exist_ok=True)
        logger.debug(f"flow.cache_dir: {cache_dir}")
        skims_mapping_ = skims_mapping(orig_col_name, dest_col_name, timeframe, stop_col_name)
        if size_term_mapping is None:
            size_term_mapping = {}

        if interacts is None:
            flow_tree = sh.DataTree(df=[] if choosers is None else choosers)
            idx_name = choosers.index.name or 'index'
            rename_dataset_cols = {
                idx_name: 'chooserindex',
            }
            if orig_col_name is not None:
                rename_dataset_cols[orig_col_name] = '_orig_col_name'
            if dest_col_name is not None:
                rename_dataset_cols[dest_col_name] = '_dest_col_name'
            if stop_col_name is not None:
                rename_dataset_cols[stop_col_name] = '_stop_col_name'
            ds = flow_tree.root_dataset.rename(
                rename_dataset_cols
            ).ensure_integer(
                ['_orig_col_name', '_dest_col_name', '_stop_col_name']
            )
            # copy back the names of the renamed dims so they can be used in spec files.
            # note this doesn't copy the *data* just makes another named reference to the
            # same data.
            for _k, _v in rename_dataset_cols.items():
                ds[_k] = ds[_v]
            flow_tree.root_dataset = ds
        else:
            top = sh.dataset.from_named_objects(
                pd.RangeIndex(len(choosers), name="chooserindex"),
                pd.RangeIndex(len(interacts), name="interactindex"),
            )
            flow_tree = sh.DataTree(start=top)
            rename_dataset_cols = {
                orig_col_name: '_orig_col_name',
                dest_col_name: '_dest_col_name',
            }
            if stop_col_name is not None:
                rename_dataset_cols[stop_col_name] = '_stop_col_name'
            choosers_ = sh.dataset.construct(
                choosers
            ).rename_or_ignore(
                rename_dataset_cols
            ).ensure_integer(
                ['_orig_col_name', '_dest_col_name', '_stop_col_name']
            )
            for _k, _v in rename_dataset_cols.items():
                if _v in choosers_:
                    choosers_[_k] = choosers_[_v]
            flow_tree.add_dataset(
                'df',
                choosers_,
                f"start.chooserindex -> df.{next(iter(choosers_.dims))}"
            )
            interacts_ = sh.dataset.construct(interacts).rename_or_ignore(rename_dataset_cols)
            flow_tree.add_dataset(
                'interact_table',
                interacts_,
                f"start.interactindex -> interact_table.{next(iter(interacts_.dims))}"
            )

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
                        defs[expr[1:]+"_"] = expr[1:]
                    else:
                        defs[expr[1:]] = expr[1:]
                else:
                    defs[label] = expr[1:]
            elif expr[0] == "_" and "@" in expr:
                # - allow temps of form _od_DIST@od_skim['DIST']
                target = expr[:expr.index('@')]
                rhs = expr[expr.index('@') + 1:]
                defs[target] = rhs
            else:
                if label == expr and expr.isidentifier():
                    defs[expr+"_"] = expr
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
            readme=readme[1:], # remove leading newline
            flow_library=_FLOWS,
            # extra_hash_data=(orig_col_name, dest_col_name),
            hashing_level=0,
        )


def size_terms_on_flow(locals_d):
    if 'size_terms_array' in locals_d:
        skim_dataset = inject.get_injectable('skim_dataset')
        dest_col_name = locals_d['od_skims'].dest_key
        a = sh.Dataset({'arry': sh.DataArray(
            locals_d['size_terms_array'],
            dims=['stoptaz', 'purpose_index'],
            coords={
                'stoptaz': np.asarray(inject.get_table("land_use").to_frame().index),
            }
        )})
        a = a.reindex(stoptaz=skim_dataset.coords['dtaz'].values)
        locals_d['size_array'] = dict(
            size_terms=a,
            relationships=(
                f"df._dest_col_name -> size_terms.stoptaz",
                f"df.purpose_index_num -> size_terms.purpose_index",
            ),
        )
    return locals_d

def apply_flow(spec, choosers, locals_d=None, trace_label=None, required=False, interacts=None):
    if sh is None:
        return None, None
    if locals_d is None:
        locals_d = {}
    with logtime("apply_flow"):
        try:
            flow = get_flow(spec, locals_d, trace_label, choosers=choosers, interacts=interacts)
        except ValueError as err:
            if "unable to rewrite" in str(err):
                logger.error(f"error in apply_flow: {err!s}")
                if required:
                    raise
                return None, None
            else:
                raise
        with logtime("flow.load", trace_label or ''):
            try:
                flow_result = flow.dot(
                    coefficients=spec.values.astype(np.float32),
                    dtype=np.float32,
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
            return flow_result, flow