import contextlib
import glob
import logging
import os
import time
from datetime import timedelta
from functools import partial
from numbers import Number
from stat import ST_MTIME

import numpy as np
import pandas as pd

from .. import __version__
from ..core import tracing
from . import config, inject
from .simulate_consts import SPEC_EXPRESSION_NAME, SPEC_LABEL_NAME
from .timetable import (
    sharrow_tt_adjacent_window_after,
    sharrow_tt_adjacent_window_before,
    sharrow_tt_max_time_block_available,
    sharrow_tt_previous_tour_begins,
    sharrow_tt_previous_tour_ends,
    sharrow_tt_remaining_periods_available,
)

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


def get_flow(
    spec, local_d, trace_label=None, choosers=None, interacts=None, zone_layer=None
):
    extra_vars = only_simple(local_d)
    orig_col_name = local_d.get("orig_col_name", None)
    dest_col_name = local_d.get("dest_col_name", None)
    stop_col_name = None
    parking_col_name = None
    primary_origin_col_name = None
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
        if primary_origin_col_name is None and "dnt_skims" in local_d:
            primary_origin_col_name = local_d["dnt_skims"].dest_key
    local_d = size_terms_on_flow(local_d)
    size_term_mapping = local_d.get("size_array", {})
    if "tt" in local_d:
        aux_vars = local_d["tt"].export_for_numba()
    else:
        aux_vars = {}
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
        zone_layer=zone_layer,
        aux_vars=aux_vars,
        primary_origin_col_name=primary_origin_col_name,
    )
    flow.tree.aux_vars = aux_vars
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
    zone_layer=None,
    primary_origin_col_name=None,
):
    logger.info("loading skims_mapping")
    logger.info(f"- orig_col_name: {orig_col_name}")
    logger.info(f"- dest_col_name: {dest_col_name}")
    logger.info(f"- stop_col_name: {stop_col_name}")
    logger.info(f"- primary_origin_col_name: {primary_origin_col_name}")
    skim_dataset = inject.get_injectable("skim_dataset")
    if zone_layer == "maz" or zone_layer is None:
        odim = "omaz" if "omaz" in skim_dataset.dims else "otaz"
        ddim = "dmaz" if "dmaz" in skim_dataset.dims else "dtaz"
    elif zone_layer == "taz":
        odim = "otaz"
        ddim = "dtaz"
        if "omaz" in skim_dataset.dims:
            # strip out all MAZ-specific features of the skim_dataset
            dropdims = ["omaz", "dmaz"]
            skim_dataset = skim_dataset.drop_dims(dropdims, errors="ignore")
            for dd in dropdims:
                if f"dim_redirection_{dd}" in skim_dataset.attrs:
                    del skim_dataset.attrs[f"dim_redirection_{dd}"]
            for attr_name in list(skim_dataset.attrs):
                if attr_name.startswith("blend"):
                    del skim_dataset.attrs[attr_name]

    else:
        raise ValueError(f"unknown zone layer {zone_layer!r}")
    if zone_layer:
        logger.info(f"- zone_layer: {zone_layer}")
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
                    "df.trip_period -> odt_skims.time_period",
                    f"df._dest_col_name -> dot_skims.{odim}",
                    f"df._orig_col_name -> dot_skims.{ddim}",
                    "df.trip_period -> dot_skims.time_period",
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
                    "df.out_period      @  odt_skims.time_period",
                    f"df._dest_col_name -> dot_skims.{odim}",
                    f"df._orig_col_name -> dot_skims.{ddim}",
                    "df.in_period       @  dot_skims.time_period",
                    f"df._orig_col_name -> odr_skims.{odim}",
                    f"df._dest_col_name -> odr_skims.{ddim}",
                    "df.in_period       @  odr_skims.time_period",
                    f"df._dest_col_name -> dor_skims.{odim}",
                    f"df._orig_col_name -> dor_skims.{ddim}",
                    "df.out_period      @  dor_skims.time_period",
                    f"df._orig_col_name -> od_skims.{odim}",
                    f"df._dest_col_name -> od_skims.{ddim}",
                ),
            )
    elif stop_col_name is not None:  # trip_destination
        return dict(
            od_skims=skim_dataset,
            dp_skims=skim_dataset,
            op_skims=skim_dataset,
            odt_skims=skim_dataset,
            dot_skims=skim_dataset,
            dpt_skims=skim_dataset,
            pdt_skims=skim_dataset,
            opt_skims=skim_dataset,
            pot_skims=skim_dataset,
            ndt_skims=skim_dataset,
            dnt_skims=skim_dataset,
            relationships=(
                f"df._orig_col_name -> od_skims.{odim}",
                f"df._dest_col_name -> od_skims.{ddim}",
                f"df._dest_col_name -> dp_skims.{odim}",
                f"df._stop_col_name -> dp_skims.{ddim}",
                f"df._orig_col_name -> op_skims.{odim}",
                f"df._stop_col_name -> op_skims.{ddim}",
                f"df._orig_col_name -> odt_skims.{odim}",
                f"df._dest_col_name -> odt_skims.{ddim}",
                "df.trip_period     -> odt_skims.time_period",
                f"df._dest_col_name -> dot_skims.{odim}",
                f"df._orig_col_name -> dot_skims.{ddim}",
                "df.trip_period     -> dot_skims.time_period",
                f"df._dest_col_name -> dpt_skims.{odim}",
                f"df._stop_col_name -> dpt_skims.{ddim}",
                "df.trip_period     -> dpt_skims.time_period",
                f"df._stop_col_name -> pdt_skims.{odim}",
                f"df._dest_col_name -> pdt_skims.{ddim}",
                "df.trip_period     -> pdt_skims.time_period",
                f"df._orig_col_name -> opt_skims.{odim}",
                f"df._stop_col_name -> opt_skims.{ddim}",
                "df.trip_period     -> opt_skims.time_period",
                f"df._stop_col_name -> pot_skims.{odim}",
                f"df._orig_col_name -> pot_skims.{ddim}",
                "df.trip_period     -> pot_skims.time_period",
                f"df._primary_origin_col_name -> ndt_skims.{odim}",
                f"df._dest_col_name -> ndt_skims.{ddim}",
                "df.trip_period     -> ndt_skims.time_period",
                f"df._dest_col_name -> dnt_skims.{odim}",
                f"df._primary_origin_col_name -> dnt_skims.{ddim}",
                "df.trip_period     -> dnt_skims.time_period",
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
                "df.trip_period    -> odt_skims.time_period",
                f"df._dest_col_name -> dot_skims.{odim}",
                f"df._orig_col_name -> dot_skims.{ddim}",
                "df.trip_period    -> dot_skims.time_period",
                f"df._orig_col_name -> opt_skims.{odim}",
                f"df._park_col_name -> opt_skims.{ddim}",
                "df.trip_period    -> opt_skims.time_period",
                f"df._park_col_name -> pdt_skims.{odim}",
                f"df._dest_col_name -> pdt_skims.{ddim}",
                "df.trip_period    -> pdt_skims.time_period",
            ),
        )
    else:
        return {}  # flows without LOS characteristics are still valid


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
    zone_layer=None,
    aux_vars=None,
    primary_origin_col_name=None,
):
    """
    Setup a new sharrow flow.

    Parameters
    ----------
    spec : pandas.DataFrame
        The spec, as usual for ActivitySim. The index should either be a basic
        single-level index containing the expressions to be evaluated, or a
        MultiIndex with at least "Expression" and "Label" levels.
    extra_vars : Mapping
        Extra values that are available to expressions and which are written
        explicitly into compiled code (and cannot be changed later).
    orig_col_name : str
        The column from the choosers table that gives the origin zone index,
        used to attach values from skims.
    dest_col_name : str
        The column from the choosers table that gives the destination zone index,
        used to attach values from skims.
    trace_label : str
        A descriptive label
    timeframe : {"tour", "timeless", "timeless_directional", "trip"}, default "tour"
        A framework for how to treat the time and directionality of skims that
        will be attached.
    choosers : pandas.DataFrame
        Attributes of the choosers, possibly interacted with attributes of the
        alternatives.  Generally this flow can and will be re-used by swapping
        out the `choosers` for a new dataframe with the same columns and
        different rows.
    stop_col_name : str
        The column from the choosers table that gives the stop zone index in
        trip destination choice, used to attach values from skims.
    parking_col_name : str
        The column from the choosers table that gives the parking zone index,
        used to attach values from skims.
    size_term_mapping : Mapping
        Size term arrays.
    interacts : pd.DataFrame, optional
        An unmerged interaction dataset, giving attributes of the alternatives
        that are not conditional on the chooser.  Use this when the choice model
        has some variables that are conditional on the chooser (and included in
        the `choosers` dataframe, and some variables that are conditional on the
        alternative but not the chooser, and when every chooser has the same set
        of possible alternatives.
    zone_layer : {'taz', 'maz'}, default 'taz'
        Specify which zone layer of the skims is to be used.  You cannot use the
        'maz' zone layer in a one-zone model, but you can use the 'taz' layer in
        a two- or three-zone model (e.g. for destination pre-sampling).
    aux_vars : Mapping
        Extra values that are available to expressions and which are written
        only by reference into compiled code (and thus can be changed later).

    Returns
    -------
    sharrow.Flow
    """

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
            zone_layer=zone_layer,
            primary_origin_col_name=primary_origin_col_name,
        )
        if size_term_mapping is None:
            size_term_mapping = {}

        def _apply_filter(_dataset, renames: list):
            renames_keys = set(i for (i, j) in rename_dataset_cols)
            ds = _dataset.ensure_integer(renames_keys)
            for _k, _v in renames:
                if _k in ds:
                    ds[_v] = ds[_k]
            return ds

        if interacts is None:
            if choosers is None:
                logger.info(f"empty flow on {trace_label}")
            else:
                logger.info(f"{len(choosers)} chooser rows on {trace_label}")
            flow_tree = sh.DataTree(df=[] if choosers is None else choosers)
            idx_name = choosers.index.name or "index"
            rename_dataset_cols = [
                (idx_name, "chooserindex"),
            ]
            if orig_col_name is not None:
                rename_dataset_cols.append((orig_col_name, "_orig_col_name"))
            if dest_col_name is not None:
                rename_dataset_cols.append((dest_col_name, "_dest_col_name"))
            if stop_col_name is not None:
                rename_dataset_cols.append((stop_col_name, "_stop_col_name"))
            if parking_col_name is not None:
                rename_dataset_cols.append((parking_col_name, "_park_col_name"))
            if primary_origin_col_name is not None:
                rename_dataset_cols.append(
                    (primary_origin_col_name, "_primary_origin_col_name")
                )

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
            rename_dataset_cols = []
            if orig_col_name is not None:
                rename_dataset_cols.append((orig_col_name, "_orig_col_name"))
            if dest_col_name is not None:
                rename_dataset_cols.append((dest_col_name, "_dest_col_name"))
            if stop_col_name is not None:
                rename_dataset_cols.append((stop_col_name, "_stop_col_name"))
            if parking_col_name is not None:
                rename_dataset_cols.append((parking_col_name, "_park_col_name"))
            if primary_origin_col_name is not None:
                rename_dataset_cols.append(
                    (primary_origin_col_name, "_primary_origin_col_name")
                )

            choosers_ = sh.dataset.construct(choosers)
            choosers_ = _apply_filter(choosers_, rename_dataset_cols)
            flow_tree.add_dataset(
                "df",
                choosers_,
                f"start.chooserindex -> df.{next(iter(choosers_.dims))}",
            )
            interacts_ = sh.dataset.construct(interacts)
            interacts_ = _apply_filter(interacts_, rename_dataset_cols)
            flow_tree.add_dataset(
                "interact_table",
                interacts_,
                f"start.interactindex -> interact_table.{next(iter(interacts_.dims))}",
            )
            flow_tree.subspace_fallbacks["df"] = ["interact_table"]

        flow_tree.add_items(skims_mapping_)
        flow_tree.add_items(size_term_mapping)
        flow_tree.extra_vars = extra_vars
        flow_tree.extra_funcs = (
            sharrow_tt_remaining_periods_available,
            sharrow_tt_previous_tour_begins,
            sharrow_tt_previous_tour_ends,
            sharrow_tt_adjacent_window_after,
            sharrow_tt_adjacent_window_before,
            sharrow_tt_max_time_block_available,
        )
        flow_tree.aux_vars = aux_vars

        # - eval spec expressions
        if isinstance(spec.index, pd.MultiIndex):
            # spec MultiIndex with expression and label
            exprs = spec.index.get_level_values(SPEC_EXPRESSION_NAME)
            labels = spec.index.get_level_values(SPEC_LABEL_NAME)
        else:
            exprs = spec.index
            labels = exprs

        defs = {}
        # duplicate labels cause problems for sharrow, so we need to dedupe
        existing_labels = set()
        for (expr, label) in zip(exprs, labels):
            while label in existing_labels:
                label = label + "_"
            existing_labels.add(label)
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
            readme += "\n        extra_vars:"
            for i, v in extra_vars.items():
                readme += f"\n            - {i}: {v}"

        logger.info(f"setting up sharrow flow {trace_label}")
        extra_hash_data = ()
        if zone_layer:
            extra_hash_data += (zone_layer,)
        return flow_tree.setup_flow(
            defs,
            cache_dir=cache_dir,
            readme=readme[1:],  # remove leading newline
            flow_library=_FLOWS,
            extra_hash_data=extra_hash_data,
            hashing_level=0,
            boundscheck=False,
        )


def size_terms_on_flow(locals_d):
    """
    Create size terms to attach to a DataTree based on destination and purpose.

    Parameters
    ----------
    locals_d : Mapping[str,Any]
        The context for the flow.  If it does not contain "size_terms_array"
        this function does nothing. Otherwise, the instructions for adding
        the size terms to the DataTree are created in a "size_array" variable
        in the same context space.

    Returns
    -------
    locals_d
    """
    if "size_terms_array" in locals_d:
        a = sh.Dataset(
            {
                "sizearray": sh.DataArray(
                    locals_d["size_terms_array"],
                    dims=["zoneid", "purpose_index"],
                    coords={
                        "zoneid": np.arange(locals_d["size_terms_array"].shape[0]),
                    },
                )
            }
        )
        locals_d["size_array"] = dict(
            size_terms=a,
            relationships=(
                "df._dest_col_name -> size_terms.zoneid",
                "df.purpose_index_num -> size_terms.purpose_index",
            ),
        )
    return locals_d


def apply_flow(
    spec,
    choosers,
    locals_d=None,
    trace_label=None,
    required=False,
    interacts=None,
    zone_layer=None,
):
    """
    Apply a sharrow flow.

    Parameters
    ----------
    spec : pd.DataFrame
    choosers : pd.DataFrame
    locals_d : Mapping[str,Any], optional
        A namespace of local variables to be made available with the
        expressions in `spec`.
    trace_label : str, optional
        A descriptive label used in logging and naming trace files.
    required : bool, default False
        Require the spec to be compile-able. If set to true, a problem will
        the flow will be raised as an error, instead of allowing this function
        to return with no result (and activitysim will then fall back to the
        legacy eval system).
    interacts : pd.DataFrame, optional
        An unmerged interaction dataset, giving attributes of the alternatives
        that are not conditional on the chooser.  Use this when the choice model
        has some variables that are conditional on the chooser (and included in
        the `choosers` dataframe, and some variables that are conditional on the
        alternative but not the chooser, and when every chooser has the same set
        of possible alternatives.
    zone_layer : {'taz', 'maz'}, default 'taz'
        Specify which zone layer of the skims is to be used.  You cannot use the
        'maz' zone layer in a one-zone model, but you can use the 'taz' layer in
        a two- or three-zone model (e.g. for destination pre-sampling).

    Returns
    -------
    flow_result : ndarray
        The computed dot-product of the utility function and the coefficients.
    flow : sharrow.Flow
        The flow object itself.  In typical application you probably don't need
        it ever again, but having a reference to it available later can be useful
        in debugging and tracing.  Flows are cached and reused anyway, so it is
        generally not important to delete this at any point to free resources.
    """
    if sh is None:
        return None, None
    if locals_d is None:
        locals_d = {}
    with logtime("apply_flow"):
        try:
            flow = get_flow(
                spec,
                locals_d,
                trace_label,
                choosers=choosers,
                interacts=interacts,
                zone_layer=zone_layer,
            )
        except ValueError as err:
            if "unable to rewrite" in str(err):
                # There has been an error in preparing this flow.
                # If in `require` mode, we report the error and keep it as an error
                # Otherwise, we report the error but then swallow it and return
                # a None result, allowing ActivitySim to fall back to legacy
                # operating mode for this utility function.
                logger.error(f"error in apply_flow: {err!s}")
                if required:
                    raise
                return None, None
            else:
                raise
        with logtime(f"{flow.name}.load", trace_label or ""):
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
                    # There has been an error in compiling this flow.
                    # If in `require` mode, we report the error and keep it as an error
                    # Otherwise, we report the error but then swallow it and return
                    # a None result, allowing ActivitySim to fall back to legacy
                    # operating mode for this utility function.
                    logger.error(f"error in apply_flow: {err!s}")
                    if required:
                        raise
                    return None, flow
                raise
            except Exception as err:
                logger.error(f"error in apply_flow: {err!s}")
                raise
            if flow.compiled_recently:
                # When compile activity is detected, we make a note in the timing log,
                # which can help explain when a component is unexpectedly slow.
                # Detecting compilation activity when in production mode is a bug
                # that should be investigated.
                tracing.timing_notes.add(f"compiled:{flow.name}")
            return flow_result, flow
