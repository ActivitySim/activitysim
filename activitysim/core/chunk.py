# ActivitySim
# See full license in LICENSE.txt.

import datetime
import glob
import logging
import math
import multiprocessing
import os
import threading
from contextlib import contextmanager

import numpy as np
import pandas as pd

from . import config, mem, tracing, util
from .util import GB

logger = logging.getLogger(__name__)

#
# CHUNK_METHODS and METRICS
#

RSS = "rss"
USS = "uss"
BYTES = "bytes"
HYBRID_RSS = "hybrid_rss"
HYBRID_USS = "hybrid_uss"

METRICS = [RSS, USS, BYTES]
CHUNK_METHODS = [RSS, USS, BYTES, HYBRID_RSS, HYBRID_USS]

USS_CHUNK_METHODS = [USS, HYBRID_USS, BYTES]
DEFAULT_CHUNK_METHOD = HYBRID_USS

"""

The chunk_cache table is a record of the memory usage and observed row_size required for chunking the various models.
The row size differs depending on whether memory usage is calculated by rss, uss, or explicitly allocated bytes.
We record all three during training so the mode can be changed without necessitating retraining.

tag,                               num_rows, rss,   uss,   bytes,    uss_row_size, hybrid_uss_row_size, bytes_row_size
atwork_subtour_frequency.simple,   3498,     86016, 81920, 811536,   24,           232,                 232
atwork_subtour_mode_choice.simple, 704,      20480, 20480, 1796608,  30,           2552,                2552
atwork_subtour_scheduling.tour_1,  701,      24576, 24576, 45294082, 36,           64614,               64614
atwork_subtour_scheduling.tour_n,  3,        20480, 20480, 97734,    6827,         32578,               32578
auto_ownership_simulate.simulate,  5000,     77824, 24576, 1400000,  5,            280,                 280

MODE_RETRAIN
    rebuild chunk_cache table and save/replace in output/cache/chunk_cache.csv
    preforms a complete rebuild of chunk_cache table by doing adaptive chunking starting with based on default initial
    settings (DEFAULT_INITIAL_ROWS_PER_CHUNK) and observing rss, uss, and allocated bytes to compute rows_size.
    This will run somewhat slower than the other modes because of overhead of small first chunk, and possible
    instability in the second chunk due to inaccuracies caused by small initial chunk_size sample

MODE_ADAPTIVE
    Use the existing chunk_cache to determine the sizing for the first chunk for each model, but also use the
    observed row_size to adjust the estimated row_size for subsequent chunks. At the end of hte run, writes the
    updated chunk_cache to the output directory, but doesn't overwrite the 'official' cache file. If the user wishes
    they can replace the chunk_cache with the updated versions but this is not done automatically as it is not clear
    this would be the desired behavior. (Might become clearer over time as this is exercised further.)

MODE_PRODUCTION
    Since overhead changes we don't necessarily want the same number of rows per chunk every time
    but we do use the row_size from cache which we trust is stable
    (the whole point of MODE_PRODUCTION is to avoid the cost of observing overhead)
    which is stored in self.initial_row_size because initial_rows_per_chunk used it for the first chunk

MODE_CHUNKLESS
    Do not do chunking, and also do not check or log memory usage, so ActivitySim can focus on performance
    assuming there is abundant RAM.
"""

MODE_RETRAIN = "training"
MODE_ADAPTIVE = "adaptive"
MODE_PRODUCTION = "production"
MODE_CHUNKLESS = "disabled"
TRAINING_MODES = [MODE_RETRAIN, MODE_ADAPTIVE, MODE_PRODUCTION, MODE_CHUNKLESS]

#
# low level
#
ENABLE_MEMORY_MONITOR = True
MEM_MONITOR_TICK = 1  # in seconds

LOG_SUBCHUNK_HISTORY = False  # only useful for debugging
WRITE_SUBCHUNK_HISTORY = False  # only useful for debugging

DEFAULT_INITIAL_ROWS_PER_CHUNK = (
    100  # fallback for default_initial_rows_per_chunk setting
)


#
# cache and history files
#

CACHE_FILE_NAME = "chunk_cache.csv"
LOG_FILE_NAME = "chunk_history.csv"
OMNIBUS_LOG_FILE_NAME = f"omnibus_{LOG_FILE_NAME}"

C_CHUNK_TAG = "tag"
C_DEPTH = "depth"
C_NUM_ROWS = "num_rows"
C_TIME = "time"

# columns to write to LOG_FILE
CUM_OVERHEAD_COLUMNS = [f"cum_overhead_{m}" for m in METRICS]
CHUNK_HISTORY_COLUMNS = (
    [C_TIME, C_CHUNK_TAG]
    + CUM_OVERHEAD_COLUMNS
    + [C_NUM_ROWS, "row_size", "chunk_size", C_DEPTH, "process", "chunk"]
)

CHUNK_CACHE_COLUMNS = [C_CHUNK_TAG, C_NUM_ROWS] + METRICS

#
# globals
#

SETTINGS = {}
CHUNK_LEDGERS = []
CHUNK_SIZERS = []

ledger_lock = threading.Lock()


def chunk_method():
    method = SETTINGS.get("chunk_method")
    if method is None:
        method = SETTINGS.setdefault(
            "chunk_method", config.setting("chunk_method", DEFAULT_CHUNK_METHOD)
        )
        assert (
            method in CHUNK_METHODS
        ), f"chunk_method setting '{method}' not recognized. Should be one of: {CHUNK_METHODS}"
    return method


def chunk_metric():
    return SETTINGS.setdefault(
        "chunk_metric", USS if chunk_method() in USS_CHUNK_METHODS else "rss"
    )


def chunk_training_mode():
    training_mode = SETTINGS.setdefault(
        "chunk_training_mode", config.setting("chunk_training_mode", MODE_ADAPTIVE)
    )
    if not training_mode:
        training_mode = MODE_CHUNKLESS
    assert (
        training_mode in TRAINING_MODES
    ), f"chunk_training_mode '{training_mode}' not one of: {TRAINING_MODES}"
    return training_mode


def chunk_logging():
    return len(CHUNK_LEDGERS) > 0


def default_initial_rows_per_chunk():
    return SETTINGS.setdefault(
        "default_initial_rows_per_chunk",
        config.setting(
            "default_initial_rows_per_chunk", DEFAULT_INITIAL_ROWS_PER_CHUNK
        ),
    )


def min_available_chunk_ratio():
    return SETTINGS.setdefault(
        "min_available_chunk_ratio", config.setting("min_available_chunk_ratio", 0)
    )


def keep_chunk_logs():
    # if we are overwriting MEM_LOG_FILE then presumably we want to delete any subprocess files
    default = LOG_FILE_NAME == OMNIBUS_LOG_FILE_NAME

    return SETTINGS.setdefault(
        "keep_chunk_logs", config.setting("keep_chunk_logs", default)
    )


def trace_label_for_chunk(trace_label, chunk_size, i):
    # add chunk_num to trace_label
    # if chunk_size > 0:
    #     trace_label = tracing.extend_trace_label(trace_label, f'chunk_{i}')
    return trace_label


def get_base_chunk_size():
    assert len(CHUNK_SIZERS) > 0
    return CHUNK_SIZERS[0].chunk_size


def overhead_for_chunk_method(overhead, method=None):
    """

    return appropriate overhead for row_size calculation based on current chunk_method

    * by ChunkSizer.adaptive_rows_per_chunk to determine observed_row_size based on cum_overhead and cum_rows
    * by ChunkSizer.initial_rows_per_chunk to determine initial row_size using cached_history and current chunk_method
    * by consolidate_logs to add informational row_size column to cache file based on chunk_method for training run

    Parameters
    ----------
    overhead: dict keyed by metric or DataFrame with columns

    Returns
    -------
        chunk_method overhead (possibly hybrid, depending on chunk_method)
    """

    def hybrid(xss, bytes):

        # this avoids pessimistic underchunking on second chunk without pre-existing cache
        # but it tends to overshoot on a trained runs
        # hybrid_overhead =  np.maximum(bytes, (xss + bytes) / 2)

        # this approach avoids problems with pure uss, especially with small chunk sizes (e.g. initial training chunks)
        # as numpy may recycle cached blocks and show no increase in uss even though data was allocated and logged
        hybrid_overhead = np.maximum(bytes, xss)

        return hybrid_overhead

    method = method or chunk_method()

    if method == HYBRID_RSS:
        oh = hybrid(overhead[RSS], overhead[BYTES])
    elif method == HYBRID_USS:
        oh = hybrid(overhead[USS], overhead[BYTES])
    else:
        # otherwise method name is same as metric name
        oh = overhead[method]

    return oh


def consolidate_logs():

    glob_file_name = config.log_file_path(f"*{LOG_FILE_NAME}", prefix=False)
    glob_files = glob.glob(glob_file_name)

    if not glob_files:
        return

    assert chunk_training_mode() not in (MODE_PRODUCTION, MODE_CHUNKLESS), (
        f"shouldn't be any chunk log files when chunk_training_mode"
        f" is {MODE_PRODUCTION} or {MODE_CHUNKLESS}"
    )

    #
    # OMNIBUS_LOG_FILE
    #

    logger.debug(f"chunk.consolidate_logs reading glob {glob_file_name}")
    omnibus_df = pd.concat((pd.read_csv(f, comment="#") for f in glob_files))

    omnibus_df = omnibus_df.sort_values(by=C_TIME)

    # shouldn't have different depths for the same chunk_tag
    multi_depth_chunk_tag = omnibus_df[[C_CHUNK_TAG, C_DEPTH]]
    multi_depth_chunk_tag = multi_depth_chunk_tag[~multi_depth_chunk_tag.duplicated()][
        [C_CHUNK_TAG]
    ]
    multi_depth_chunk_tag = multi_depth_chunk_tag[
        multi_depth_chunk_tag[C_CHUNK_TAG].duplicated()
    ]
    assert (
        len(multi_depth_chunk_tag) == 0
    ), f"consolidate_logs multi_depth_chunk_tags \n{multi_depth_chunk_tag.values}"

    if not keep_chunk_logs():
        util.delete_files(glob_files, "chunk.consolidate_logs")

    log_output_path = config.log_file_path(OMNIBUS_LOG_FILE_NAME, prefix=False)
    logger.debug(f"chunk.consolidate_logs writing omnibus log to {log_output_path}")
    omnibus_df.to_csv(log_output_path, mode="w", index=False)

    #
    # CACHE_FILE
    #

    # write cached chunk_history file for use on subsequent runs, optimized for use by ChunkHistorian
    # Essentially a pandas DataFrame keyed by chunk_tag with columns num_rows, rss, uss, bytes

    omnibus_df = omnibus_df[omnibus_df[C_DEPTH] == 1]
    zero_rows = omnibus_df[C_NUM_ROWS] <= 0
    if zero_rows.any():
        # this should only happen when chunk_log() instantiates the base ChunkSizer.
        # Since chunk_log is not chunked (chunk_size is always 0) there is no need for its history record in the cache
        logger.debug(
            f"consolidate_logs dropping {zero_rows.sum()} rows where {C_NUM_ROWS} == 0"
        )
        omnibus_df = omnibus_df[omnibus_df[C_NUM_ROWS] > 0]

    omnibus_df = omnibus_df[[C_CHUNK_TAG, C_NUM_ROWS] + CUM_OVERHEAD_COLUMNS]

    # aggregate by chunk_tag
    omnibus_df = omnibus_df.groupby(C_CHUNK_TAG).sum().reset_index(drop=False)

    # rename cum_overhead_xxx to xxx
    omnibus_df = omnibus_df.rename(columns={f"cum_overhead_{m}": m for m in METRICS})

    # compute row_size
    num_rows = omnibus_df[C_NUM_ROWS]
    for m in USS_CHUNK_METHODS:
        omnibus_df[f"{m}_row_size"] = np.ceil(
            overhead_for_chunk_method(omnibus_df, m) / num_rows
        ).astype(int)

    omnibus_df = omnibus_df.sort_values(by=C_CHUNK_TAG)

    log_dir_output_path = config.log_file_path(CACHE_FILE_NAME, prefix=False)
    logger.debug(
        f"chunk.consolidate_logs writing omnibus chunk cache to {log_dir_output_path}"
    )
    omnibus_df.to_csv(log_dir_output_path, mode="w", index=False)

    if (chunk_training_mode() == MODE_RETRAIN) or not _HISTORIAN.have_cached_history:

        if config.setting("resume_after"):
            # FIXME
            logger.warning(
                f"Not updating chunk_log cache directory because resume_after"
            )
        else:
            cache_dir_output_path = os.path.join(
                config.get_cache_dir(), CACHE_FILE_NAME
            )
            logger.debug(
                f"chunk.consolidate_logs writing chunk cache to {cache_dir_output_path}"
            )
            omnibus_df.to_csv(cache_dir_output_path, mode="w", index=False)


class ChunkHistorian(object):
    """
    Utility for estimating row_size
    """

    def __init__(self):

        self.chunk_log_path = None
        self.have_cached_history = None
        self.cached_history_df = None

    def load_cached_history(self):

        if chunk_training_mode() == MODE_RETRAIN:
            # don't need cached history if retraining
            return

        if self.have_cached_history is not None:
            # already loaded, nothing to do
            return

        chunk_cache_path = os.path.join(config.get_cache_dir(), CACHE_FILE_NAME)

        logger.debug(
            f"ChunkHistorian load_cached_history chunk_cache_path {chunk_cache_path}"
        )

        if os.path.exists(chunk_cache_path):
            logger.debug(
                f"ChunkHistorian load_cached_history reading cached chunk history from {CACHE_FILE_NAME}"
            )
            df = pd.read_csv(chunk_cache_path, comment="#")

            self.cached_history_df = df

            for c in CHUNK_CACHE_COLUMNS:
                assert (
                    c in df
                ), f"Expected column '{c}' not in chunk_cache: {chunk_cache_path}"

            self.have_cached_history = True
        else:
            self.have_cached_history = False

            if chunk_training_mode() == MODE_CHUNKLESS:
                return

            if chunk_training_mode() == MODE_PRODUCTION:
                # raise RuntimeError(f"chunk_training_mode is {MODE_PRODUCTION} but no chunk_cache: {chunk_cache_path}")

                SETTINGS["chunk_training_mode"] = MODE_RETRAIN
                logger.warning(
                    f"chunk_training_mode is {MODE_PRODUCTION} but no chunk_cache: {chunk_cache_path}"
                )
                logger.warning(
                    f"chunk_training_mode falling back to {chunk_training_mode()}"
                )

    def cached_history_for_chunk_tag(self, chunk_tag):

        history = {}
        self.load_cached_history()

        if self.have_cached_history:

            try:
                df = self.cached_history_df[
                    self.cached_history_df[C_CHUNK_TAG] == chunk_tag
                ]
                if len(df) > 0:

                    if len(df) > 1:
                        # don't expect this, but not fatal
                        logger.warning(
                            f"ChunkHistorian aggregating {len(df)} multiple rows for {chunk_tag}"
                        )

                    # history for this chunk_tag as dict column sums ('num_rows' and cum_overhead for each metric)
                    # {'num_rows: <n>, 'rss': <n>, 'uss': <n>, 'bytes': <n>}
                    history = df.sum().to_dict()

            except Exception as e:
                logger.warning(
                    f"ChunkHistorian Error loading cached history for {chunk_tag}"
                )
                raise e

        return history

    def cached_row_size(self, chunk_tag):

        row_size = 0

        cached_history = self.cached_history_for_chunk_tag(chunk_tag)
        if cached_history:
            cum_overhead = {m: cached_history[m] for m in METRICS}
            num_rows = cached_history[C_NUM_ROWS]

            # initial_row_size based on cum_overhead and rows_processed from chunk_cache
            row_size = math.ceil(overhead_for_chunk_method(cum_overhead) / num_rows)

        return row_size

    def write_history(self, history, chunk_tag):

        assert chunk_training_mode() not in (MODE_PRODUCTION, MODE_CHUNKLESS)

        history_df = pd.DataFrame.from_dict(history)

        # just want the last, most up to date row
        history_df = history_df.tail(1)

        history_df[C_CHUNK_TAG] = chunk_tag
        history_df["process"] = multiprocessing.current_process().name

        history_df = history_df[CHUNK_HISTORY_COLUMNS]

        if self.chunk_log_path is None:
            self.chunk_log_path = config.log_file_path(LOG_FILE_NAME)

        tracing.write_df_csv(
            history_df,
            self.chunk_log_path,
            index_label=None,
            columns=None,
            column_labels=None,
            transpose=False,
        )


_HISTORIAN = ChunkHistorian()


class ChunkLedger(object):
    """ """

    def __init__(self, trace_label, chunk_size, baseline_rss, baseline_uss, headroom):
        self.trace_label = trace_label
        self.chunk_size = chunk_size
        self.headroom = headroom
        self.base_chunk_size = get_base_chunk_size()

        self.tables = {}
        self.hwm_bytes = {"value": 0, "info": f"{trace_label}.init"}
        self.hwm_rss = {"value": baseline_rss, "info": f"{trace_label}.init"}
        self.hwm_uss = {"value": baseline_uss, "info": f"{trace_label}.init"}
        self.total_bytes = 0

    def audit(self, msg, bytes=0, rss=0, uss=0, from_rss_monitor=False):

        assert chunk_training_mode() not in (MODE_PRODUCTION, MODE_CHUNKLESS)

        MAX_OVERDRAFT = 0.2

        if not self.base_chunk_size:
            return

        mem_panic_threshold = self.base_chunk_size * (1 + MAX_OVERDRAFT)
        bytes_panic_threshold = self.headroom + (self.base_chunk_size * MAX_OVERDRAFT)

        if bytes > bytes_panic_threshold:
            logger.warning(
                f"out_of_chunk_memory: "
                f"bytes: {bytes} headroom: {self.headroom} chunk_size: {self.base_chunk_size} {msg}"
            )

        if chunk_metric() == RSS and rss > mem_panic_threshold:
            rss, _ = mem.get_rss(force_garbage_collect=True, uss=False)
            if rss > mem_panic_threshold:
                logger.warning(
                    f"out_of_chunk_memory: "
                    f"rss: {rss} chunk_size: {self.base_chunk_size} {msg}"
                )

        if chunk_metric() == USS and uss > mem_panic_threshold:
            _, uss = mem.get_rss(force_garbage_collect=True, uss=True)
            if uss > mem_panic_threshold:
                logger.warning(
                    f"out_of_chunk_memory: "
                    f"uss: {uss} chunk_size: {self.base_chunk_size} {msg}"
                )

    def close(self):
        logger.debug(f"ChunkLedger.close trace_label: {self.trace_label}")
        logger.debug(
            f"ChunkLedger.close hwm_bytes: {self.hwm_bytes.get('value', 0)} {self.hwm_bytes['info']}"
        )
        logger.debug(
            f"ChunkLedger.close hwm_rss {self.hwm_rss['value']} {self.hwm_rss['info']}"
        )
        logger.debug(
            f"ChunkLedger.close hwm_uss {self.hwm_uss['value']} {self.hwm_uss['info']}"
        )

    def log_df(self, table_name, df):
        def size_it(df):
            if isinstance(df, pd.Series):
                elements = util.iprod(df.shape)
                bytes = 0 if not elements else df.memory_usage(index=True)
            elif isinstance(df, pd.DataFrame):
                elements = util.iprod(df.shape)
                bytes = 0 if not elements else df.memory_usage(index=True).sum()
            elif isinstance(df, np.ndarray):
                elements = util.iprod(df.shape)
                bytes = df.nbytes
            elif isinstance(df, list):
                # dict of series, dataframe, or ndarray (e.g. assign assign_variables target and temp dicts)
                elements = 0
                bytes = 0
                for v in df:
                    e, b = size_it(v)
                    elements += e
                    bytes += b
            elif isinstance(df, dict):
                # dict of series, dataframe, or ndarray (e.g. assign assign_variables target and temp dicts)
                elements = 0
                bytes = 0
                for k, v in df.items():
                    e, b = size_it(v)
                    elements += e
                    bytes += b
            else:
                logger.error(f"size_it unknown type: {type(df)}")
                assert False
            return elements, bytes

        assert chunk_training_mode() not in (MODE_PRODUCTION, MODE_CHUNKLESS)

        if df is None:
            elements, bytes = (0, 0)
            delta_bytes = bytes - self.tables.get(table_name, 0)
            self.tables[table_name] = bytes
        else:
            elements, bytes = size_it(df)
            delta_bytes = bytes - self.tables.get(table_name, 0)
            self.tables[table_name] = bytes

        # shape is informational and only used for logging
        if df is None:
            shape = None
        elif isinstance(df, list):
            shape = f"list({[x.shape for x in df]})"
        elif isinstance(df, dict):
            shape = f"dict({[v.shape for v in df.values()]})"
        else:
            shape = df.shape

        logger.debug(
            f"log_df delta_bytes: {util.INT(delta_bytes).rjust(12)} {table_name} {shape} {self.trace_label}"
        )

        # update current total_bytes count
        self.total_bytes = sum(self.tables.values())

    def check_local_hwm(self, hwm_trace_label, rss, uss, total_bytes):

        assert chunk_training_mode() not in (MODE_PRODUCTION, MODE_CHUNKLESS)

        from_rss_monitor = total_bytes is None

        info = (
            f"rss: {GB(rss)} "
            f"uss: {GB(uss)} "
            f"base_chunk_size: {GB(self.base_chunk_size)} "
            f"op: {hwm_trace_label}"
        )

        if total_bytes:
            info = f"bytes: {GB(total_bytes)} " + info

            if total_bytes > self.hwm_bytes["value"]:
                # total_bytes high water mark
                self.hwm_bytes["value"] = total_bytes
                self.hwm_bytes["info"] = info
                self.audit(hwm_trace_label, bytes=total_bytes)

        if rss > self.hwm_rss["value"]:
            # rss high water mark
            self.hwm_rss["value"] = rss
            self.hwm_rss["info"] = info
            self.audit(hwm_trace_label, rss=rss, from_rss_monitor=from_rss_monitor)

        if uss > self.hwm_uss["value"]:
            # uss high water mark
            self.hwm_uss["value"] = uss
            self.hwm_uss["info"] = info
            self.audit(hwm_trace_label, uss=uss, from_rss_monitor=from_rss_monitor)

        # silently registers global high water mark
        mem.check_global_hwm(RSS, rss, hwm_trace_label)
        mem.check_global_hwm(USS, uss, hwm_trace_label)
        if total_bytes:
            mem.check_global_hwm(BYTES, total_bytes, hwm_trace_label)

    def get_hwm_rss(self):
        with ledger_lock:
            net_rss = self.hwm_rss["value"]
        return net_rss

    def get_hwm_uss(self):
        with ledger_lock:
            net_uss = self.hwm_uss["value"]
        return net_uss

    def get_hwm_bytes(self):
        return self.hwm_bytes["value"]


def log_rss(trace_label, force=False):

    if chunk_training_mode() == MODE_CHUNKLESS:
        # no memory tracing at all in chunkless mode
        return

    assert len(CHUNK_LEDGERS) > 0, f"log_rss called without current chunker."

    hwm_trace_label = f"{trace_label}.log_rss"

    if chunk_training_mode() == MODE_PRODUCTION:
        # FIXME - this trace_memory_info call slows things down a lot so it is turned off for now
        # trace_ticks = 0 if force else mem.MEM_TRACE_TICK_LEN
        # mem.trace_memory_info(hwm_trace_label, trace_ticks=trace_ticks)
        return

    rss, uss = mem.trace_memory_info(hwm_trace_label)

    # check local hwm for all ledgers
    with ledger_lock:
        for c in CHUNK_LEDGERS:
            c.check_local_hwm(hwm_trace_label, rss, uss, total_bytes=None)


def log_df(trace_label, table_name, df):

    if chunk_training_mode() in (MODE_PRODUCTION, MODE_CHUNKLESS):
        return

    assert len(CHUNK_LEDGERS) > 0, f"log_df called without current chunker."

    op = "del" if df is None else "add"
    hwm_trace_label = f"{trace_label}.{op}.{table_name}"

    rss, uss = mem.trace_memory_info(hwm_trace_label)

    cur_chunker = CHUNK_LEDGERS[-1]

    # registers this df and recalc total_bytes
    cur_chunker.log_df(table_name, df)

    total_bytes = sum([c.total_bytes for c in CHUNK_LEDGERS])

    # check local hwm for all ledgers
    with ledger_lock:
        for c in CHUNK_LEDGERS:
            c.check_local_hwm(hwm_trace_label, rss, uss, total_bytes)


class MemMonitor(threading.Thread):
    def __init__(self, trace_label, stop_snooping):
        self.trace_label = trace_label
        self.stop_snooping = stop_snooping
        threading.Thread.__init__(self)

    def run(self):
        log_rss(self.trace_label)
        while not self.stop_snooping.wait(timeout=mem.MEM_SNOOP_TICK_LEN):
            log_rss(self.trace_label)


class ChunkSizer(object):
    """ """

    def __init__(self, chunk_tag, trace_label, num_choosers=0, chunk_size=0):

        self.depth = len(CHUNK_SIZERS) + 1

        if chunk_training_mode() != MODE_CHUNKLESS:
            if chunk_metric() == USS:
                self.rss, self.uss = mem.get_rss(force_garbage_collect=True, uss=True)
            else:
                self.rss, _ = mem.get_rss(force_garbage_collect=True, uss=False)
                self.uss = 0

            if self.depth > 1:
                # nested chunkers should be unchunked
                assert chunk_size == 0

                # if we are in a nested call, then we must be in the scope of active Ledger
                # so any rss accumulated so far should be attributed to the parent active ledger
                assert len(CHUNK_SIZERS) == len(CHUNK_LEDGERS)
                parent = CHUNK_SIZERS[-1]
                assert parent.chunk_ledger is not None

                log_rss(
                    trace_label
                )  # give parent a complementary log_rss reading entering sub context
        else:
            self.rss, self.uss = 0, 0
            chunk_size = 0
            config.override_setting("chunk_size", 0)

        self.chunk_tag = chunk_tag
        self.trace_label = trace_label
        self.chunk_size = chunk_size

        self.num_choosers = num_choosers
        self.rows_processed = 0

        min_chunk_ratio = min_available_chunk_ratio()
        assert (
            0 <= min_chunk_ratio <= 1
        ), f"min_chunk_ratio setting {min_chunk_ratio} is not in range [0..1]"
        self.min_chunk_size = chunk_size * min_chunk_ratio

        self.initial_row_size = 0
        self.rows_per_chunk = 0
        self.chunk_ledger = None
        self.history = {}

        self.cum_rows = 0
        self.cum_overhead = {m: 0 for m in METRICS}

        # if production mode, to reduce volatility, initialize cum_overhead and cum_rows from cache
        if chunk_training_mode() in [MODE_ADAPTIVE, MODE_PRODUCTION]:
            cached_history = _HISTORIAN.cached_history_for_chunk_tag(self.chunk_tag)
            if cached_history:
                self.cum_overhead = {m: cached_history[m] for m in METRICS}
                self.cum_rows = cached_history[C_NUM_ROWS]

                logger.debug(
                    f"{self.trace_label}.ChunkSizer - cached history "
                    f"cum_rows: {self.cum_rows} "
                    f"cum_overhead: {self.cum_overhead} "
                )

        # add self to CHUNK_SIZERS list before setting base_chunk_size (since we might be base chunker)
        CHUNK_SIZERS.append(self)

        self.base_chunk_size = CHUNK_SIZERS[0].chunk_size

        # need base_chunk_size to calc headroom
        self.headroom = self.available_headroom(
            self.uss if chunk_metric() == USS else self.rss
        )

    def close(self):

        if ((self.depth == 1) or WRITE_SUBCHUNK_HISTORY) and (
            chunk_training_mode() not in (MODE_PRODUCTION, MODE_CHUNKLESS)
        ):
            _HISTORIAN.write_history(self.history, self.chunk_tag)

        _chunk_sizer = CHUNK_SIZERS.pop()
        assert _chunk_sizer == self

    def available_headroom(self, xss):

        headroom = self.base_chunk_size - xss

        # adjust deficient headroom to min_chunk_size
        if headroom < self.min_chunk_size:

            if self.base_chunk_size > 0:
                logger.warning(
                    f"Not enough memory for minimum chunk_size without exceeding specified chunk_size. "
                    f"available_headroom: {util.INT(headroom)} "
                    f"min_chunk_size: {util.INT(self.min_chunk_size)} "
                    f"base_chunk_size: {util.INT(self.base_chunk_size)}"
                )

            headroom = self.min_chunk_size

        return headroom

    def initial_rows_per_chunk(self):

        # whatever the TRAINING_MODE, use cache to determine initial_row_size
        # (presumably preferable to default_initial_rows_per_chunk)
        self.initial_row_size = _HISTORIAN.cached_row_size(self.chunk_tag)

        if self.chunk_size == 0:
            rows_per_chunk = self.num_choosers
            estimated_number_of_chunks = 1
            self.initial_row_size = 0
        else:

            # we should be a base chunker
            assert len(CHUNK_LEDGERS) == 0, f"len(CHUNK_LEDGERS): {len(CHUNK_LEDGERS)}"

            if self.initial_row_size > 0:
                max_rows_per_chunk = np.maximum(
                    int(self.headroom / self.initial_row_size), 1
                )
                rows_per_chunk = np.clip(max_rows_per_chunk, 1, self.num_choosers)
                estimated_number_of_chunks = math.ceil(
                    self.num_choosers / rows_per_chunk
                )

                logger.debug(
                    f"{self.trace_label}.initial_rows_per_chunk - initial_row_size: {self.initial_row_size}"
                )
            else:
                # if no initial_row_size from cache, fall back to default_initial_rows_per_chunk
                self.initial_row_size = 0
                rows_per_chunk = min(
                    self.num_choosers, default_initial_rows_per_chunk()
                )
                estimated_number_of_chunks = None

                assert chunk_training_mode() != MODE_PRODUCTION

        # cum_rows is out of phase with cum_overhead
        # since we won't know observed_chunk_size until AFTER yielding the chunk
        self.rows_per_chunk = rows_per_chunk
        self.rows_processed += rows_per_chunk
        self.cum_rows += rows_per_chunk

        logger.debug(
            f"{self.trace_label}.initial_rows_per_chunk - "
            f"rows_per_chunk: {self.rows_per_chunk} "
            f"headroom: {self.headroom} "
            f"initial_row_size: {self.initial_row_size} "
        )

        return rows_per_chunk, estimated_number_of_chunks

    def adaptive_rows_per_chunk(self, i):
        # rows_processed is out of phase with cum_overhead
        # overhead is the actual bytes/rss used top process chooser chunk with prev_rows_per_chunk rows

        prev_rows_per_chunk = self.rows_per_chunk
        prev_rows_processed = self.rows_processed
        prev_cum_rows = self.cum_rows
        prev_headroom = self.headroom

        prev_rss = self.rss
        prev_uss = self.uss

        if chunk_training_mode() != MODE_PRODUCTION:

            if chunk_metric() == USS:
                self.rss, self.uss = mem.get_rss(force_garbage_collect=True, uss=True)
            else:
                self.rss, _ = mem.get_rss(force_garbage_collect=True, uss=False)
                self.uss = 0

        self.headroom = self.available_headroom(
            self.uss if chunk_metric() == USS else self.rss
        )

        rows_remaining = self.num_choosers - prev_rows_processed

        if chunk_training_mode() == MODE_PRODUCTION:
            # since overhead changes we don't necessarily want the same number of rows per chunk every time
            # but we do use the row_size from cache which we trust is stable
            # which is stored in self.initial_row_size because initial_rows_per_chunk used it for the first chunk
            observed_row_size = self.initial_row_size
            overhead = self.cum_overhead.copy()
        else:

            # calculate overhead for this chunk iteration
            overhead = {}
            overhead[BYTES] = self.chunk_ledger.get_hwm_bytes()
            overhead[RSS] = self.chunk_ledger.get_hwm_rss() - prev_rss
            overhead[USS] = self.chunk_ledger.get_hwm_uss() - prev_uss

            for m in METRICS:
                self.cum_overhead[m] += overhead[m]

            observed_row_size = prev_cum_rows and math.ceil(
                overhead_for_chunk_method(self.cum_overhead) / prev_cum_rows
            )

        # rows_per_chunk is closest number of chooser rows to achieve chunk_size without exceeding it
        if observed_row_size > 0:
            self.rows_per_chunk = int(self.headroom / observed_row_size)
        else:
            # they don't appear to have used any memory; increase cautiously in case small sample size was to blame
            self.rows_per_chunk = 2 * prev_rows_per_chunk

        self.rows_per_chunk = np.clip(self.rows_per_chunk, 1, rows_remaining)
        self.rows_processed += self.rows_per_chunk
        estimated_number_of_chunks = (
            i + math.ceil(rows_remaining / self.rows_per_chunk) if rows_remaining else i
        )

        self.history.setdefault(C_TIME, []).append(
            datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S.%f")
        )
        self.history.setdefault(C_DEPTH, []).append(self.depth)
        for m in METRICS:
            self.history.setdefault(f"cum_overhead_{m}", []).append(
                self.cum_overhead[m]
            )
        self.history.setdefault(C_NUM_ROWS, []).append(prev_cum_rows)
        self.history.setdefault("chunk", []).append(i)
        self.history.setdefault("chunk_size", []).append(self.chunk_size)
        self.history.setdefault("row_size", []).append(observed_row_size)

        # diagnostics not reported by ChunkHistorian

        if chunk_metric() == USS:
            self.history.setdefault("prev_uss", []).append(prev_uss)
            self.history.setdefault("cur_uss", []).append(self.uss)
        else:
            self.history.setdefault("prev_rss", []).append(prev_rss)
            self.history.setdefault("cur_rss", []).append(self.rss)

        self.history.setdefault("prev_headroom", []).append(prev_headroom)
        self.history.setdefault("cur_headroom", []).append(self.headroom)

        for m in METRICS:
            self.history.setdefault(f"overhead_{m}", []).append(overhead[m])

        self.history.setdefault("new_rows_processed", []).append(self.rows_processed)
        self.history.setdefault("new_rows_per_chunk", []).append(self.rows_per_chunk)
        self.history.setdefault("estimated_num_chunks", []).append(
            estimated_number_of_chunks
        )

        history_df = pd.DataFrame.from_dict(self.history)
        if LOG_SUBCHUNK_HISTORY:
            logger.debug(
                f"ChunkSizer.adaptive_rows_per_chunk {self.chunk_tag}\n{history_df.transpose()}"
            )

        # input()

        if chunk_training_mode() not in (MODE_PRODUCTION, MODE_CHUNKLESS):
            self.cum_rows += self.rows_per_chunk

        return self.rows_per_chunk, estimated_number_of_chunks

    @contextmanager
    def ledger(self):

        # don't do anything in chunkless mode
        if chunk_training_mode() == MODE_CHUNKLESS:
            yield
            return

        mem_monitor = None

        # nested chunkers should be unchunked
        if len(CHUNK_LEDGERS) > 0:
            assert self.chunk_size == 0

        with ledger_lock:
            self.chunk_ledger = ChunkLedger(
                self.trace_label, self.chunk_size, self.rss, self.uss, self.headroom
            )
            CHUNK_LEDGERS.append(self.chunk_ledger)

        # reality check - there should be one ledger per sizer
        assert len(CHUNK_LEDGERS) == len(CHUNK_SIZERS)

        try:
            # all calls to log_df within this block will be directed to top level chunk_ledger
            # and passed on down the stack to the base to support hwm tallies

            # if this is a base chunk_sizer (and ledger) then start a thread to monitor rss usage
            if (len(CHUNK_LEDGERS) == 1) and ENABLE_MEMORY_MONITOR:
                stop_snooping = threading.Event()
                mem_monitor = MemMonitor(self.trace_label, stop_snooping)
                mem_monitor.start()

            log_rss(
                self.trace_label, force=True
            )  # make sure we get at least one reading
            yield
            log_rss(
                self.trace_label, force=True
            )  # make sure we get at least one reading

        finally:

            if mem_monitor is not None:

                if not mem_monitor.is_alive():
                    logger.error(f"mem_monitor for {self.trace_label} died!")
                    bug  # bug

                stop_snooping.set()
                while mem_monitor.is_alive():
                    logger.debug(
                        f"{self.trace_label} waiting for mem_monitor thread to terminate"
                    )
                    mem_monitor.join(timeout=MEM_MONITOR_TICK)

            with ledger_lock:
                self.chunk_ledger.close()
                CHUNK_LEDGERS.pop()
                self.chunk_ledger = None


@contextmanager
def chunk_log(trace_label, chunk_tag=None, base=False):

    # With `base=True` this method can be used to instantiate
    # a ChunkSizer class object without actually chunking. This
    # avoids breaking the assertion below.

    assert base == (len(CHUNK_SIZERS) == 0)

    trace_label = f"{trace_label}.chunk_log"

    chunk_tag = chunk_tag or trace_label
    num_choosers = 0
    chunk_size = 0

    chunk_sizer = ChunkSizer(chunk_tag, trace_label, num_choosers, chunk_size)

    chunk_sizer.initial_rows_per_chunk()

    with chunk_sizer.ledger():

        yield

        if chunk_training_mode() != MODE_CHUNKLESS:
            chunk_sizer.adaptive_rows_per_chunk(1)

    chunk_sizer.close()


@contextmanager
def chunk_log_skip():

    yield

    None


def adaptive_chunked_choosers(choosers, chunk_size, trace_label, chunk_tag=None):

    # generator to iterate over choosers

    chunk_tag = chunk_tag or trace_label

    num_choosers = len(choosers.index)
    assert num_choosers > 0
    assert chunk_size >= 0

    logger.info(
        f"{trace_label} Running adaptive_chunked_choosers with {num_choosers} choosers"
    )

    chunk_sizer = ChunkSizer(chunk_tag, trace_label, num_choosers, chunk_size)

    rows_per_chunk, estimated_number_of_chunks = chunk_sizer.initial_rows_per_chunk()

    i = offset = 0
    while offset < num_choosers:

        i += 1
        assert offset + rows_per_chunk <= num_choosers

        chunk_trace_label = trace_label_for_chunk(trace_label, chunk_size, i)

        with chunk_sizer.ledger():

            # grab the next chunk based on current rows_per_chunk
            chooser_chunk = choosers[offset : offset + rows_per_chunk]

            logger.info(
                f"Running chunk {i} of {estimated_number_of_chunks or '?'} "
                f"with {len(chooser_chunk)} of {num_choosers} choosers"
            )

            yield i, chooser_chunk, chunk_trace_label

            offset += rows_per_chunk

            if chunk_training_mode() != MODE_CHUNKLESS:
                (
                    rows_per_chunk,
                    estimated_number_of_chunks,
                ) = chunk_sizer.adaptive_rows_per_chunk(i)

    chunk_sizer.close()


def adaptive_chunked_choosers_and_alts(
    choosers, alternatives, chunk_size, trace_label, chunk_tag=None
):
    """
    generator to iterate over choosers and alternatives in chunk_size chunks

    like chunked_choosers, but also chunks alternatives
    for use with sampled alternatives which will have different alternatives (and numbers of alts)

    There may be up to sample_size (or as few as one) alternatives for each chooser
    because alternatives may have been sampled more than once,  but pick_count for those
    alternatives will always sum to sample_size.

    When we chunk the choosers, we need to take care chunking the alternatives as there are
    varying numbers of them for each chooser. Since alternatives appear in the same order
    as choosers, we can use cumulative pick_counts to identify boundaries of sets of alternatives

    Parameters
    ----------
    choosers
    alternatives : pandas DataFrame
        sample alternatives including pick_count column in same order as choosers
    rows_per_chunk : int

    Yields
    ------
    i : int
        one-based index of current chunk
    num_chunks : int
        total number of chunks that will be yielded
    choosers : pandas DataFrame slice
        chunk of choosers
    alternatives : pandas DataFrame slice
        chunk of alternatives for chooser chunk
    """

    chunk_tag = chunk_tag or trace_label

    num_choosers = len(choosers.index)
    num_alternatives = len(alternatives.index)
    assert num_choosers > 0

    # alternatives index should match choosers (except with duplicate repeating alt rows)
    assert choosers.index.equals(
        alternatives.index[~alternatives.index.duplicated(keep="first")]
    )

    last_repeat = alternatives.index != np.roll(alternatives.index, -1)

    assert (num_choosers == 1) or choosers.index.equals(alternatives.index[last_repeat])
    assert (
        "pick_count" in alternatives.columns
        or choosers.index.name == alternatives.index.name
    )
    assert choosers.index.name == alternatives.index.name

    logger.info(
        f"{trace_label} Running adaptive_chunked_choosers_and_alts "
        f"with {num_choosers} choosers and {num_alternatives} alternatives"
    )

    chunk_sizer = ChunkSizer(chunk_tag, trace_label, num_choosers, chunk_size)
    rows_per_chunk, estimated_number_of_chunks = chunk_sizer.initial_rows_per_chunk()
    assert (rows_per_chunk > 0) and (rows_per_chunk <= num_choosers)

    # alt chunks boundaries are where index changes
    alt_ids = alternatives.index.values
    alt_chunk_ends = np.where(alt_ids[:-1] != alt_ids[1:])[0] + 1
    alt_chunk_ends = np.append(
        [0], alt_chunk_ends
    )  # including the first to simplify indexing
    alt_chunk_ends = np.append(
        alt_chunk_ends, [len(alternatives.index)]
    )  # end of final chunk

    i = offset = alt_offset = 0
    while offset < num_choosers:
        i += 1

        assert (
            offset + rows_per_chunk <= num_choosers
        ), f"i {i} offset {offset} rows_per_chunk {rows_per_chunk} num_choosers {num_choosers}"

        chunk_trace_label = trace_label_for_chunk(trace_label, chunk_size, i)

        with chunk_sizer.ledger():

            chooser_chunk = choosers[offset : offset + rows_per_chunk]

            alt_end = alt_chunk_ends[offset + rows_per_chunk]
            alternative_chunk = alternatives[alt_offset:alt_end]

            assert len(chooser_chunk.index) == len(
                np.unique(alternative_chunk.index.values)
            )
            assert (
                chooser_chunk.index == np.unique(alternative_chunk.index.values)
            ).all()

            logger.info(
                f"Running chunk {i} of {estimated_number_of_chunks or '?'} "
                f"with {len(chooser_chunk)} of {num_choosers} choosers"
            )

            yield i, chooser_chunk, alternative_chunk, chunk_trace_label

            offset += rows_per_chunk
            alt_offset = alt_end

            if chunk_training_mode() != MODE_CHUNKLESS:
                (
                    rows_per_chunk,
                    estimated_number_of_chunks,
                ) = chunk_sizer.adaptive_rows_per_chunk(i)

    chunk_sizer.close()


def adaptive_chunked_choosers_by_chunk_id(
    choosers, chunk_size, trace_label, chunk_tag=None
):
    # generator to iterate over choosers in chunk_size chunks
    # like chunked_choosers but based on chunk_id field rather than dataframe length
    # (the presumption is that choosers has multiple rows with the same chunk_id that
    # all have to be included in the same chunk)
    # FIXME - we pathologically know name of chunk_id col in households table

    chunk_tag = chunk_tag or trace_label

    num_choosers = choosers["chunk_id"].max() + 1
    assert num_choosers > 0

    chunk_sizer = ChunkSizer(chunk_tag, trace_label, num_choosers, chunk_size)

    rows_per_chunk, estimated_number_of_chunks = chunk_sizer.initial_rows_per_chunk()

    i = offset = 0
    while offset < num_choosers:

        i += 1
        assert offset + rows_per_chunk <= num_choosers

        chunk_trace_label = trace_label_for_chunk(trace_label, chunk_size, i)

        with chunk_sizer.ledger():

            chooser_chunk = choosers[
                choosers["chunk_id"].between(offset, offset + rows_per_chunk - 1)
            ]

            logger.info(
                f"{trace_label} Running chunk {i} of {estimated_number_of_chunks or '?'} "
                f"with {rows_per_chunk} of {num_choosers} choosers"
            )

            yield i, chooser_chunk, chunk_trace_label

            offset += rows_per_chunk

            if chunk_training_mode() != MODE_CHUNKLESS:
                (
                    rows_per_chunk,
                    estimated_number_of_chunks,
                ) = chunk_sizer.adaptive_rows_per_chunk(i)

    chunk_sizer.close()
