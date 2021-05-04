# ActivitySim
# See full license in LICENSE.txt.
from builtins import input

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

from . import mem
from . import tracing
from . import config
from . import util

from .util import GB

logger = logging.getLogger(__name__)


TRACK_USS = True  #bug - remove?

CHUNK_METHOD_RSS = 'rss'
CHUNK_METHOD_USS = 'uss'
CHUNK_METHOD_BYTES = 'bytes'
CHUNK_METHOD_HYBRID_RSS = 'hybrid_rss'
CHUNK_METHOD_HYBRID_USS = 'hybrid_uss'
DEFAULT_CHUNK_METHOD = CHUNK_METHOD_HYBRID_USS if TRACK_USS else CHUNK_METHOD_HYBRID_RSS
CHUNK_METHODS = [CHUNK_METHOD_RSS, CHUNK_METHOD_USS, CHUNK_METHOD_BYTES,
                 CHUNK_METHOD_HYBRID_RSS, CHUNK_METHOD_HYBRID_USS]
USS_CHUNK_METHODS = [CHUNK_METHOD_USS, CHUNK_METHOD_HYBRID_USS, CHUNK_METHOD_BYTES]  #bug - CHUNK_METHOD_BYTES?

CHUNK_LEDGERS = []
CHUNK_SIZERS = []

BYTES_PER_ELEMENT = 8

DEFAULT_INITIAL_ROWS_PER_CHUNK = 10  # fallback for setting

MEM_MONITOR_TICK = 1  # in seconds
ENABLE_MEMORY_MONITOR = True

LOG_SUBCHUNKS = True

# FIXME should we update this every chunk? Or does it make available_headroom too jittery?
RESET_RSS_BASELINE_FOR_EACH_CHUNK = True

CACHE_HISTORY = True
CACHE_FILE_NAME = 'cached_chunk_log.csv'
LOG_FILE_NAME = 'chunk_history.csv'
OMNIBUS_LOG_FILE_NAME = f"omnibus_{LOG_FILE_NAME}"

HWM = {}
_SHARED_MEM_SIZE = None

ledger_lock = threading.Lock()

C_CHUNK_TAG = 'tag'
C_DEPTH = 'depth'
C_NUM_ROWS = 'rows_processed'
C_TIME = 'time'


# columns to write to LOG_FILE
CUM_OVERHEAD_COLUMNS = [f'cum_overhead_{m}' for m in CHUNK_METHODS]
CHUNK_HISTORY_COLUMNS = [C_TIME, C_CHUNK_TAG] + CUM_OVERHEAD_COLUMNS + \
                        [C_NUM_ROWS, 'row_size', 'chunk_size', C_DEPTH, 'process', 'chunk']


def chunk_metric(chunk_method):
    assert chunk_method in CHUNK_METHODS
    return 'uss' if chunk_method in USS_CHUNK_METHODS else 'rss'


def update_cached_chunk_log():
    # no need to complain if not chunk_training?
    return config.setting('update_cached_chunk_log', True)


def chunk_training():
    return config.setting('chunk_training', True)


def chunk_logging():
    return len(CHUNK_LEDGERS) > 0


def get_default_initial_rows_per_chunk():
    return config.setting('default_initial_rows_per_chunk', DEFAULT_INITIAL_ROWS_PER_CHUNK)


def trace_label_for_chunk(trace_label, chunk_size, i):
    # add chunk_num to trace_label
    # if chunk_size > 0:
    #     trace_label = tracing.extend_trace_label(trace_label, f'chunk_{i}')
    return trace_label


def get_base_chunk_size():
    assert len(CHUNK_SIZERS) > 0
    return CHUNK_SIZERS[0].chunk_size


def out_of_chunk_memory(msg, bytes=None, rss=None, uss=None, from_rss_monitor=False):

    MAX_OVERDRAFT_RATIO = 1.2

    if from_rss_monitor:
        return

    bytes = bytes or 0

    base_chunk_size = get_base_chunk_size()
    assert base_chunk_size > 0
    panic_threshold = base_chunk_size * MAX_OVERDRAFT_RATIO

    if max(bytes, rss) > panic_threshold:

        # if things are that dire, force_garbage_collect
        rss_after_gc, _ = mem.get_rss(force_garbage_collect=True)

        if rss_after_gc > panic_threshold:

            logger.warning(f"out_of_chunk_memory: base_chunk_size: {base_chunk_size} cur_rss: {rss_after_gc} {msg}")

            # for s in CHUNK_SIZERS[::-1]:
            #     logger.error(f"CHUNK_SIZER {s.trace_label}")
            #
            # for s in CHUNK_LEDGERS[::-1]:
            #     logger.error(f"CHUNK_LOGGER {s.trace_label}")
            #     logger.error(f"--- hwm_bytes {INT(s.hwm_bytes['value'])} {s.hwm_bytes['info']}")
            #     logger.error(f"--- hwm_rss {INT(s.hwm_rss['value'])} {s.hwm_rss['info']}")


def consolidate_logs():

    glob_file_name = config.log_file_path(f"*{LOG_FILE_NAME}", prefix=False)
    glob_files = glob.glob(glob_file_name)

    if not glob_files:
        return

    #
    # OMNIBUS_LOG_FILE
    #

    logger.debug(f"chunk.consolidate_logs reading glob {glob_file_name}")
    omnibus_df = pd.concat((pd.read_csv(f, comment='#') for f in glob_files))

    omnibus_df = omnibus_df.sort_values(by=C_TIME)

    # if we are overwriting MEM_LOG_FILE then presumably we want to delete any subprocess files
    if (LOG_FILE_NAME == OMNIBUS_LOG_FILE_NAME) or len(glob_files) == 1:
        util.delete_files(glob_files, 'chunk.consolidate_logs')

    log_output_path = config.log_file_path(OMNIBUS_LOG_FILE_NAME, prefix=False)
    logger.debug(f"chunk.consolidate_logs writing omnibus log to {log_output_path}")
    omnibus_df.to_csv(log_output_path, mode='w', index=False)

    #
    # CACHE_FILE
    #

    # shouldn't have different depths for the same chunk_tag
    assert not omnibus_df[[C_CHUNK_TAG, C_DEPTH]]\
        .groupby([C_CHUNK_TAG, C_DEPTH]).size()\
        .reset_index(level=1).index.duplicated().any()

    omnibus_df = omnibus_df[omnibus_df[C_DEPTH] == 1]
    zero_rows = omnibus_df[C_NUM_ROWS] <= 0
    if zero_rows.any():
        logger.warning(f"consolidate_logs dropping {zero_rows.sum()} rows where {C_NUM_ROWS} == 0")
        omnibus_df = omnibus_df[omnibus_df[C_NUM_ROWS] > 0]

    omnibus_df = omnibus_df[[C_CHUNK_TAG, C_NUM_ROWS] + CUM_OVERHEAD_COLUMNS]

    # aggregate by chunk_tag
    omnibus_df = omnibus_df.groupby(C_CHUNK_TAG).sum().reset_index(drop=False)

    # compute row_size
    chunk_method = config.setting('chunk_method', DEFAULT_CHUNK_METHOD)
    c_overhead = f'cum_overhead_{chunk_method}'
    omnibus_df['row_size'] = np.ceil(omnibus_df[c_overhead] / omnibus_df[C_NUM_ROWS]).astype(int)

    omnibus_df = omnibus_df.sort_values(by=C_CHUNK_TAG)

    if update_cached_chunk_log():
        cache_output_path = os.path.join(config.get_cache_dir(), CACHE_FILE_NAME)
    else:
        cache_output_path = config.log_file_path(CACHE_FILE_NAME, prefix=False)

    logger.debug(f"chunk.consolidate_logs writing omnibus chunk cache to {cache_output_path}")
    omnibus_df.to_csv(cache_output_path, mode='w', index=False)


class ChunkHistorian(object):
    """
    Utility for estimating row_size
    """
    def __init__(self):

        self.chunk_log_path = None

        self.have_cached_history = None
        self.cached_history_df = None

    def load_cached_history(self):

        chunk_cache_path = os.path.join(config.get_cache_dir(), CACHE_FILE_NAME)

        logger.debug(f"ChunkHistorian load_cached_history chunk_cache_path {chunk_cache_path}")

        if os.path.exists(chunk_cache_path):
            logger.debug(f"ChunkHistorian load_cached_history reading cached chunk history from {CACHE_FILE_NAME}")
            df = pd.read_csv(chunk_cache_path, comment='#')

            self.cached_history_df = df
            self.have_cached_history = True
        else:
            self.have_cached_history = False

        if not (self.have_cached_history or chunk_training()):
            # if we are chunk_training and there is no cache
            raise RuntimeError(f"setting chunk_training is False but expected cache not found: {chunk_cache_path}")

    def cached_row_size(self, chunk_tag, chunk_method):

        if self.have_cached_history is None:
            self.load_cached_history()

        row_size = 0  # this is out fallback

        if self.have_cached_history:

            try:
                df = self.cached_history_df[self.cached_history_df[C_CHUNK_TAG] == chunk_tag]
                if len(df) > 0:

                    if len(df) > 1:
                        # don't expect this, but not fatal
                        logger.warning(f"ChunkHistorian aggregating {len(df)} multiple rows for {chunk_tag}")

                    overhead_column_name = f'cum_overhead_{chunk_method}'

                    assert overhead_column_name in df, \
                        f"ChunkHistorian.cached_row_size unknown chunk_method: {chunk_method}"

                    overhead = df[overhead_column_name].sum()
                    num_rows = df[C_NUM_ROWS].sum()
                    if num_rows > 0:
                        row_size = overhead / num_rows

            except Exception as e:
                logger.warning(f"ChunkHistorian Error calculating row_size for {chunk_tag}")
                raise e

        if (row_size == 0) and not chunk_training():
            raise RuntimeError(f"setting chunk_training is False but no tag in cache for: {chunk_tag}")

        return row_size

    def write_history(self, history, chunk_tag):

        assert chunk_training()

        history_df = pd.DataFrame.from_dict(history)
        logger.debug(f"ChunkSizer {chunk_tag}\n{history_df.transpose()}")

        # just want the last, most up to date row
        history_df = history_df.tail(1)

        history_df[C_CHUNK_TAG] = chunk_tag
        history_df['process'] = multiprocessing.current_process().name

        history_df = history_df[CHUNK_HISTORY_COLUMNS]

        if self.chunk_log_path is None:
            self.chunk_log_path = config.log_file_path(LOG_FILE_NAME)

        tracing.write_df_csv(history_df, self.chunk_log_path, index_label=None,
                             columns=None, column_labels=None, transpose=False)


_HISTORIAN = ChunkHistorian()


class RowSizeEstimator(object):
    """
    Utility for estimating row_size
    """
    def __init__(self, trace_label):
        self.row_size = 0
        self.hwm = 0  # element byte at high water mark
        self.hwm_tag = None  # tag that drove hwm (with ref count)
        self.trace_label = trace_label
        self.bytes = {}
        self.tag_count = {}

    def add_elements(self, elements, tag):
        bytes = elements * BYTES_PER_ELEMENT
        self.bytes[tag] = bytes
        self.tag_count[tag] = self.tag_count.setdefault(tag, 0) + 1  # number of times tag has been seen
        self.row_size += bytes
        logger.debug(f"{self.trace_label} #rowsize {tag} {bytes} ({self.row_size})")

        if self.row_size > self.hwm:
            self.hwm = self.row_size
            # tag that drove hwm (with ref count)
            self.hwm_tag = f'{tag}_{self.tag_count[tag]}' if self.tag_count[tag] > 1 else tag

    def drop_elements(self, tag):
        self.row_size -= self.bytes[tag]
        self.bytes[tag] = 0
        logger.debug(f"{self.trace_label} #rowsize {tag} <drop> ({self.row_size})")

    def get_hwm(self):
        logger.debug(f"{self.trace_label} #rowsize hwm {self.hwm} after {self.hwm_tag}")
        hwm = self.hwm
        return hwm


class ChunkLedger(object):
    """
    ::
    """
    def __init__(self, trace_label, chunk_size, baseline_rss, baseline_uss):
        self.trace_label = trace_label
        self.chunk_size = chunk_size

        self.tables = {}
        self.hwm_bytes = {'value': 0, 'info': f'{trace_label}.init'}
        self.hwm_rss = {'value': baseline_rss, 'info': f'{trace_label}.init'}
        self.hwm_uss = {'value': baseline_uss, 'info': f'{trace_label}.init'}
        self.total_bytes = 0

    def close(self):
        logger.debug(f"ChunkLedger.close hwm_bytes: {self.hwm_bytes.get('value', 0)} {self.hwm_bytes['info']}")
        logger.debug(f"ChunkLedger.close hwm_rss {self.hwm_rss['value']} {self.hwm_rss['info']}")

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
        elif isinstance(df, dict):
            # ordinarily all elements are same length in assign_variables, unless expresssion file is being clever
            n = len(df.keys())
            shape = (n, elements / n if n else 0)
        else:
            shape = df.shape

        logger.debug(f"log_df delta_bytes: {util.INT(delta_bytes).rjust(12)} {table_name} {shape}")

        self.total_bytes = sum(self.tables.values())
        # logger.debug(f"log_df bytes: {util.INT(bytes)} total_bytes {util.INT(self.total_bytes)} {table_name}")

    def check_hwm(self, hwm_trace_label, cur_rss, cur_uss, total_bytes):
        """

        Parameters

        ----------
        hwm_trace_label: str
        cur_rss: int
            current absolute rss (easier to deal with this as absolute than as delta)
        total_bytes: int
            total current bytes summed for all active ChunkLedgers

        Returns
        -------

        """

        from_rss_monitor = total_bytes is None
        base_chunk_size = get_base_chunk_size()

        info = f"rss: {GB(cur_rss)} " \
               f"uss: {GB(cur_uss)} " \
               f"base_chunk_size: {GB(base_chunk_size)} " \
               f"op: {hwm_trace_label}"

        if total_bytes:
            info = f"bytes: {GB(total_bytes)} " + info

            if total_bytes > self.hwm_bytes['value']:
                self.hwm_bytes['value'] = total_bytes
                self.hwm_bytes['info'] = info

                # if this is a high water mark, check whether we are exceeding base_chunk_size
                if base_chunk_size > 0 and total_bytes > base_chunk_size:
                    out_of_chunk_memory(hwm_trace_label, rss=cur_rss, uss=cur_uss, bytes=total_bytes)

        if cur_rss > self.hwm_rss['value']:

            self.hwm_rss['value'] = cur_rss
            self.hwm_rss['info'] = info

            # if this is a high water mark, check whether we are exceeding base_chunk_size
            if base_chunk_size > 0 and cur_rss > base_chunk_size:
                out_of_chunk_memory(hwm_trace_label, rss=cur_rss, uss=cur_uss, bytes=total_bytes,
                                    from_rss_monitor=from_rss_monitor)

        if cur_uss > self.hwm_uss['value']:

            self.hwm_uss['value'] = cur_uss
            self.hwm_uss['info'] = info

            # if this is a high water mark, check whether we are exceeding base_chunk_size
            if base_chunk_size > 0 and cur_rss > base_chunk_size:
                out_of_chunk_memory(hwm_trace_label, rss=cur_rss, uss=cur_uss, bytes=total_bytes,
                                    from_rss_monitor=from_rss_monitor)

        mem.check_global_hwm('rss', cur_rss, hwm_trace_label)
        mem.check_global_hwm('uss', cur_uss, hwm_trace_label)
        mem.check_global_hwm('bytes', total_bytes, hwm_trace_label)

    def get_hwm_rss(self):
        with ledger_lock:
            net_rss = self.hwm_rss['value']
        return net_rss

    def get_hwm_uss(self):
        with ledger_lock:
            net_uss = self.hwm_uss['value']
        return net_uss

    def get_hwm_bytes(self):
        return self.hwm_bytes['value']


def _log_memory(trace_label, op_tag, table_name=None, df=None):
    """
    Parameters
    ----------
    trace_label :
        serves as a label for this nesting level of logging
    table_name : str
        name to use logging df, and which will be used in any subsequent calls reporting activity on the table
    df: numpy.ndarray, pandas.Series, pandas.DataFrame, or None
        table to log (or None if df was deleted)
    """

    if len(CHUNK_LEDGERS) == 0:
        logger.warning(f"log_df called without current chunker. Did you forget to call log_open?")
        return

    cur_chunker = CHUNK_LEDGERS[-1]

    hwm_trace_label = f"{trace_label}.{op_tag}"

    if table_name is not None:
        cur_chunker.log_df(table_name, df)

    mem.trace_memory_info(hwm_trace_label)

    total_bytes = sum([c.total_bytes for c in CHUNK_LEDGERS])

    rss, uss = mem.get_rss(uss=TRACK_USS)

    with ledger_lock:
        for c in CHUNK_LEDGERS:
            c.check_hwm(trace_label, rss, uss, total_bytes)


def log_rss(trace_label):

    op_tag = 'log_rss'

    if not chunk_training():
        mem.trace_memory_info(f"{trace_label}.{op_tag}")  #bug how expensive is this?
        return

    _log_memory(trace_label, op_tag)


def log_df(trace_label, table_name, df):

    if not chunk_training():
        return

    op = 'del' if df is None else 'add'
    op_tag = f"{op}.{table_name}"
    _log_memory(trace_label, op_tag, table_name, df)


class MemMonitor(threading.Thread):

    def __init__(self, trace_label, stop_snooping):
        self.trace_label = trace_label
        self.stop_snooping = stop_snooping
        threading.Thread.__init__(self)

    def run(self):
        log_rss(self.trace_label)
        while not self.stop_snooping.wait(timeout=mem.MEM_TICK_LEN):
            log_rss(self.trace_label)


class ChunkSizer(object):
    """
    ::
    """
    def __init__(self, chunk_tag, trace_label, num_choosers=0, chunk_size=0):

        self.depth = len(CHUNK_SIZERS) + 1
        self.cur_rss, self.cur_uss = mem.get_rss(force_garbage_collect=True, uss=TRACK_USS)

        if self.depth > 1:
            # nested chunkers should be unchunked
            assert chunk_size == 0

            # if we are in a nested call, then we must be in the scope of active Ledger
            # so any rss accumulated so far should be attributed to the parent active ledger
            assert len(CHUNK_SIZERS) == len(CHUNK_LEDGERS)
            parent = CHUNK_SIZERS[-1]
            assert parent.chunk_ledger is not None

            log_rss(trace_label)  # make sure we get at least one reading

        self.chunk_tag = chunk_tag
        self.trace_label = trace_label
        self.num_choosers = num_choosers
        self.chunk_size = chunk_size

        self.chunk_method = config.setting('chunk_method', DEFAULT_CHUNK_METHOD)
        assert self.chunk_method in CHUNK_METHODS, \
            f"chunk_method setting '{self.chunk_method}' not recognized. " \
            f"Should be one of: {CHUNK_METHODS}"

        min_available_chunk_ratio = config.setting('min_available_chunk_ratio', 0)
        assert 0 <= min_available_chunk_ratio <= 1, \
            f"min_available_chunk_ratio setting {min_available_chunk_ratio} is not in range [0..1]"
        self.min_chunk_size = chunk_size * min_available_chunk_ratio

        self.cum_overhead = {m: 0 for m in CHUNK_METHODS}
        self.rows_processed = 0
        self.rows_per_chunk = 0
        self.history = {}
        self.chunk_ledger = None

        # add self to CHUNK_SIZERS list before setting base_chunk_size (since we might be base chunker)
        CHUNK_SIZERS.append(self)

        self.base_chunk_size = CHUNK_SIZERS[0].chunk_size

        if self.base_chunk_size > 0 and self.cur_rss >= self.base_chunk_size:
            out_of_chunk_memory(f"{self.trace_label}.ChunkSizer.init", rss=self.cur_rss, uss=self.cur_uss)

        logger.debug(f"{self.trace_label} ChunkSizer.init "
                     f"base_chunk_size: {self.base_chunk_size} "
                     f"cur_rss: {self.cur_rss} cur_uss: {self.cur_uss}")

    def close(self):

        if self.history:

            if chunk_training():
                _HISTORIAN.write_history(self.history, self.chunk_tag)

        _chunk_sizer = CHUNK_SIZERS.pop()
        assert _chunk_sizer == self

    def available_headroom(self, xss=None):

        if xss is None:
            if chunk_metric(self.chunk_method) == 'uss':
                _, xss = mem.get_rss(uss=True)
            else:
                xss, _ = mem.get_rss(uss=False)

        headroom = self.base_chunk_size - xss

        # adjust deficient headroom to min_chunk_size
        if headroom < self.min_chunk_size:

            if self.base_chunk_size > 0:
                logger.warning(f"Not enough memory for minimum chunk_size without exceeding specified chunk_size."
                               f"available_headroom: {util.INT(headroom)} "
                               f"min_chunk_size: {util.INT(self.min_chunk_size)} "
                               f"base_chunk_size: {util.INT(self.base_chunk_size)}")

            headroom = self.min_chunk_size

        return headroom

    def initial_rows_per_chunk(self):

        # initialize values only needed if actually chunking

        if self.chunk_size == 0:
            rows_per_chunk = self.num_choosers
            estimated_number_of_chunks = 1
            self.initial_row_size = 0
        else:

            self.initial_row_size = _HISTORIAN.cached_row_size(self.chunk_tag, self.chunk_method)

            # cached_row_size if no cache

            assert len(CHUNK_LEDGERS) == 0, f"len(CHUNK_LEDGERS): {len(CHUNK_LEDGERS)}"
            if self.initial_row_size == 0:
                rows_per_chunk = min(self.num_choosers, get_default_initial_rows_per_chunk())
                estimated_number_of_chunks = None
            else:
                available_headroom = self.available_headroom()
                max_rows_per_chunk = np.maximum(int(available_headroom / self.initial_row_size), 1)
                rows_per_chunk = np.clip(max_rows_per_chunk, 1, self.num_choosers)
                estimated_number_of_chunks = math.ceil(self.num_choosers / rows_per_chunk)

        logger.debug(f"{self.trace_label} initial rows_per_chunk: {rows_per_chunk}")

        # cum_rows_per_chunk is out of phase with cum_chunk_size
        # since we won't know observed_chunk_size until AFTER yielding the chunk
        self.rows_per_chunk = rows_per_chunk
        self.rows_processed = rows_per_chunk

        return rows_per_chunk, estimated_number_of_chunks

    def adaptive_rows_per_chunk(self, i):
        # rows_processed is out of phase with cum_overhead
        # observed_overhead is the actual bytes/rss used top process chooser chunk with prev_rows_per_chunk rows

        if not LOG_SUBCHUNKS and (self.depth > 1):
            return 0, 1

        prev_rows_per_chunk = self.rows_per_chunk
        prev_rows_processed = self.rows_processed

        initial_rss = self.cur_rss
        initial_uss = self.cur_uss
        final_rss, final_uss = mem.get_rss(force_garbage_collect=True, uss=TRACK_USS)

        if chunk_metric(self.chunk_method) == 'uss':
            available_headroom = self.available_headroom(final_uss)
        else:
            available_headroom = self.available_headroom(final_rss)

        rows_remaining = self.num_choosers - prev_rows_processed

        if not chunk_training():
            observed_row_size = self.initial_row_size
        else:
            observed_overhead = {m: 0 for m in CHUNK_METHODS}

            # calculate overhead for this chunk iteration
            # use chunk_ledger to revise predicted row_size based on observed_overhead
            oh_rss = self.chunk_ledger.get_hwm_rss() - initial_rss
            oh_bytes = self.chunk_ledger.get_hwm_bytes()
            oh_uss = self.chunk_ledger.get_hwm_uss() - initial_uss

            observed_overhead[CHUNK_METHOD_RSS] = oh_rss
            observed_overhead[CHUNK_METHOD_BYTES] = oh_bytes
            observed_overhead[CHUNK_METHOD_USS] = oh_uss

            # CHUNK_ALGORITHM hybrid is average of rss/uss and bytes
            observed_overhead[CHUNK_METHOD_HYBRID_RSS] = int((oh_rss + oh_bytes) / 2)
            if TRACK_USS:
                observed_overhead[CHUNK_METHOD_HYBRID_USS] = int((oh_uss + oh_bytes) / 2)

            # update cumulative overhead
            for m in CHUNK_METHODS:
                self.cum_overhead[m] += observed_overhead[m]

            if prev_rows_processed:
                observed_row_size = math.ceil(self.cum_overhead[self.chunk_method] / prev_rows_processed)
            else:
                observed_row_size = 0

        # rows_per_chunk is closest number of chooser rows to achieve chunk_size without exceeding it
        if observed_row_size > 0:
            self.rows_per_chunk = int(available_headroom / observed_row_size)
        else:
            # they don't appear to have used any memory; increase cautiously in case small sample size was to blame
            self.rows_per_chunk = 2 * prev_rows_per_chunk

        self.rows_per_chunk = np.clip(self.rows_per_chunk, 1, rows_remaining)
        self.rows_processed = prev_rows_processed + self.rows_per_chunk
        estimated_number_of_chunks = i + math.ceil(rows_remaining / self.rows_per_chunk) if rows_remaining else i

        self.history.setdefault(C_TIME, []).append(datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S.%f"))
        self.history.setdefault(C_DEPTH, []).append(self.depth)
        for m in CHUNK_METHODS:
            self.history.setdefault(f'cum_overhead_{m}', []).append(self.cum_overhead[m])
        self.history.setdefault(C_NUM_ROWS, []).append(prev_rows_processed)
        self.history.setdefault('chunk', []).append(i)
        self.history.setdefault('chunk_size', []).append(self.chunk_size)
        self.history.setdefault('row_size', []).append(observed_row_size)

        self.history.setdefault('initial_rss', []).append(initial_rss)
        self.history.setdefault('final_rss', []).append(final_rss)

        # diagnostics not required by ChunkHistorian
        self.history.setdefault('available_headroom', []).append(available_headroom)
        if chunk_training():
            for m in CHUNK_METHODS:
                self.history.setdefault(f'observed_overhead_{m}', []).append(observed_overhead[m])

        self.history.setdefault('new_rows_per_chunk', []).append(self.rows_per_chunk)
        self.history.setdefault('estimated_num_chunks', []).append(estimated_number_of_chunks)

        self.cur_rss = final_rss
        self.cur_uss = final_uss

        logger.debug(f"#chunk_calc chunk: ChunkSizer.adaptive_rows_per_chunk "
                     f"base_chunk_size: {self.base_chunk_size} cur_rss: {self.cur_rss} "
                     f"prev headroom: {self.base_chunk_size - self.cur_rss}"
                     f"new headroom: {self.base_chunk_size - self.cur_rss}")

        history_df = pd.DataFrame.from_dict(self.history)
        logger.debug(f"ChunkSizer {self.chunk_tag}\n{history_df.transpose()}")

        # input()

        return self.rows_per_chunk, estimated_number_of_chunks

    @contextmanager
    def ledger(self):

        mem_monitor = None

        # nested chunkers should be unchunked
        if len(CHUNK_LEDGERS) > 0:
            assert self.chunk_size == 0

        with ledger_lock:
            self.chunk_ledger = ChunkLedger(self.trace_label, self.chunk_size, self.cur_rss, self.cur_uss)
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

            log_rss(f"{self.trace_label}.ledger.pre-yield")
            yield
            log_rss(f"{self.trace_label}.ledger.post-yield")

        finally:

            if mem_monitor is not None:

                if not mem_monitor.is_alive():
                    logger.error(f"mem_monitor for {self.trace_label} died!")
                    bug  # bug

                stop_snooping.set()
                while mem_monitor.is_alive():
                    logger.debug(f"{self.trace_label} waiting for mem_monitor thread to terminate")
                    mem_monitor.join(timeout=MEM_MONITOR_TICK)

            with ledger_lock:
                self.chunk_ledger.close()
                CHUNK_LEDGERS.pop()
                self.chunk_ledger = None


@contextmanager
def chunk_log(trace_label, chunk_tag=None):

    trace_label = f"{trace_label}.chunk_log"

    chunk_tag = chunk_tag or trace_label
    num_choosers = 0
    chunk_size = 0

    chunk_sizer = ChunkSizer(chunk_tag, trace_label, num_choosers, chunk_size)

    chunk_sizer.initial_rows_per_chunk()

    with chunk_sizer.ledger():

        yield

        chunk_sizer.adaptive_rows_per_chunk(1)

    chunk_sizer.close()


def adaptive_chunked_choosers(choosers, chunk_size, row_size, trace_label, chunk_tag=None):

    # generator to iterate over choosers

    chunk_tag = chunk_tag or trace_label

    num_choosers = len(choosers.index)
    assert num_choosers > 0
    assert chunk_size >= 0
    assert row_size >= 0

    logger.info(f"{trace_label} Running adaptive_chunked_choosers with {num_choosers} choosers")

    chunk_sizer = ChunkSizer(chunk_tag, trace_label, num_choosers, chunk_size)

    rows_per_chunk, estimated_number_of_chunks = chunk_sizer.initial_rows_per_chunk()

    i = offset = 0
    while offset < num_choosers:

        i += 1
        assert offset + rows_per_chunk <= num_choosers

        chunk_trace_label = trace_label_for_chunk(trace_label, chunk_size, i)

        with chunk_sizer.ledger():

            # grab the next chunk based on current rows_per_chunk
            chooser_chunk = choosers[offset: offset + rows_per_chunk]

            logger.info(f"Running chunk {i} of {estimated_number_of_chunks or '?'} "
                        f"with {len(chooser_chunk)} of {num_choosers} choosers")

            yield i, chooser_chunk, chunk_trace_label

            offset += rows_per_chunk

            rows_per_chunk, estimated_number_of_chunks = chunk_sizer.adaptive_rows_per_chunk(i)

    chunk_sizer.close()


def adaptive_chunked_choosers_and_alts(choosers, alternatives, chunk_size, row_size, trace_label, chunk_tag=None):
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
    -------
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
    assert choosers.index.equals(alternatives.index[~alternatives.index.duplicated(keep='first')])

    last_repeat = alternatives.index != np.roll(alternatives.index, -1)

    assert (num_choosers == 1) or choosers.index.equals(alternatives.index[last_repeat])
    assert 'pick_count' in alternatives.columns or choosers.index.name == alternatives.index.name
    assert choosers.index.name == alternatives.index.name

    logger.info(f"{trace_label} Running adaptive_chunked_choosers_and_alts "
                f"with {num_choosers} choosers and {num_alternatives} alternatives")

    chunk_sizer = ChunkSizer(chunk_tag, trace_label, num_choosers, chunk_size)
    rows_per_chunk, estimated_number_of_chunks = chunk_sizer.initial_rows_per_chunk()
    assert (rows_per_chunk > 0) and (rows_per_chunk <= num_choosers)

    # alt chunks boundaries are where index changes
    alt_ids = alternatives.index.values
    alt_chunk_ends = np.where(alt_ids[:-1] != alt_ids[1:])[0] + 1
    alt_chunk_ends = np.append([0], alt_chunk_ends)  # including the first to simplify indexing
    alt_chunk_ends = np.append(alt_chunk_ends, [len(alternatives.index)])  # end of final chunk

    i = offset = alt_offset = 0
    while offset < num_choosers:
        i += 1

        assert offset + rows_per_chunk <= num_choosers, \
            f"i {i} offset {offset} rows_per_chunk {rows_per_chunk} num_choosers {num_choosers}"

        chunk_trace_label = trace_label_for_chunk(trace_label, chunk_size, i)

        with chunk_sizer.ledger():

            chooser_chunk = choosers[offset: offset + rows_per_chunk]

            alt_end = alt_chunk_ends[offset + rows_per_chunk]
            alternative_chunk = alternatives[alt_offset: alt_end]

            assert len(chooser_chunk.index) == len(np.unique(alternative_chunk.index.values))
            assert (chooser_chunk.index == np.unique(alternative_chunk.index.values)).all()

            logger.info(f"Running chunk {i} of {estimated_number_of_chunks or '?'} "
                        f"with {len(chooser_chunk)} of {num_choosers} choosers")

            yield i, chooser_chunk, alternative_chunk, chunk_trace_label

            offset += rows_per_chunk
            alt_offset = alt_end

            rows_per_chunk, estimated_number_of_chunks = chunk_sizer.adaptive_rows_per_chunk(i)

    chunk_sizer.close()


def adaptive_chunked_choosers_by_chunk_id(choosers, chunk_size, row_size, trace_label, chunk_tag=None):
    # generator to iterate over choosers in chunk_size chunks
    # like chunked_choosers but based on chunk_id field rather than dataframe length
    # (the presumption is that choosers has multiple rows with the same chunk_id that
    # all have to be included in the same chunk)
    # FIXME - we pathologically know name of chunk_id col in households table

    chunk_tag = chunk_tag or trace_label

    num_choosers = choosers['chunk_id'].max() + 1
    assert num_choosers > 0

    chunk_sizer = ChunkSizer(chunk_tag, trace_label, num_choosers, chunk_size)

    rows_per_chunk, estimated_number_of_chunks = chunk_sizer.initial_rows_per_chunk()

    i = offset = 0
    while offset < num_choosers:

        i += 1
        assert offset + rows_per_chunk <= num_choosers

        chunk_trace_label = trace_label_for_chunk(trace_label, chunk_size, i)

        with chunk_sizer.ledger():

            chooser_chunk = choosers[choosers['chunk_id'].between(offset, offset + rows_per_chunk - 1)]

            logger.info(f"{trace_label} Running chunk {i} of {estimated_number_of_chunks or '?'} "
                        f"with {rows_per_chunk} of {num_choosers} choosers")

            yield i, chooser_chunk, chunk_trace_label

            offset += rows_per_chunk

            rows_per_chunk, estimated_number_of_chunks = chunk_sizer.adaptive_rows_per_chunk(i)

    chunk_sizer.close()
