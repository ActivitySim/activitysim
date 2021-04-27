# ActivitySim
# See full license in LICENSE.txt.
from builtins import input

import math
import logging
import psutil
import threading
import _thread
import os

from contextlib import contextmanager

import numpy as np
import pandas as pd

from . import util
from . import mem
from . import tracing
from . import config
from . import inject

from .util import GB
from .util import INT_STR

logger = logging.getLogger(__name__)

CHUNK_LEDGERS = []
CHUNK_SIZERS = []

MAX_ROWSIZE_ERROR = 0.5  # estimated_row_size percentage error warning threshold

BYTES_PER_ELEMENT = 8


MEM_MONITOR_TICK = 1  # in seconds
ENABLE_MEMORY_MONITOR = True

# FIXME should we copy chooser chunk after slicing
# presumably not needed as long as we check if chooser_chunk_is_view before and after yield
COPY_CHOOSER_CHUNKS = False

# FIXME should we update this every chunk? Or does it make available_chunk_size too jittery?
RESET_RSS_BASELINE_FOR_EACH_CHUNK = True

# FIXME should we add chunk number to trace_label when chunking
TRACE_LABEL_CHOOSER_CHUNK_NUM = False

CACHE_HISTORY = True


HWM = {}

ledger_lock = threading.Lock()

C_CHUNK_TAG = 'tag'
C_DEPTH = 'depth'
C_OVERHEAD = 'cum_overhead'
C_OVERHEAD_RSS = 'cum_overhead_rss'
C_OVERHEAD_BYTES = 'cum_overhead_bytes'
C_NUM_ROWS = 'rows_processed'

C_NUM_CHUNKS = C_CHUNK = 'chunk'  # chunk num becomes num_chunks when we drop all but last
C_CHUNK_SIZE = 'chunk_size'
CHUNK_HISTORY_COLUMNS = [C_CHUNK_TAG, C_DEPTH, C_OVERHEAD, C_OVERHEAD_RSS, C_OVERHEAD_BYTES,
                         'observed_row_size', 'observed_row_size_rss', 'observed_row_size_bytes',
                         'initial_rss', 'final_rss',
                         C_NUM_ROWS, C_CHUNK_SIZE, C_NUM_CHUNKS]


def chunk_logging():
    return len(CHUNK_LEDGERS) > 0


def get_rss(force_garbage_collect=False):

    if force_garbage_collect:
        mem.force_garbage_collect()

    return psutil.Process().memory_info().rss


def trace_label_for_chunk(trace_label, chunk_size, i):
    # add chunk_num to trace_label if TRACE_LABEL_CHOOSER_CHUNK_NUM flag
    if TRACE_LABEL_CHOOSER_CHUNK_NUM and chunk_size > 0:
        trace_label = tracing.extend_trace_label(trace_label, f'chunk_{i}')
    return trace_label


def get_base_chunk_size():
    assert len(CHUNK_SIZERS) > 0
    return CHUNK_SIZERS[0].chunk_size


def log_write_hwm():

    for tag, hwm in HWM.items():
        hwm = HWM[tag]
        logger.debug("high_water_mark %s: %s (%s) in %s" %
                     (tag, GB(hwm['mark']), hwm['info'], hwm['trace_label']), )

    print(f"\n")
    for tag, hwm in HWM.items():
        hwm = HWM[tag]
        print(f"chunk_hwm high_water_mark {tag}: {GB(hwm['mark'])} in {hwm['trace_label']}")


def check_global_hwm(tag, value, info, trace_label):

    if value:
        hwm = HWM.setdefault(tag, {})

        if value > hwm.get('mark', 0):
            hwm['mark'] = value
            hwm['info'] = info
            hwm['trace_label'] = trace_label


def out_of_chunk_memory(msg, bytes=None, rss=None, from_rss_monitor=False):

    MAX_OVERDRAFT_RATIO = 1.2

    if from_rss_monitor:
        return

    bytes = bytes or 0

    base_chunk_size = get_base_chunk_size()
    assert base_chunk_size > 0
    panic_threshold = base_chunk_size * MAX_OVERDRAFT_RATIO

    if max(bytes, rss) > panic_threshold:

        # if things are that dire, force_garbage_collect
        rss_after_gc = get_rss(force_garbage_collect=True)

        if rss_after_gc > panic_threshold:

            logger.warning(f"out_of_chunk_memory: base_chunk_size: {base_chunk_size} cur_rss: {rss_after_gc} {msg}")

            # for s in CHUNK_SIZERS[::-1]:
            #     logger.error(f"CHUNK_SIZER {s.trace_label}")
            #
            # for s in CHUNK_LEDGERS[::-1]:
            #     logger.error(f"CHUNK_LOGGER {s.trace_label}")
            #     logger.error(f"--- hwm_bytes {INT_STR(s.hwm_bytes['value'])} {s.hwm_bytes['info']}")
            #     logger.error(f"--- hwm_rss {INT_STR(s.hwm_rss['value'])} {s.hwm_rss['info']}")


class ChunkHistorian(object):
    """
    Utility for estimating row_size
    """
    def __init__(self):

        self.chunk_log_path = None

        self.have_cached_history = None
        self.cached_history_df = None

        self.DEFAULT_INITIAL_ROWS_PER_CHUNK = 10
        self.CACHE_FILE_NAME = 'chunk_log.csv'
        self.LOG_FILE_NAME = 'chunk_log.csv'

    def load_cached_history(self):

        chunk_cache_path = os.path.join(config.get_cache_dir(), self.CACHE_FILE_NAME)

        logger.debug(f"ChunkHistorian load_cached_history chunk_cache_path {chunk_cache_path}")

        if CACHE_HISTORY and os.path.exists(chunk_cache_path):
            logger.debug(f"ChunkHistorian load_cached_history reading cached chunk history from {self.CACHE_FILE_NAME}")
            df = pd.read_csv(chunk_cache_path, comment='#')

            missing_columns = [c for c in CHUNK_HISTORY_COLUMNS if c not in df]
            if missing_columns:
                logger.error(f"cached chunk log is missing columns: {missing_columns}")

            df = df[df[C_DEPTH] == 1]

            # cached_row_size method handles duplicates
            # if df[C_CHUNK_TAG].duplicated().any():
            #     logger.warning(f"ChunkHistorian load_cached_history found duplicate {C_CHUNK_TAG} rows in cache")
            #     df = df[~df[C_CHUNK_TAG].duplicated(keep='last')]

            self.cached_history_df = df
            self.have_cached_history = True
        else:
            self.have_cached_history = False

    def cached_row_size(self, chunk_tag):

        if self.have_cached_history is None:
            self.load_cached_history()

        row_size = 0  # this is out fallback

        if self.have_cached_history:

            try:
                df = self.cached_history_df[self.cached_history_df[C_CHUNK_TAG] == chunk_tag]
                if len(df) > 0:

                    if len(df) > 1:
                        logger.info(f"ChunkHistorian aggregating {len(df)} multiple rows for {chunk_tag}")

                    overhead = df[C_OVERHEAD].sum()
                    num_rows = df[C_NUM_ROWS].sum()
                    if num_rows > 0:
                        row_size = overhead / num_rows

            except Exception as e:
                logger.warning(f"ChunkHistorian Error calculating row_size for {chunk_tag}")
                raise e

        return row_size

    def write_history(self, history_df, chunk_tag):

        # just want the last, most up to date row
        history_df = history_df.tail(1)

        history_df[C_CHUNK_TAG] = chunk_tag

        missing_columns = [c for c in CHUNK_HISTORY_COLUMNS if c not in history_df]
        if missing_columns:
            logger.error(f"ChunkHistorian.write_history: history_df is missing columns: {missing_columns}")
        history_df = history_df[CHUNK_HISTORY_COLUMNS]

        if self.chunk_log_path is None:
            self.chunk_log_path = config.log_file_path(self.LOG_FILE_NAME)

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
    def __init__(self, trace_label, chunk_size, baseline_rss):
        self.trace_label = trace_label
        self.chunk_size = chunk_size

        self.last_rss = baseline_rss

        self.tables = {}
        self.hwm_bytes = {'value': 0, 'info': f'{trace_label}.init'}
        self.hwm_rss = {'value': baseline_rss, 'info': f'{trace_label}.init'}
        self.last_op = None
        self.total_bytes = 0

    def close(self):
        logger.debug(f"ChunkLedger.close hwm_bytes: {self.hwm_bytes.get('value', 0)} {self.hwm_bytes['info']}")
        logger.debug(f"ChunkLedger.close hwm_rss {self.hwm_rss['value']} {self.hwm_rss['info']}")

    def log_df(self, trace_label, table_name, df):

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
            shape = (0, 0)
        elif isinstance(df, dict):
            # ordinarily all elements are same length in assign_variables, unless expresssion file is being clever
            n = len(df.keys())
            shape = (n, elements / n if n else 0)
        else:
            shape = df.shape

        cur_rss = get_rss()
        delta_rss = cur_rss - self.last_rss
        self.last_rss = cur_rss

        def f(n):
            return INT_STR(n).rjust(12)

        logger.debug(f"log_df delta_bytes: {f(delta_bytes)} delta_rss: {f(delta_rss)} {table_name} {shape}")

        self.total_bytes = sum(self.tables.values())
        # logger.debug(f"log_df bytes: {INT_STR(bytes)} total_bytes {INT_STR(self.total_bytes)} {table_name}")

    def check_hwm(self, hwm_trace_label, cur_rss, total_bytes=None):
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
               f"base_chunk_size: {GB(base_chunk_size)} " \
               f"op: {hwm_trace_label}"

        if total_bytes:
            info = f"bytes: {GB(total_bytes)} " + info

            if total_bytes > self.hwm_bytes['value']:
                self.hwm_bytes['value'] = total_bytes
                self.hwm_bytes['info'] = info

                # if this is a high water mark, check whether we are exceeding base_chunk_size
                if base_chunk_size > 0 and total_bytes > base_chunk_size:
                    out_of_chunk_memory(hwm_trace_label, rss=cur_rss, bytes=total_bytes)

            self.last_op = hwm_trace_label

        if cur_rss > self.hwm_rss['value']:

            self.hwm_rss['value'] = cur_rss
            self.hwm_rss['info'] = info

            # if this is a high water mark, check whether we are exceeding base_chunk_size
            if base_chunk_size > 0 and cur_rss > base_chunk_size:
                out_of_chunk_memory(hwm_trace_label, rss=cur_rss, bytes=total_bytes, from_rss_monitor=from_rss_monitor)

        check_global_hwm('rss', cur_rss, info, hwm_trace_label)
        check_global_hwm('bytes', total_bytes, info, hwm_trace_label)

    def get_hwm_gross_rss(self):
        with ledger_lock:
            net_rss = self.hwm_rss['value']
        return net_rss

    def get_hwm_rss(self):
        with ledger_lock:
            net_rss = self.hwm_rss['value']
        return net_rss

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

    cur_chunker.log_df(hwm_trace_label, table_name, df)

    mem.trace_memory_info(hwm_trace_label)

    total_bytes = sum([c.total_bytes for c in CHUNK_LEDGERS])
    cur_rss = get_rss()

    with ledger_lock:
        for c in CHUNK_LEDGERS:
            c.check_hwm(trace_label, cur_rss, total_bytes)


def log_rss(trace_label):
    _log_memory(trace_label, 'log_rss')


def log_df(trace_label, table_name, df):
    op = 'del' if df is None else 'add'
    _log_memory(trace_label, f"{op}.{table_name}", table_name, df)


class MemMonitor(threading.Thread):

    def __init__(self, trace_label, stop_snooping):
        self.trace_label = trace_label
        self.stop_snooping = stop_snooping
        threading.Thread.__init__(self)

    def run(self):
        i = 1

        log_rss(f"{self.trace_label}.tick_{i}")
        while not self.stop_snooping.wait(timeout=MEM_MONITOR_TICK):
            log_rss(f"{self.trace_label}.{i}")
            i += 1


class ChunkSizer(object):
    """
    ::
    """
    def __init__(self, chunk_tag, trace_label, num_choosers=0, chunk_size=0):

        self.depth = len(CHUNK_SIZERS) + 1
        self.cur_rss = get_rss(force_garbage_collect=True)

        if self.depth > 1:
            # nested chunkers should be unchunked
            assert chunk_size == 0

            # if we are in a nested call, then we must be in the scope of active Ledger
            # so any rss accumulated so far should be attributed to the parent active ledger
            assert len(CHUNK_SIZERS) == len(CHUNK_LEDGERS)
            parent = CHUNK_SIZERS[-1]
            assert parent.chunk_ledger is not None

            log_rss(trace_label)

        self.chunk_tag = chunk_tag
        self.trace_label = trace_label

        self.num_choosers = num_choosers

        self.history = {}
        self.chunk_ledger = None
        self.chunk_size = chunk_size

        CHUNK_SIZERS.append(self)

        self.base_chunk_size = CHUNK_SIZERS[0].chunk_size

        if self.base_chunk_size > 0 and self.cur_rss >= self.base_chunk_size:
            out_of_chunk_memory(f"{self.trace_label}.set_base_rss", rss=self.cur_rss)

        logger.debug(f"#chunk_calc chunk: ChunkSizer.init "
                     f"base_chunk_size: {self.base_chunk_size} cur_rss: {self.cur_rss} "
                     f"headroom: {self.base_chunk_size - self.cur_rss}")

    def close(self):

        if self.history:

            history_df = pd.DataFrame.from_dict(self.history)
            logger.debug(f"ChunkSizer {self.trace_label}\n{history_df.transpose()}")

            _HISTORIAN.write_history(history_df, self.chunk_tag)

        _chunk_sizer = CHUNK_SIZERS.pop()
        assert _chunk_sizer == self

    def initial_rows_per_chunk(self):

        # initialize values only needed if actually chunking

        self.cum_overhead_rss = 0
        self.cum_overhead_bytes = 0
        self.cum_overhead = 0  # hybrid based on greater of rss and bytes on each iteration
        self.rows_processed = 0

        self.initial_row_size = _HISTORIAN.cached_row_size(self.chunk_tag) if self.chunk_size > 0 else 0

        #########

        if self.chunk_size == 0:
            assert self.initial_row_size == 0  # we ignore this but make sure caller realizes that
            rows_per_chunk = self.num_choosers
            estimated_number_of_chunks = 1
        else:

            assert len(CHUNK_LEDGERS) == 0, f"len(CHUNK_LEDGERS): {len(CHUNK_LEDGERS)}"
            if self.initial_row_size == 0:
                rows_per_chunk = min(self.num_choosers, _HISTORIAN.DEFAULT_INITIAL_ROWS_PER_CHUNK)
                estimated_number_of_chunks = None
            else:
                available_chunk_size = self.base_chunk_size - self.cur_rss

                max_rows_per_chunk = np.maximum(int(available_chunk_size / self.initial_row_size), 1)
                logger.debug(f"#chunk_calc chunk: max rows_per_chunk {max_rows_per_chunk} "
                             f"based on initial_row_size {self.initial_row_size}")

                rows_per_chunk = np.clip(max_rows_per_chunk, 1, self.num_choosers)
                estimated_number_of_chunks = math.ceil(self.num_choosers / rows_per_chunk)

            logger.debug(f"#chunk_calc chunk: initial rows_per_chunk {rows_per_chunk}")

        # cum_rows_per_chunk is out of phase with cum_chunk_size
        # since we won't know observed_chunk_size until AFTER yielding the chunk
        self.rows_per_chunk = rows_per_chunk
        self.rows_processed = rows_per_chunk

        return rows_per_chunk, estimated_number_of_chunks

    def adaptive_rows_per_chunk(self, i):
        # rows_processed is out of phase with cum_overhead
        # observed_overhead is the overhead for processing chooser chunk with prev_rows_per_chunk rows

        prev_rows_per_chunk = self.rows_per_chunk
        prev_rows_processed = self.rows_processed

        initial_rss = self.cur_rss
        final_rss = get_rss(force_garbage_collect=True)

        new_cur_rss = final_rss if RESET_RSS_BASELINE_FOR_EACH_CHUNK else initial_rss

        rows_remaining = self.num_choosers - prev_rows_processed

        # use chunk_ledger to revise predicted row_size based on observed_overhead
        observed_overhead_rss = self.chunk_ledger.get_hwm_rss() - initial_rss
        self.cum_overhead_rss += observed_overhead_rss
        observed_row_size_rss = math.ceil(self.cum_overhead_rss / prev_rows_processed)

        observed_overhead_bytes = self.chunk_ledger.get_hwm_bytes()
        self.cum_overhead_bytes += observed_overhead_bytes
        observed_row_size_bytes = math.ceil(self.cum_overhead_bytes / prev_rows_processed)

        observed_overhead = max(observed_overhead_rss, observed_overhead_bytes)
        self.cum_overhead += observed_overhead  # could be hybrid of rss and bytes
        observed_row_size = math.ceil(self.cum_overhead / prev_rows_processed)

        # rows_per_chunk is closest number of chooser rows to achieve chunk_size without exceeding it
        available_chunk_size = self.base_chunk_size - new_cur_rss
        if observed_row_size > 0:
            self.rows_per_chunk = int(available_chunk_size / observed_row_size)
        else:
            # they don't appear to have used any memory; increase cautiously in case small sample size was to blame
            self.rows_per_chunk = 10 * prev_rows_per_chunk
            logger.warning(f"{self.trace_label} adaptive_rows_per_chunk_{i} observed_row_size == 0")

        self.rows_per_chunk = np.clip(self.rows_per_chunk, 1, rows_remaining)
        self.rows_processed = prev_rows_processed + self.rows_per_chunk
        estimated_number_of_chunks = i + math.ceil(rows_remaining / self.rows_per_chunk) if rows_remaining else i

        self.history.setdefault(C_DEPTH, []).append(self.depth)
        self.history.setdefault(C_OVERHEAD, []).append(self.cum_overhead)
        self.history.setdefault(C_OVERHEAD_RSS, []).append(self.cum_overhead_rss)
        self.history.setdefault(C_OVERHEAD_BYTES, []).append(self.cum_overhead_bytes)
        self.history.setdefault(C_NUM_ROWS, []).append(prev_rows_processed)
        self.history.setdefault(C_CHUNK, []).append(i)
        self.history.setdefault(C_CHUNK_SIZE, []).append(self.chunk_size)

        # for legibility
        self.history.setdefault('observed_row_size', []).append(observed_row_size)
        self.history.setdefault('observed_row_size_rss', []).append(observed_row_size_rss)
        self.history.setdefault('observed_row_size_bytes', []).append(observed_row_size_bytes)

        self.history.setdefault('initial_rss', []).append(initial_rss)
        self.history.setdefault('final_rss', []).append(final_rss)

        # diagnostics not required by ChunkHistorian
        self.history.setdefault('available_chunk_size', []).append(available_chunk_size)
        self.history.setdefault('prev_rows_per_chunk', []).append(prev_rows_per_chunk)
        self.history.setdefault('observed_overhead', []).append(observed_overhead)
        self.history.setdefault('observed_overhead_rss', []).append(observed_overhead_rss)
        self.history.setdefault('observed_overhead_bytes', []).append(observed_overhead_bytes)
        self.history.setdefault('new_rows_per_chunk', []).append(self.rows_per_chunk)
        self.history.setdefault('estimated_num_chunks', []).append(estimated_number_of_chunks)

        self.cur_rss = new_cur_rss

        logger.debug(f"#chunk_calc chunk: ChunkSizer.adaptive_rows_per_chunk "
                     f"base_chunk_size: {self.base_chunk_size} cur_rss: {self.cur_rss} "
                     f"prev headroom: {self.base_chunk_size - self.cur_rss}"
                     f"new headroom: {self.base_chunk_size - self.cur_rss}")

        return self.rows_per_chunk, estimated_number_of_chunks

    @contextmanager
    def ledger(self):

        mem_monitor = None

        # nested chunkers should be unchunked
        if len(CHUNK_LEDGERS) > 0:
            assert self.chunk_size == 0

        with ledger_lock:
            self.chunk_ledger = ChunkLedger(self.trace_label, self.chunk_size, self.cur_rss)
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

            yield

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
    """
    context manager to provide chunk logging for unchunked choosers
    without the overhead of a adaptive_chunked_choosers generator
    """

    chunk_tag = chunk_tag or trace_label

    chunk_sizer = ChunkSizer(chunk_tag, trace_label)
    with chunk_sizer.ledger():

        yield

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

            if COPY_CHOOSER_CHUNKS:
                chooser_chunk = chooser_chunk.copy()
            chooser_chunk_is_view = chooser_chunk._is_view
            if not chooser_chunk_is_view:
                log_df(trace_label, 'chooser_chunk', chooser_chunk)

            logger.info(f"Running chunk {i} of {estimated_number_of_chunks or '?'} "
                        f"with {len(chooser_chunk)} of {num_choosers} choosers")

            yield i, chooser_chunk, chunk_trace_label

            assert chooser_chunk._is_view == chooser_chunk_is_view
            if not chooser_chunk_is_view:
                del chooser_chunk
                log_df(trace_label, 'chooser_chunk', None)

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
            if COPY_CHOOSER_CHUNKS:
                chooser_chunk = chooser_chunk.copy()
            chooser_chunk_is_view = chooser_chunk._is_view
            if not chooser_chunk_is_view:
                log_df(trace_label, 'chooser_chunk', chooser_chunk)

            alt_end = alt_chunk_ends[offset + rows_per_chunk]
            alternative_chunk = alternatives[alt_offset: alt_end]
            if COPY_CHOOSER_CHUNKS:
                alternative_chunk = alternative_chunk.copy()
            alternative_chunk_is_view = alternative_chunk._is_view
            if not alternative_chunk_is_view:
                log_df(trace_label, 'alternative_chunk', alternative_chunk)

            assert len(chooser_chunk.index) == len(np.unique(alternative_chunk.index.values))
            assert (chooser_chunk.index == np.unique(alternative_chunk.index.values)).all()

            logger.info(f"Running chunk {i} of {estimated_number_of_chunks or '?'} "
                        f"with {len(chooser_chunk)} of {num_choosers} choosers")

            yield i, chooser_chunk, alternative_chunk, chunk_trace_label

            assert alternative_chunk._is_view == alternative_chunk_is_view
            if not alternative_chunk_is_view:
                del alternative_chunk
                log_df(trace_label, 'alternative_chunk', None)

            assert chooser_chunk._is_view == chooser_chunk_is_view
            if not chooser_chunk_is_view:
                del chooser_chunk
                log_df(trace_label, 'chooser_chunk', None)

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
    chunk_trace_label = trace_label
    while offset < num_choosers:

        i += 1
        assert offset + rows_per_chunk <= num_choosers

        chunk_trace_label = trace_label_for_chunk(trace_label, chunk_size, i)

        with chunk_sizer.ledger():

            chooser_chunk = choosers[choosers['chunk_id'].between(offset, offset + rows_per_chunk - 1)]

            if COPY_CHOOSER_CHUNKS:
                chooser_chunk = chooser_chunk.copy()
            chooser_chunk_is_view = chooser_chunk._is_view
            if not chooser_chunk_is_view:
                log_df(trace_label, 'chooser_chunk', chooser_chunk)

            logger.info(f"{trace_label} Running chunk {i} of {estimated_number_of_chunks or '?'} "
                        f"with {rows_per_chunk} of {num_choosers} choosers")

            yield i, chooser_chunk, chunk_trace_label

            assert chooser_chunk._is_view == chooser_chunk_is_view
            if not chooser_chunk_is_view:
                del chooser_chunk
                log_df(trace_label, 'chooser_chunk', None)

            offset += rows_per_chunk

            rows_per_chunk, estimated_number_of_chunks = chunk_sizer.adaptive_rows_per_chunk(i)

    chunk_sizer.close()
