# ActivitySim
# See full license in LICENSE.txt.
from builtins import input

import math
import logging
from collections import OrderedDict
from contextlib import contextmanager

import numpy as np
import pandas as pd

from . import util
from . import mem
from . import tracing

logger = logging.getLogger(__name__)

# dict of table_dicts keyed by trace_label
# table_dicts are dicts tuples of {table_name: (elements, bytes, mem), ...}
CHUNK_LOG = OrderedDict()

# array of chunk_size active CHUNK_LOG
CHUNK_SIZE = []

HWM = [{}]

INITIAL_ROWS_PER_CHUNK = 10
MAX_ROWSIZE_ERROR = 0.5  # estimated_row_size percentage error warning threshold
INTERACTIVE_TRACE_CHUNKING = False
INTERACTIVE_TRACE_CHUNK_WARNING = False

CHUNK_RSS = False
BYTES_PER_ELEMENT = 8

# chunk size being a bit opaque, it may be helpful to know the chunk size of a small sample run to titrate chunk_size
CHUNK_HISTORY = True  # always log chunk history even if chunk_size == 0


def GB(bytes):
    # symbols = ('', 'K', 'M', 'G', 'T')
    symbols = ('', ' KB', ' MB', ' GB', ' TB')
    fmt = "%.1f%s"
    for i, s in enumerate(symbols):
        units = 1 << i * 10
        if bytes < units * 1024:
            return fmt % (bytes / units, s)


def commas(x):
    x = int(x)
    if x < 10000:
        return str(x)
    result = ''
    while x >= 1000:
        x, r = divmod(x, 1000)
        result = ",%03d%s" % (r, result)
    return "%d%s" % (x, result)


class RowSizeEstimator(object):
    """
    Utility for estimating row_size
    """
    def __init__(self, trace_label):
        self.row_size = 0
        self.hwm = 0  # element count at high water mark
        self.hwm_tag = None  # tag that drove hwm (with ref count)
        self.trace_label = trace_label
        self.elements = {}
        self.tag_count = {}

    def add_elements(self, elements, tag):
        self.elements[tag] = elements
        self.tag_count[tag] = self.tag_count.setdefault(tag, 0) + 1  # number of times tag has been seen
        self.row_size += elements
        logger.debug(f"{self.trace_label} #chunk_calc {tag} {elements} ({self.row_size})")
        input("add_elements>") if INTERACTIVE_TRACE_CHUNKING else None

        if self.row_size > self.hwm:
            self.hwm = self.row_size
            # tag that drove hwm (with ref count)
            self.hwm_tag = f'{tag}_{self.tag_count[tag]}' if self.tag_count[tag] > 1 else tag

    def drop_elements(self, tag):
        self.row_size -= self.elements[tag]
        self.elements[tag] = 0
        logger.debug(f"{self.trace_label} #chunk_calc {tag} <drop> ({self.row_size})")

    def get_hwm(self):
        logger.debug(f"{self.trace_label} #chunk_calc hwm {self.hwm} after {self.hwm_tag}")
        input("get_hwm>") if INTERACTIVE_TRACE_CHUNKING else None
        return self.hwm


def get_high_water_mark(tag='elements'):

    # should always have at least the base chunker
    assert len(HWM) > 0

    hwm = HWM[-1]

    # hwm might be empty if there were no calls to log_df
    mark = hwm.get(tag).get('mark') if hwm else 0

    return mark


@contextmanager
def chunk_log(trace_label, chunk_size=0):
    log_open(trace_label, chunk_size)
    try:
        yield
    finally:
        log_close(trace_label)


def log_open(trace_label, chunk_size=0):

    # nested chunkers should be unchunked
    if len(CHUNK_LOG) > 0:
        assert chunk_size == 0
        assert trace_label not in CHUNK_LOG

    CHUNK_LOG[trace_label] = OrderedDict()
    CHUNK_SIZE.append(chunk_size)

    HWM.append({})


def log_close(trace_label):

    # they should be closing the last log opened (LIFO)
    assert CHUNK_LOG and next(reversed(CHUNK_LOG)) == trace_label

    # if we are closing base level chunker
    if len(CHUNK_LOG) == 1:
        log_write_hwm()

    label, _ = CHUNK_LOG.popitem(last=True)
    assert label == trace_label
    CHUNK_SIZE.pop()
    HWM.pop()


def log_df(trace_label, table_name, df):
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

    if df is None:
        # FIXME force_garbage_collect on delete?
        mem.force_garbage_collect()

    try:
        cur_chunker = next(reversed(CHUNK_LOG))
    except StopIteration:
        logger.warning(f"log_df called without current chunker. Did you forget to call log_open?")
        return

    if df is None:
        # remove table from log
        CHUNK_LOG.get(cur_chunker).pop(table_name)
        op = 'del'
        logger.debug(f"log_df {table_name} "
                     f"elements: {0} : {trace_label}")
    else:

        elements = util.iprod(df.shape)
        op = 'add'

        if isinstance(df, pd.Series):
            bytes = df.memory_usage(index=True)
        elif isinstance(df, pd.DataFrame):
            bytes = df.memory_usage(index=True).sum()
        elif isinstance(df, np.ndarray):
            bytes = df.nbytes
        else:
            logger.error("log_df %s unknown type: %s" % (table_name, type(df)))
            assert False

        CHUNK_LOG.get(cur_chunker)[table_name] = (elements, bytes)

        # log this df
        logger.debug(f"log_df {table_name} "
                     f"elements: {commas(elements)} "
                     f"bytes: {GB(bytes)} "
                     f"shape: {df.shape} : {trace_label}")

    hwm_trace_label = "%s.%s.%s" % (trace_label, op, table_name)
    mem.trace_memory_info(hwm_trace_label)

    total_elements, total_bytes = _chunk_totals()  # new chunk totals
    cur_rss = mem.get_rss()

    # - check high_water_marks
    info = f"elements: {commas(total_elements)} " \
           f"bytes: {GB(total_bytes)} " \
           f"rss: {GB(cur_rss)} " \
           f"chunk_size: {commas(CHUNK_SIZE[0])}"

    if INTERACTIVE_TRACE_CHUNKING:
        print(f"table_name {table_name} {df.shape if df is not None else 0}")
        print(f"table_name {table_name} {info}")
        input("log_df>")

    check_for_hwm('elements', total_elements, info, hwm_trace_label)
    check_for_hwm('bytes', total_bytes, info, hwm_trace_label)
    check_for_hwm('rss', cur_rss, info, hwm_trace_label)


def _chunk_totals():

    total_elements = 0
    total_bytes = 0
    for label in CHUNK_LOG:
        tables = CHUNK_LOG[label]
        for table_name in tables:
            elements, bytes = tables[table_name]
            total_elements += elements
            total_bytes += bytes

    return total_elements, total_bytes


def check_for_hwm(tag, value, info, trace_label):

    for d in HWM:

        hwm = d.setdefault(tag, {})

        if value > hwm.get('mark', 0):
            hwm['mark'] = value
            hwm['info'] = info
            hwm['trace_label'] = trace_label


def log_write_hwm():

    d = HWM[0]
    for tag in d:
        hwm = d[tag]
        logger.debug("#chunk_hwm high_water_mark %s: %s (%s) in %s" %
                     (tag, hwm['mark'], hwm['info'], hwm['trace_label']), )

    # - elements shouldn't exceed chunk_size of base chunker
    def check_chunk_size(hwm, chunk_size, label, max_leeway):
        elements = hwm['mark']
        if chunk_size and max_leeway and elements > chunk_size * max_leeway:  # too high
            # FIXME check for #warning in log - there is nothing the user can do about this
            logger.debug("#chunk_hwm #warning total_elements (%s) > %s (%s) %s : %s " %
                         (commas(elements), label, commas(chunk_size),
                          hwm['info'], hwm['trace_label']))

    # if we are in a chunker
    if len(HWM) > 1 and HWM[1]:
        assert 'elements' in HWM[1]  # expect an 'elements' hwm dict for base chunker
        hwm = HWM[1].get('elements')
        check_chunk_size(hwm, CHUNK_SIZE[0], 'chunk_size', max_leeway=1)


def write_history(caller, history, trace_label):

    observed_size = history.observed_chunk_size.sum()
    number_of_rows = history.rows_per_chunk.sum()
    observed_row_size = math.ceil(observed_size / number_of_rows)  # FIXME

    num_chunks = len(history)

    logger.info(f"#chunk_history {caller} {trace_label} "
                f"number_of_rows: {number_of_rows} "
                f"observed_row_size: {observed_row_size} "
                f"num_chunks: {num_chunks}")

    logger.debug(f"#chunk_history {caller} {trace_label}\n{history}")

    initial_row_size = history.row_size.values[0]
    if initial_row_size > 0:

        # if they provided an initial estimated row size, then report error

        error = (initial_row_size - observed_row_size) / observed_row_size
        percent_error = round(error * 100, 1)

        logger.info(f"#chunk_history {caller} {trace_label} "
                    f"initial_row_size: {initial_row_size} "
                    f"observed_row_size: {observed_row_size} "
                    f"percent_error: {percent_error}%")

        if abs(error) > MAX_ROWSIZE_ERROR:

            logger.warning(f"#chunk_history MAX_ROWSIZE_ERROR "
                           f"initial_row_size {initial_row_size} "
                           f"observed_row_size {observed_row_size} "
                           f"percent_error: {percent_error}% in {trace_label}")

            if INTERACTIVE_TRACE_CHUNK_WARNING:
                # for debugging adaptive chunking internals
                print(history)
                input(f"{trace_label} type any key to continue")


def adaptive_chunked_choosers(choosers, chunk_size, row_size, trace_label):

    # generator to iterate over choosers

    num_choosers = len(choosers.index)
    assert num_choosers > 0
    assert chunk_size >= 0
    assert row_size >= 0

    logger.info(f"Running adaptive_chunked_choosers with chunk_size {chunk_size} and {num_choosers} choosers")

    # FIXME do we care if it is an int?
    row_size = math.ceil(row_size)

    # #CHUNK_RSS
    mem.force_garbage_collect()
    initial_rss = mem.get_rss()
    logger.debug(f"#CHUNK_RSS initial_rss: {initial_rss}")

    if chunk_size == 0:
        assert row_size == 0  # we ignore this but make sure caller realizes that
        rows_per_chunk = num_choosers
        estimated_number_of_chunks = 1
    else:
        assert len(HWM) == 1, f"len(HWM): {len(HWM)}"
        if row_size == 0:
            rows_per_chunk = min(num_choosers, INITIAL_ROWS_PER_CHUNK)  # FIXME parameterize
            estimated_number_of_chunks = None
        else:
            max_rows_per_chunk = np.maximum(int(chunk_size / row_size), 1)
            logger.debug(f"#chunk_calc chunk: max rows_per_chunk {max_rows_per_chunk} based on row_size {row_size}")

            rows_per_chunk = np.clip(int(chunk_size / row_size), 1, num_choosers)
            estimated_number_of_chunks = math.ceil(num_choosers / rows_per_chunk)

        logger.debug(f"#chunk_calc chunk: initial rows_per_chunk {rows_per_chunk} based on row_size {row_size}")

    history = {}
    i = offset = 0
    while offset < num_choosers:

        assert offset + rows_per_chunk <= num_choosers

        chunk_trace_label = \
            tracing.extend_trace_label(trace_label, f'chunk_{i + 1}') if chunk_size > 0 else trace_label

        # grab the next chunk based on current rows_per_chunk
        chooser_chunk = choosers.iloc[offset: offset + rows_per_chunk]

        logger.info(f"Running chunk {i+1} of {estimated_number_of_chunks or '?'} "
                    f"with {len(chooser_chunk)} of {num_choosers} choosers")

        with chunk_log(trace_label, chunk_size):

            yield i+1, chooser_chunk, chunk_trace_label

            # get number of elements allocated during this chunk from the high water mark dict
            observed_chunk_size = get_high_water_mark()

            # #CHUNK_RSS
            observed_rss_size = (get_high_water_mark('rss') - initial_rss)
            observed_rss_size = math.ceil(observed_rss_size / BYTES_PER_ELEMENT)
            logger.debug(f"#CHUNK_RSS chunk {i+1} observed_chunk_size: {observed_chunk_size} "
                         f"observed_rss_size {observed_rss_size}")
            observed_rss_size = max(observed_rss_size, 0)
            if CHUNK_RSS:
                observed_chunk_size = observed_rss_size

        i += 1
        offset += rows_per_chunk
        rows_remaining = num_choosers - offset

        if CHUNK_HISTORY or chunk_size > 0:

            history.setdefault('chunk', []).append(i)
            history.setdefault('row_size', []).append(row_size)
            history.setdefault('rows_per_chunk', []).append(rows_per_chunk)
            history.setdefault('observed_chunk_size', []).append(observed_chunk_size)

            # revise predicted row_size based on observed_chunk_size
            row_size = math.ceil(observed_chunk_size / rows_per_chunk)

            # closest number of chooser rows to achieve chunk_size without exceeding it
            if row_size == 0:
                # they don't appear to have used any memory; increase cautiously in case small sample size was to blame
                if rows_per_chunk > INITIAL_ROWS_PER_CHUNK * 100:
                    rows_per_chunk = rows_remaining
                else:
                    rows_per_chunk = 10 * rows_per_chunk
            else:
                rows_per_chunk = int(chunk_size / row_size)
            rows_per_chunk = np.clip(rows_per_chunk, 1, rows_remaining)

            estimated_number_of_chunks = i + math.ceil(rows_remaining / rows_per_chunk) if rows_remaining else i

            history.setdefault('new_row_size', []).append(row_size)
            history.setdefault('new_rows_per_chunk', []).append(rows_per_chunk)
            history.setdefault('estimated_number_of_chunks', []).append(estimated_number_of_chunks)

    if history:
        history = pd.DataFrame.from_dict(history)
        write_history('adaptive_chunked_choosers', history, trace_label)


def adaptive_chunked_choosers_and_alts(choosers, alternatives, chunk_size, row_size, trace_label):
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

    # if not choosers.index.is_monotonic_increasing:
    #     logger.warning('sorting choosers because not monotonic increasing')
    #     choosers = choosers.sort_index()

    num_choosers = len(choosers.index)
    num_alternatives = len(alternatives.index)

    assert num_choosers > 0

    # alternatives index should match choosers (except with duplicate repeating alt rows)
    assert choosers.index.equals(alternatives.index[~alternatives.index.duplicated(keep='first')])

    last_repeat = alternatives.index != np.roll(alternatives.index, -1)

    assert (num_choosers == 1) or choosers.index.equals(alternatives.index[last_repeat])
    assert 'pick_count' in alternatives.columns or choosers.index.name == alternatives.index.name
    assert choosers.index.name == alternatives.index.name

    logger.info(f"Running adaptive_chunked_choosers_and_alts with chunk_size {chunk_size} "
                f"and {num_choosers} choosers and {num_alternatives} alternatives")

    # FIXME do we care if it is an int?
    row_size = math.ceil(row_size)

    # #CHUNK_RSS
    mem.force_garbage_collect()
    initial_rss = mem.get_rss()
    logger.debug(f"#CHUNK_RSS initial_rss: {initial_rss}")

    if chunk_size == 0:
        assert row_size == 0  # we ignore this but make sure caller realizes that
        rows_per_chunk = num_choosers
        estimated_number_of_chunks = 1
        row_size = 0
    else:
        assert len(HWM) == 1
        if row_size == 0:
            rows_per_chunk = min(num_choosers, INITIAL_ROWS_PER_CHUNK)  # FIXME parameterize
            estimated_number_of_chunks = None
        else:
            max_rows_per_chunk = np.maximum(int(chunk_size / row_size), 1)
            logger.debug(f"#chunk_calc chunk: max rows_per_chunk {max_rows_per_chunk} based on row_size {row_size}")

            rows_per_chunk = np.clip(int(chunk_size / row_size), 1, num_choosers)
            estimated_number_of_chunks = math.ceil(num_choosers / rows_per_chunk)
        logger.debug(f"#chunk_calc chunk: initial rows_per_chunk {rows_per_chunk} based on row_size {row_size}")

    assert (rows_per_chunk > 0) and (rows_per_chunk <= num_choosers)

    # alt chunks boundaries are where index changes
    alt_ids = alternatives.index.values
    alt_chunk_ends = np.where(alt_ids[:-1] != alt_ids[1:])[0] + 1
    alt_chunk_ends = np.append([0], alt_chunk_ends)  # including the first to simplify indexing
    alt_chunk_ends = np.append(alt_chunk_ends, [len(alternatives.index)])  # end of final chunk

    history = {}
    i = offset = alt_offset = 0
    while offset < num_choosers:
        assert offset + rows_per_chunk <= num_choosers

        chunk_trace_label = tracing.extend_trace_label(trace_label, f'chunk_{i + 1}') if chunk_size > 0 else trace_label

        chooser_chunk = choosers[offset: offset + rows_per_chunk]

        alt_end = alt_chunk_ends[offset + rows_per_chunk]
        alternative_chunk = alternatives[alt_offset: alt_end]

        assert len(chooser_chunk.index) == len(np.unique(alternative_chunk.index.values))
        assert (chooser_chunk.index == np.unique(alternative_chunk.index.values)).all()

        logger.info(f"Running chunk {i+1} of {estimated_number_of_chunks or '?'} "
                    f"with {len(chooser_chunk)} of {num_choosers} choosers")

        with chunk_log(trace_label, chunk_size):

            yield i+1, chooser_chunk, alternative_chunk, chunk_trace_label

            # get number of elements allocated during this chunk from the high water mark dict
            observed_chunk_size = get_high_water_mark()

            # #CHUNK_RSS
            observed_rss_size = (get_high_water_mark('rss') - initial_rss)
            observed_rss_size = math.ceil(observed_rss_size / BYTES_PER_ELEMENT)
            logger.debug(f"#CHUNK_RSS observed_chunk_size: {observed_chunk_size} observed_rss_size {observed_rss_size}")
            observed_rss_size = max(observed_rss_size, 0)
            if CHUNK_RSS:
                observed_chunk_size = observed_rss_size

        alt_offset = alt_end

        i += 1
        offset += rows_per_chunk
        rows_remaining = num_choosers - offset

        if CHUNK_HISTORY or chunk_size > 0:

            history.setdefault('chunk', []).append(i)
            history.setdefault('row_size', []).append(row_size)
            history.setdefault('rows_per_chunk', []).append(rows_per_chunk)
            history.setdefault('observed_chunk_size', []).append(observed_chunk_size)

            # revise predicted row_size based on observed_chunk_size
            row_size = math.ceil(observed_chunk_size / rows_per_chunk)

            # closest number of chooser rows to achieve chunk_size without exceeding it
            if row_size == 0:
                # they don't appear to have used any memory; increase cautiously in case small sample size was to blame
                if rows_per_chunk > INITIAL_ROWS_PER_CHUNK * 100:
                    rows_per_chunk = rows_remaining
                else:
                    rows_per_chunk = 10 * rows_per_chunk
            else:
                rows_per_chunk = int(chunk_size / row_size)
            rows_per_chunk = np.clip(rows_per_chunk, 1, rows_remaining)

            estimated_number_of_chunks = i + math.ceil(rows_remaining / rows_per_chunk) if rows_remaining else i

            history.setdefault('new_row_size', []).append(row_size)
            history.setdefault('new_rows_per_chunk', []).append(rows_per_chunk)
            history.setdefault('estimated_number_of_chunks', []).append(estimated_number_of_chunks)

    if history:
        history = pd.DataFrame.from_dict(history)
        write_history('adaptive_chunked_choosers_and_alts', history, trace_label)


def adaptive_chunked_choosers_by_chunk_id(choosers, chunk_size, row_size, trace_label):
    # generator to iterate over choosers in chunk_size chunks
    # like chunked_choosers but based on chunk_id field rather than dataframe length
    # (the presumption is that choosers has multiple rows with the same chunk_id that
    # all have to be included in the same chunk)
    # FIXME - we pathologically know name of chunk_id col in households table

    num_choosers = choosers['chunk_id'].max() + 1
    assert num_choosers > 0

    # FIXME do we care if it is an int?
    row_size = math.ceil(row_size)

    # #CHUNK_RSS
    mem.force_garbage_collect()
    initial_rss = mem.get_rss()
    logger.debug(f"#CHUNK_RSS initial_rss: {initial_rss}")

    if chunk_size == 0:
        assert row_size == 0  # we ignore this but make sure caller realizes that
        rows_per_chunk = num_choosers
        estimated_number_of_chunks = 1
    else:
        assert len(HWM) == 1

        if row_size == 0:
            rows_per_chunk = min(num_choosers, INITIAL_ROWS_PER_CHUNK)  # FIXME parameterize
            estimated_number_of_chunks = None
            logger.debug(f"#chunk_calc chunk: initial rows_per_chunk {rows_per_chunk} "
                         f"based on INITIAL_ROWS_PER_CHUNK {INITIAL_ROWS_PER_CHUNK}")

        else:
            max_rpc = np.maximum(int(chunk_size / row_size), 1)
            logger.debug(f"#chunk_calc chunk: max rows_per_chunk {max_rpc} based on row_size {row_size}")

            rows_per_chunk = np.clip(int(chunk_size / row_size), 1, num_choosers)
            estimated_number_of_chunks = math.ceil(num_choosers / rows_per_chunk)

            logger.debug(f"#chunk_calc chunk: initial rows_per_chunk {rows_per_chunk} based on row_size {row_size}")

    history = {}
    i = offset = 0
    while offset < num_choosers:

        assert offset + rows_per_chunk <= num_choosers

        chunk_trace_label = \
            tracing.extend_trace_label(trace_label, f'chunk_{i + 1}') if chunk_size > 0 else trace_label

        chooser_chunk = choosers[choosers['chunk_id'].between(offset, offset + rows_per_chunk - 1)]

        logger.info(f"Running chunk {i+1} of {estimated_number_of_chunks or '?'} "
                    f"with {rows_per_chunk} of {num_choosers} choosers")

        with chunk_log(trace_label, chunk_size):

            yield i+1, chooser_chunk, chunk_trace_label

            # get number of elements allocated during this chunk from the high water mark dict
            observed_chunk_size = get_high_water_mark()

            # #CHUNK_RSS
            observed_rss_size = (get_high_water_mark('rss') - initial_rss)
            observed_rss_size = math.ceil(observed_rss_size / BYTES_PER_ELEMENT)
            logger.debug(f"#CHUNK_RSS observed_chunk_size: {observed_chunk_size} observed_rss_size {observed_rss_size}")
            observed_rss_size = max(observed_rss_size, 0)
            if CHUNK_RSS:
                observed_chunk_size = observed_rss_size

        i += 1
        offset += rows_per_chunk
        rows_remaining = num_choosers - offset

        if CHUNK_HISTORY or chunk_size > 0:

            history.setdefault('chunk', []).append(i)
            history.setdefault('row_size', []).append(row_size)
            history.setdefault('rows_per_chunk', []).append(rows_per_chunk)
            history.setdefault('observed_chunk_size', []).append(observed_chunk_size)

            # revise predicted row_size based on observed_chunk_size
            row_size = math.ceil(observed_chunk_size / rows_per_chunk)

            if row_size > 0:
                # closest number of chooser rows to achieve chunk_size without exceeding it
                rows_per_chunk = int(chunk_size / row_size)
            if row_size == 0:
                # they don't appear to have used any memory; increase cautiously in case small sample size was to blame
                if rows_per_chunk > INITIAL_ROWS_PER_CHUNK * 100:
                    rows_per_chunk = rows_remaining
                else:
                    rows_per_chunk = 10 * rows_per_chunk

            rows_per_chunk = np.clip(rows_per_chunk, 1, rows_remaining)

            estimated_number_of_chunks = i + math.ceil(rows_remaining / rows_per_chunk) if rows_remaining else i

            history.setdefault('new_row_size', []).append(row_size)
            history.setdefault('new_rows_per_chunk', []).append(rows_per_chunk)
            history.setdefault('estimated_number_of_chunks', []).append(estimated_number_of_chunks)

    if history:
        history = pd.DataFrame.from_dict(history)
        write_history('adaptive_chunked_choosers_by_chunk_id', history, trace_label)
