# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402

from builtins import input

import logging
from collections import OrderedDict

import numpy as np
import pandas as pd

from . import util
from . import mem

logger = logging.getLogger(__name__)

# dict of table_dicts keyed by trace_label
# table_dicts are dicts tuples of {table_name: (elements, bytes, mem), ...}
CHUNK_LOG = OrderedDict()

# array of chunk_size active CHUNK_LOG
CHUNK_SIZE = []
EFFECTIVE_CHUNK_SIZE = []

HWM = [{}]


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


def log_open(trace_label, chunk_size, effective_chunk_size):

    # nested chunkers should be unchunked
    if len(CHUNK_LOG) > 0:
        assert chunk_size == 0
        assert trace_label not in CHUNK_LOG

    logger.debug("log_open chunker %s chunk_size %s effective_chunk_size %s" %
                 (trace_label, commas(chunk_size), commas(effective_chunk_size)))

    CHUNK_LOG[trace_label] = OrderedDict()
    CHUNK_SIZE.append(chunk_size)
    EFFECTIVE_CHUNK_SIZE.append(effective_chunk_size)

    HWM.append({})


def log_close(trace_label):

    assert CHUNK_LOG and next(reversed(CHUNK_LOG)) == trace_label

    logger.debug("log_close %s" % trace_label)

    # if we are closing base level chunker
    if len(CHUNK_LOG) == 1:
        log_write_hwm()

    label, _ = CHUNK_LOG.popitem(last=True)
    assert label == trace_label
    CHUNK_SIZE.pop()
    EFFECTIVE_CHUNK_SIZE.pop()

    HWM.pop()


def log_df(trace_label, table_name, df):

    if df is None:
        # FIXME force_garbage_collect on delete?
        mem.force_garbage_collect()

    cur_chunker = next(reversed(CHUNK_LOG))

    if df is None:
        CHUNK_LOG.get(cur_chunker).pop(table_name)
        op = 'del'

        logger.debug("log_df del %s : %s " % (table_name, trace_label))

    else:

        shape = df.shape
        elements = np.prod(shape)
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
        logger.debug("log_df add %s elements: %s bytes: %s shape: %s : %s " %
                     (table_name, commas(elements), GB(bytes), shape, trace_label))

    total_elements, total_bytes = _chunk_totals()  # new chunk totals
    cur_mem = mem.get_memory_info()
    hwm_trace_label = "%s.%s.%s" % (trace_label, op, table_name)

    # logger.debug("total_elements: %s, total_bytes: %s cur_mem: %s: %s " %
    #              (total_elements, GB(total_bytes), GB(cur_mem), hwm_trace_label))

    mem.trace_memory_info(hwm_trace_label)

    # - check high_water_marks

    info = "elements: %s bytes: %s mem: %s chunk_size: %s effective_chunk_size: %s" % \
           (commas(total_elements), GB(total_bytes), GB(cur_mem),
            commas(CHUNK_SIZE[0]), commas(EFFECTIVE_CHUNK_SIZE[0]))

    check_hwm('elements', total_elements, info, hwm_trace_label)
    check_hwm('bytes', total_bytes, info, hwm_trace_label)
    check_hwm('mem', cur_mem, info, hwm_trace_label)


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


def check_hwm(tag, value, info, trace_label):

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

    # - elements shouldn't exceed chunk_size or effective_chunk_size of base chunker
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
        check_chunk_size(hwm, EFFECTIVE_CHUNK_SIZE[0],  'effective_chunk_size', max_leeway=1.1)
        check_chunk_size(hwm, CHUNK_SIZE[0], 'chunk_size', max_leeway=1)


def rows_per_chunk(chunk_size, row_size, num_choosers, trace_label):

    if chunk_size > 0:
        # closest number of chooser rows to achieve chunk_size without exceeding
        rpc = int(chunk_size / float(row_size))
    else:
        rpc = num_choosers

    rpc = max(rpc, 1)
    rpc = min(rpc, num_choosers)

    # chunks = int(ceil(num_choosers / float(rpc)))
    effective_chunk_size = row_size * rpc
    num_chunks = (num_choosers // rpc) + (num_choosers % rpc > 0)

    logger.debug("#chunk_calc num_chunks: %s, rows_per_chunk: %s, "
                 "effective_chunk_size: %s, num_choosers: %s : %s" %
                 (num_chunks, rpc, effective_chunk_size, num_choosers, trace_label))

    return rpc, effective_chunk_size


def chunked_choosers(choosers, rows_per_chunk):

    assert choosers.shape[0] > 0

    # generator to iterate over choosers in chunk_size chunks
    num_choosers = len(choosers.index)
    num_chunks = (num_choosers // rows_per_chunk) + (num_choosers % rows_per_chunk > 0)

    i = offset = 0
    while offset < num_choosers:
        yield i+1, num_chunks, choosers.iloc[offset: offset+rows_per_chunk]
        offset += rows_per_chunk
        i += 1


def chunked_choosers_and_alts(choosers, alternatives, rows_per_chunk):
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

    # alternatives index should match choosers (except with duplicate repeating alt rows)
    assert choosers.index.equals(alternatives.index[~alternatives.index.duplicated(keep='first')])

    last_repeat = alternatives.index != np.roll(alternatives.index, -1)
    assert (choosers.shape[0] == 1) or choosers.index.equals(alternatives.index[last_repeat])

    assert choosers.shape[0] > 0
    assert 'pick_count' in alternatives.columns or choosers.index.name == alternatives.index.name

    num_choosers = len(choosers.index)
    num_chunks = (num_choosers // rows_per_chunk) + (num_choosers % rows_per_chunk > 0)

    assert choosers.index.name == alternatives.index.name

    # alt chunks boundaries are where index changes
    alt_ids = alternatives.index.values
    alt_chunk_end = np.where(alt_ids[:-1] != alt_ids[1:])[0] + 1
    alt_chunk_end = np.append([0], alt_chunk_end)  # including the first...
    alt_chunk_end = alt_chunk_end[rows_per_chunk::rows_per_chunk]

    # add index to end of array to capture any final partial chunk
    alt_chunk_end = np.append(alt_chunk_end, [len(alternatives.index)])

    i = offset = alt_offset = 0
    while offset < num_choosers:

        alt_end = alt_chunk_end[i]

        chooser_chunk = choosers[offset: offset + rows_per_chunk]
        alternative_chunk = alternatives[alt_offset: alt_end]

        assert len(chooser_chunk.index) == len(np.unique(alternative_chunk.index.values))

        yield i+1, num_chunks, chooser_chunk, alternative_chunk

        i += 1
        offset += rows_per_chunk
        alt_offset = alt_end


def chunked_choosers_by_chunk_id(choosers, rows_per_chunk):
    # generator to iterate over choosers in chunk_size chunks
    # like chunked_choosers but based on chunk_id field rather than dataframe length
    # (the presumption is that choosers has multiple rows with the same chunk_id that
    # all have to be included in the same chunk)
    # FIXME - we pathologically know name of chunk_id col in households table

    assert choosers.shape[0] > 0

    num_choosers = choosers['chunk_id'].max() + 1
    num_chunks = (num_choosers // rows_per_chunk) + (num_choosers % rows_per_chunk > 0)

    i = offset = 0
    while offset < num_choosers:
        chooser_chunk = choosers[choosers['chunk_id'].between(offset, offset + rows_per_chunk - 1)]
        yield i+1, num_chunks, chooser_chunk
        offset += rows_per_chunk
        i += 1
