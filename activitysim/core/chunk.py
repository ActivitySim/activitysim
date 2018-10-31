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
# table_dicts are dicts tuples of (elements, bytes, shape) keyed by table_name
CHUNK_LOG = OrderedDict()

# array of chunk_size active CHUNK_LOG
CHUNK_SIZE = []

ELEMENTS_HWM = [{}]
BYTES_HWM = [{}]
MEM_HWM = [{}]


def GB(bytes, sign=False):
    if bytes < (1024 * 1024):
        return ("%+.2f KB" if sign else "%.2f KB") % (bytes / 1024)
    elif bytes < (1024 * 1024 * 1024):
        return ("%+.2f MB" if sign else "%.2f MB") % (bytes / (1024 * 1024))
    else:
        return ("%+.2f GB" if sign else "%.2f GB") % (bytes / (1024 * 1024 * 1024))


def log_open(trace_label, chunk_size):

    if len(CHUNK_LOG) > 0:
        assert chunk_size == 0
        logger.debug("log_open nested chunker %s" % trace_label)
        assert trace_label not in CHUNK_LOG

    CHUNK_LOG[trace_label] = OrderedDict()
    CHUNK_SIZE.append(chunk_size)

    ELEMENTS_HWM.append({})
    BYTES_HWM.append({})
    MEM_HWM.append({})


def log_close(trace_label):

    assert CHUNK_LOG and next(reversed(CHUNK_LOG)) == trace_label

    logger.debug("log_close %s" % trace_label)

    log_write_hwm()

    label, _ = CHUNK_LOG.popitem(last=True)
    assert label == trace_label
    CHUNK_SIZE.pop()

    ELEMENTS_HWM.pop()
    BYTES_HWM.pop()
    MEM_HWM.pop()


def log_df(trace_label, table_name, df):

    # if df is None:
    #     mem.force_garbage_collect()
    # return

    cur_chunker = next(reversed(CHUNK_LOG))
    op = 'del' if df is None else 'add'

    if df is None:
        CHUNK_LOG.get(cur_chunker).pop(table_name)
        elements = bytes = 0
        shape = (0, 0)

        mem.force_garbage_collect()
    else:

        shape = df.shape
        elements = np.prod(shape)

        if isinstance(df, pd.Series):
            bytes = df.memory_usage(index=True)
        elif isinstance(df, pd.DataFrame):
            bytes = df.memory_usage(index=True).sum()
        elif isinstance(df, np.ndarray):
            bytes = df.nbytes
        else:
            logger.error("log_df %s unknown type: %s" % (table_name, type(df)))
            assert False

        CHUNK_LOG.get(cur_chunker)[table_name] = (elements, bytes, shape)

    # log this df
    logger.debug("%s %s df %s %s %s : %s " %
                 (op, table_name, elements, shape, GB(bytes), trace_label))

    total_elements, total_bytes = _chunk_totals()
    cur_mem = mem.get_memory_info()

    # # log current totals
    # logger.debug("%s %s total elements: %d (%+d) bytes: %s (%s) mem: %s " %
    #              (op, table_name,
    #               total_elements, total_elements-prev_elements,
    #               GB(total_bytes), GB(total_bytes - prev_bytes, sign=True),
    #               GB(cur_mem), ))

    # - check high_water_marks
    hwm_trace_label = "%s.%s.%s" % (trace_label, op, table_name)
    for hwm in ELEMENTS_HWM:
        if total_elements > hwm.get('mark', 0):
            hwm['mark'] = total_elements
            hwm['trace_label'] = hwm_trace_label
            hwm['info'] = "bytes: %s mem: %s" % (GB(total_bytes), GB(cur_mem))

    for hwm in BYTES_HWM:
        if total_bytes > hwm.get('mark', 0):
            hwm['mark'] = total_bytes
            hwm['trace_label'] = hwm_trace_label
            hwm['info'] = "elements: %s mem: %s" % (total_elements, GB(cur_mem))

    for hwm in MEM_HWM:
        if cur_mem > hwm.get('mark', 0):
            hwm['mark'] = cur_mem
            hwm['trace_label'] = hwm_trace_label
            hwm['info'] = "elements: %s bytes: %s" % (total_elements, GB(total_bytes))


def _chunk_totals():

    total_elements = 0
    total_bytes = 0
    for label in CHUNK_LOG:
        tables = CHUNK_LOG[label]
        for table_name in tables:
            elements, bytes, shape = tables[table_name]
            total_elements += elements
            total_bytes += bytes

    return total_elements, total_bytes


def log_write_hwm():

    hwm = ELEMENTS_HWM[-1]
    if 'mark' in hwm:
        logger.info("high_water_mark elements: %s (%s) in %s" %
                    (hwm['mark'], hwm['info'], hwm['trace_label']), )
    hwm = BYTES_HWM[-1]
    if 'mark' in hwm:
        logger.info("high_water_mark bytes: %s (%s) in %s" %
                    (GB(hwm['mark']), hwm['info'], hwm['trace_label']), )

    hwm = MEM_HWM[-1]
    if 'mark' in hwm:
        logger.info("high_water_mark mem: %s (%s) in %s" %
                    (GB(hwm['mark']), hwm['info'], hwm['trace_label']), )


def rows_per_chunk(chunk_size, row_size, num_choosers, trace_label):

    # closest number of chooser rows to achieve chunk_size
    rpc = int(round(chunk_size / float(row_size)))

    rpc = int(chunk_size / float(row_size))

    rpc = max(rpc, 1)
    rpc = min(rpc, num_choosers)

    # chunks = int(ceil(num_choosers / float(rpc)))
    # effective_chunk_size = row_size * rpc
    #
    # logger.debug("%s #chunk_calc chunk_size %s" % (trace_label, chunk_size))
    # logger.debug("%s #chunk_calc num_choosers %s" % (trace_label, num_choosers))
    # logger.debug("%s #chunk_calc total row_size %s" % (trace_label, row_size))
    # logger.debug("%s #chunk_calc rows_per_chunk %s" % (trace_label, rpc))
    # logger.debug("%s #chunk_calc effective_chunk_size %s" % (trace_label, effective_chunk_size))
    # logger.debug("%s #chunk_calc chunks %s" % (trace_label, chunks))

    return rpc


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
