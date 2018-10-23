# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402

import logging

import numpy as np
import pandas as pd

from . import util

logger = logging.getLogger(__name__)


def log_df_size(trace_label, table_name, df, cum_size):

    if isinstance(df, pd.Series):
        elements = df.shape[0]
        bytes = df.memory_usage(index=True)
    elif isinstance(df, pd.DataFrame):
        elements = df.shape[0] * df.shape[1]
        bytes = df.memory_usage(index=True).sum()
    else:
        assert False

    # logger.debug("%s #chunk log_df_size %s %s %s %s" %
    #              (trace_label, table_name, df.shape, elements, util.GB(bytes)))

    if cum_size:
        elements += cum_size[0]
        bytes += cum_size[1]

    return elements, bytes


def log_chunk_size(trace_label, cum):

    elements = cum[0]
    bytes = cum[1]

    logger.debug("%s #chunk CUM %s %s" % (trace_label, elements, util.GB(bytes)))
    logger.debug("%s %s" % (trace_label, util.memory_info()))
    logger.debug("%s %s" % (trace_label, util.memory_info(full=True)))


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
