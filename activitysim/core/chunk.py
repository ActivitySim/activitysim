# ActivitySim
# See full license in LICENSE.txt.

from math import ceil
import os
import logging

import numpy as np
import pandas as pd

from .skim import SkimDictWrapper, SkimStackWrapper
from . import logit
from . import tracing
from . import pipeline
from . import util

logger = logging.getLogger(__name__)


def log_chunk_df(trace_label, df):

    elements = df.shape[0] * df.shape[1]
    bytes = df.memory_usage(index=True).sum()

    logger.debug("%s log_chunk_df #chunk %s %s" % (trace_label, elements, util.GB(bytes)))


def calc_rows_per_chunk(chunk_size, choosers, alternatives=None,
                        sample_size=None, alt_sample=None, by_chunk_id=False):

    if by_chunk_id:
        num_choosers = choosers['chunk_id'].max() + 1
    else:
        num_choosers = len(choosers.index)

    # FIXME - except we want logging?...
    # if not chunking, then return num_choosers
    # if chunk_size == 0:
    #     return num_choosers

    chooser_row_size = len(choosers.columns)

    assert (alternatives is None) or (alt_sample is None)

    SKIM_COLUMNS = 1

    if alternatives is not None:
        JOIN_COLUMN = 1
        alt_row_size = alternatives.shape[1] + SKIM_COLUMNS + JOIN_COLUMN
        sample_size = sample_size or alternatives.shape[0]
        row_size = (chooser_row_size + alt_row_size) * sample_size
    elif alt_sample is not None:
        alt_row_size = alt_sample.shape[1] + SKIM_COLUMNS
        # average sample size
        sample_size = alt_sample.shape[0] / float(num_choosers)
        row_size = (chooser_row_size + alt_row_size) * sample_size
    else:
        alt_row_size = 0
        sample_size = 1
        row_size = chooser_row_size

    if by_chunk_id:
        # scale row_size by average number of chooser rows per chunk_id
        rows_per_chunk_id = len(choosers.index) / float(num_choosers)
        row_size = int(rows_per_chunk_id * row_size)

    if chunk_size > 0:
        # closest number of chooser rows to achieve chunk_size
        rows_per_chunk = int(round(chunk_size / float(row_size)))
        rows_per_chunk = max(rows_per_chunk, 1)

        rows_per_chunk = min(rows_per_chunk, num_choosers)
    else:
        # if not chunking, then return num_choosers
        rows_per_chunk = num_choosers

    chunks = int(ceil(num_choosers / float(rows_per_chunk)))
    effective_chunk_size = row_size * rows_per_chunk

    logger.debug("calc_rows_per_chunk #chunk chunk_size %s" % chunk_size)
    logger.debug("calc_rows_per_chunk #chunk num_choosers %s" % num_choosers)
    logger.debug("calc_rows_per_chunk #chunk chooser_row_size %s" % chooser_row_size)
    logger.debug("calc_rows_per_chunk #chunk sample_size %s" % sample_size)
    logger.debug("calc_rows_per_chunk #chunk alt_row_size %s" % alt_row_size)
    logger.debug("calc_rows_per_chunk #chunk total row_size %s" % row_size)
    logger.debug("calc_rows_per_chunk #chunk rows_per_chunk %s" % rows_per_chunk)
    logger.debug("calc_rows_per_chunk #chunk effective_chunk_size %s" % effective_chunk_size)
    logger.debug("calc_rows_per_chunk #chunk chunks %s" % chunks)

    return rows_per_chunk


def chunked_choosers(choosers, rows_per_chunk):
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

    assert 'pick_count' in alternatives.columns or choosers.index.name == alternatives.index.name

    num_choosers = len(choosers.index)
    num_chunks = (num_choosers // rows_per_chunk) + (num_choosers % rows_per_chunk > 0)

    if choosers.index.name == alternatives.index.name:
        assert choosers.index.name == alternatives.index.name

        # alt chunks boundaries are where index changes
        alt_ids = alternatives.index.values
        alt_chunk_end = np.where(alt_ids[:-1] != alt_ids[1:])[0] + 1
        alt_chunk_end = np.append([0], alt_chunk_end)  # including the first...
        alt_chunk_end = alt_chunk_end[rows_per_chunk::rows_per_chunk]

    else:
        # used to do it this way for school and workplace (which are sampled based on prob)
        # since the utility expressions need to know pick_count for sample correction
        # but for now the assumption that choosers and alternatives share indexes is more general
        # leaving this (previously correct) code here for now in case that changes...
        assert False

        # assert 'pick_count' in alternatives.columns
        # assert 'cum_pick_count' not in alternatives.columns
        # alternatives['cum_pick_count'] = alternatives['pick_count'].cumsum()
        #
        # # currently no convenient way to remember sample_size across steps
        # pick_count = alternatives.cum_pick_count.iat[-1]
        # sample_size = pick_count / len(choosers.index)
        # assert pick_count % sample_size == 0
        #
        # alt_chunk_size = rows_per_chunk * sample_size
        #
        # # array of indices of starts of alt chunks
        # alt_chunk_end = np.where(alternatives['cum_pick_count'] % alt_chunk_size == 0)[0] + 1

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


def hh_chunked_choosers(choosers, rows_per_chunk):
    # generator to iterate over choosers in chunk_size chunks
    # like chunked_choosers but based on chunk_id field rather than dataframe length
    # (the presumption is that choosers has multiple rows with the same chunk_id that
    # all have to be included in the same chunk)
    # FIXME - we pathologically know name of chunk_id col in households table

    num_choosers = choosers['chunk_id'].max() + 1
    num_chunks = (num_choosers // rows_per_chunk) + (num_choosers % rows_per_chunk > 0)

    i = offset = 0
    while offset < num_choosers:
        chooser_chunk = choosers[choosers['chunk_id'].between(offset, offset + rows_per_chunk - 1)]
        yield i+1, num_chunks, chooser_chunk
        offset += rows_per_chunk
        i += 1
