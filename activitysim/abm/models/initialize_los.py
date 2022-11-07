# ActivitySim
# See full license in LICENSE.txt.
import logging
import multiprocessing
import os
import time
from contextlib import contextmanager

import numba
import numpy as np
import pandas as pd

from activitysim.core import (
    assign,
    chunk,
    config,
    inject,
    los,
    pathbuilder,
    pipeline,
    simulate,
    tracing,
)

logger = logging.getLogger(__name__)


@contextmanager
def lock_data(lock):
    if lock is not None:
        with lock:
            yield
    else:
        yield


@numba.njit(nogil=True)
def any_nans(data):
    # equivalent to np.isnan(x).any()
    # short circuit in any nans - but the real point is to save memory (np.flat is a nop-copy iterator)
    for x in data.flat:
        if np.isnan(x):
            return True
    return False


@numba.njit(nogil=True)
def num_nans(data):
    # equivalent to np.isnan(x).sum() but with no transient memory overhead (np.flat is a nop-copy iterator)
    n = 0
    for x in data.flat:
        if np.isnan(x):
            n += 1
    return n


def any_uninitialized(data, lock=None):

    with lock_data(lock):
        result = any_nans(data)
    return result


def num_uninitialized(data, lock=None):

    with lock_data(lock):
        result = num_nans(data)
    return result


@inject.step()
def initialize_los(network_los):
    """
    Currently, this step is only needed for THREE_ZONE systems in which the tap_tap_utilities are precomputed
    in the (presumably subsequent) initialize_tvpb step.

    Adds attribute_combinations_df table to the pipeline so that it can be used to as the slicer
    for multiprocessing the initialize_tvpb s.tep

    FIXME - this step is only strictly necessary when multiprocessing, but initialize_tvpb would need to be tweaked
    FIXME - to instantiate attribute_combinations_df if the pipeline table version were not available.
    """

    trace_label = "initialize_los"

    if network_los.zone_system == los.THREE_ZONE:

        tap_cache = network_los.tvpb.tap_cache
        uid_calculator = network_los.tvpb.uid_calculator
        attribute_combinations_df = uid_calculator.scalar_attribute_combinations()

        # - write table to pipeline (so we can slice it, when multiprocessing)
        pipeline.replace_table("attribute_combinations", attribute_combinations_df)

        # clean up any unwanted cache files from previous run
        if network_los.rebuild_tvpb_cache:
            network_los.tvpb.tap_cache.cleanup()

        # if multiprocessing make sure shared cache is filled with np.nan
        # so that initialize_tvpb subprocesses can detect when cache is fully populated
        if network_los.multiprocess():
            (
                data,
                lock,
            ) = (
                tap_cache.get_data_and_lock_from_buffers()
            )  # don't need lock here since single process

            if os.path.isfile(tap_cache.cache_path):
                # fully populated cache should have been loaded from saved cache
                assert not network_los.rebuild_tvpb_cache
                assert not any_uninitialized(data, lock=None)
            else:
                # shared cache should be filled with np.nan so that initialize_tvpb
                # subprocesses can detect when cache is fully populated
                with lock_data(lock):
                    np.copyto(data, np.nan)


def compute_utilities_for_attribute_tuple(
    network_los, scalar_attributes, data, chunk_size, trace_label
):

    # scalar_attributes is a dict of attribute name/value pairs for this combination
    # (e.g. {'demographic_segment': 0, 'tod': 'AM', 'access_mode': 'walk'})

    logger.info(f"{trace_label} scalar_attributes: {scalar_attributes}")

    uid_calculator = network_los.tvpb.uid_calculator

    attributes_as_columns = network_los.setting(
        "TVPB_SETTINGS.tour_mode_choice.tap_tap_settings.attributes_as_columns", []
    )
    model_settings = network_los.setting(
        f"TVPB_SETTINGS.tour_mode_choice.tap_tap_settings"
    )
    model_constants = network_los.setting(
        f"TVPB_SETTINGS.tour_mode_choice.CONSTANTS"
    ).copy()
    model_constants.update(scalar_attributes)

    data = data.reshape(uid_calculator.fully_populated_shape)

    # get od skim_offset dataframe with uid index corresponding to scalar_attributes
    choosers_df = uid_calculator.get_od_dataframe(scalar_attributes)

    # choosers_df is pretty big and was custom made for compute_utilities but we don't need to chunk_log it
    # since it is created outside of adaptive_chunked_choosers and so will show up in baseline
    assert not chunk.chunk_logging()  # otherwise we should chunk_log this

    chunk_tag = "initialize_tvpb"  # all attribute_combinations can use same cached data for row_size calc

    for i, chooser_chunk, chunk_trace_label in chunk.adaptive_chunked_choosers(
        choosers_df, chunk_size, trace_label, chunk_tag=chunk_tag
    ):
        # we should count choosers_df as chunk overhead since its pretty big and was custom made for compute_utilities
        assert chooser_chunk._is_view  # otherwise copying it is wasteful
        chooser_chunk = chooser_chunk.copy()
        chunk.log_df(trace_label, "attribute_chooser_chunk", chooser_chunk)

        # add any attribute columns specified as column attributes in settings (the rest will be scalars in locals_dict)
        for attribute_name in attributes_as_columns:
            chooser_chunk[attribute_name] = scalar_attributes[attribute_name]

        chunk.log_df(trace_label, "attribute_chooser_chunk", chooser_chunk)

        utilities_df = pathbuilder.compute_utilities(
            network_los,
            model_settings=model_settings,
            choosers=chooser_chunk,
            model_constants=model_constants,
            trace_label=trace_label,
        )

        chunk.log_df(trace_label, "utilities_df", utilities_df)

        assert len(utilities_df) == len(chooser_chunk)
        assert len(utilities_df.columns) == data.shape[1]
        assert not any_uninitialized(utilities_df.values)

        data[chooser_chunk.index.values, :] = utilities_df.values

        del chooser_chunk
        chunk.log_df(trace_label, "attribute_chooser_chunk", None)

    logger.debug(f"{trace_label} updated utilities")


@inject.step()
def initialize_tvpb(network_los, attribute_combinations, chunk_size):
    """
    Initialize STATIC tap_tap_utility cache and write mmap to disk.

    uses pipeline attribute_combinations table created in initialize_los to determine which attribute tuples
    to compute utilities for.

    if we are single-processing, this will be the entire set of attribute tuples required to fully populate cache

    if we are multiprocessing, then the attribute_combinations will have been sliced and we compute only a subset
    of the tuples (and the other processes will compute the rest). All process wait until the cache is fully
    populated before returning, and the locutor process writes the results.


    FIXME - if we did not close this, we could avoid having to reload it from mmap when single-process?
    """

    trace_label = "initialize_tvpb"

    if network_los.zone_system != los.THREE_ZONE:
        logger.info(
            f"{trace_label} - skipping step because zone_system is not THREE_ZONE"
        )
        return

    attribute_combinations_df = attribute_combinations.to_frame()
    multiprocess = network_los.multiprocess()
    uid_calculator = network_los.tvpb.uid_calculator

    tap_cache = network_los.tvpb.tap_cache
    assert not tap_cache.is_open

    # if cache already exists,
    if os.path.isfile(tap_cache.cache_path):
        # otherwise should have been deleted by TVPBCache.cleanup in initialize_los step
        assert not network_los.rebuild_tvpb_cache
        logger.info(
            f"{trace_label} skipping rebuild of STATIC cache because rebuild_tvpb_cache setting is False"
            f" and cache already exists: {tap_cache.cache_path}"
        )
        return

    if multiprocess:
        # we will compute 'skim' chunks at offsets specified by our slice of attribute_combinations_df
        data, lock = tap_cache.get_data_and_lock_from_buffers()
    else:
        data = tap_cache.allocate_data_buffer(shared=False)
        lock = None

    logger.debug(
        f"{trace_label} processing {len(attribute_combinations_df)} attribute_combinations"
    )
    logger.debug(
        f"{trace_label} compute utilities for attribute_combinations_df\n{attribute_combinations_df}"
    )

    for offset, scalar_attributes in attribute_combinations_df.to_dict("index").items():
        # compute utilities for this 'skim' with a single full set of scalar attributes

        offset = network_los.tvpb.uid_calculator.get_skim_offset(scalar_attributes)
        tuple_trace_label = tracing.extend_trace_label(trace_label, f"offset{offset}")

        compute_utilities_for_attribute_tuple(
            network_los, scalar_attributes, data, chunk_size, tuple_trace_label
        )

        # make sure we populated the entire offset
        assert not any_uninitialized(
            data.reshape(uid_calculator.skim_shape)[offset], lock
        )

    if multiprocess and not inject.get_injectable("locutor", False):
        return

    write_results = not multiprocess or inject.get_injectable("locutor", False)
    if write_results:

        if multiprocess:
            # if multiprocessing, wait for all processes to fully populate share data before writing results
            # (the other processes don't have to wait, since we were sliced by attribute combination
            # and they must wait to coalesce at the end of the multiprocessing_step)
            # FIXME testing entire array is costly in terms of RAM)

            while any_uninitialized(data, lock):
                logger.debug(
                    f"{trace_label}.{multiprocessing.current_process().name} waiting for other processes"
                    f" to populate {num_uninitialized(data, lock)} uninitialized data values"
                )
                time.sleep(5)

        logger.info(f"{trace_label} writing static cache.")
        with lock_data(lock):
            tap_cache.write_static_cache(data)
