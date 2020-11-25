# ActivitySim
# See full license in LICENSE.txt.
import logging
import os
import time
import multiprocessing

from contextlib import contextmanager

import pandas as pd
import numpy as np

from activitysim.core import config
from activitysim.core import pipeline
from activitysim.core import tracing
from activitysim.core import chunk
from activitysim.core import inject
from activitysim.core import pathbuilder_cache
from activitysim.core import los

from activitysim.core import pathbuilder

logger = logging.getLogger(__name__)


def od_choosers(network_los, uid_calculator, scalar_attributes):
    """

    Parameters
    ----------
    scalar_attributes: dict of scalar attribute name:value pairs

    Returns
    -------
    pandas.Dataframe
    """

    tap_ids = network_los.tap_df['TAP'].values

    # create OD dataframe in ROW_MAJOR_LAYOUT
    od_choosers_df = pd.DataFrame(
        data={
            'btap': np.repeat(tap_ids, len(tap_ids)),
            'atap': np.tile(tap_ids, len(tap_ids))
        }
    )

    # add any attribute columns specified in settings (the rest will be scalars in locals_dict)
    attributes_as_columns = \
        network_los.setting('TVPB_SETTINGS.tour_mode_choice.tap_tap_settings.attributes_as_columns', [])
    for attribute_name in attributes_as_columns:
        od_choosers_df[attribute_name] = scalar_attributes[attribute_name]

    od_choosers_df.index = uid_calculator.get_unique_ids(od_choosers_df, scalar_attributes)
    assert not od_choosers_df.index.duplicated().any()

    return od_choosers_df


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

    trace_label = 'initialize_los'

    if network_los.zone_system == los.THREE_ZONE:

        tap_cache = network_los.tvpb.tap_cache
        uid_calculator = network_los.tvpb.uid_calculator
        attribute_combinations_df = uid_calculator.scalar_attribute_combinations()

        # - write table to pipeline (so we can slice it, when multiprocessing)
        pipeline.replace_table('attribute_combinations', attribute_combinations_df)

        # clean up any unwanted cache files from previous run
        network_los.tvpb.tap_cache.cleanup()

        # if multiprocessing make sure shred cache was filled with pathbuilder_cache.UNINITIALIZED
        # so that initialize_tvpb subprocesses can detect when cache is fully populated
        if network_los.multiprocess():
            data, lock = tap_cache.get_data_and_lock_from_buffers()
            with lock:
                assert np.all(data == pathbuilder_cache.UNINITIALIZED)



@contextmanager
def lock_data(lock):
    if lock is not None:
        with lock:
            yield
    else:
        yield


def any_uninitialized(data, lock):
    with lock_data(lock):
        return np.any(data == pathbuilder_cache.UNINITIALIZED)


@inject.step()
def initialize_tvpb(network_los, attribute_combinations):
    """
    Initialize STATIC tap_tap_utility cache and write mmap to disk.

    uses pipeline attribute_combinations table created in initialize_los to determine which attribute tuples
    to compute utilities for.

    if we are single-processing, this will be the entire set of attribute tuples required to fully populate cache

    if we are multiprocessing, then the attribute_combinations will have been sliced and we compute only a subset
    of the tuples (and the other processes will compute the rest). All process wait until the cache is fully
    populated before returning, and the spokesman/locutor process writes the results.


    FIXME - if we did not close this, we could avoid having to reload it from mmap when single-process?
    """

    trace_label = 'initialize_tvpb'

    if network_los.zone_system != los.THREE_ZONE:
        logger.info(f"{trace_label} - skipping step because zone_system is not THREE_ZONE")
        return

    attribute_combinations_df = attribute_combinations.to_frame()
    multiprocess = network_los.multiprocess()

    tap_cache = network_los.tvpb.tap_cache
    uid_calculator = network_los.tvpb.uid_calculator
    assert not tap_cache.is_open

    # if cache already exists,
    if os.path.isfile(tap_cache.cache_path(pathbuilder_cache.STATIC)):
        # otherwise should have been deleted by TVPBCache.cleanup in initialize_los step
        assert not network_los.rebuild_tvpb_cache
        logger.info(f"{trace_label} skipping rebuild of STATIC cache because rebuild_tvpb_cache setting is False"
                    f" and cache already exists: {tap_cache.cache_path(pathbuilder_cache.STATIC)}")
        return

    if multiprocess:
        # we will compute 'skim' chunks at offsets specified by our slice of attribute_combinations_df
        data, lock = tap_cache.get_data_and_lock_from_buffers()
        write_results = inject.get_injectable('locutor', False)

        # possible (though unlikely) timing bug here
        #assert np.all(data == pathbuilder_cache.UNINITIALIZED)
    else:
        data = tap_cache.allocate_data_buffer(shared=False)
        lock = None
        write_results = True

    data = data.reshape(uid_calculator.skim_shape)

    logger.debug(f"{trace_label} processing {len(attribute_combinations_df)} {data[0].shape} attribute_combinations")
    logger.debug(f"{trace_label} compute utilities for attribute_combinations_df\n{attribute_combinations_df}")

    with chunk.chunk_log(trace_label):

        for offset, scalar_attributes in attribute_combinations_df.to_dict('index').items():
            # compute utilities for this 'skim' with a single full set of scalar attributes

            logger.info(f"{trace_label} compute utilities for offset {offset} scalar_attributes: {scalar_attributes}")

            assert (offset == uid_calculator.get_skim_offset(scalar_attributes))

            # scalar_attributes is a dict of attribute name/value pairs for this combination
            # (e.g. {'demographic_segment': 0, 'tod': 'AM', 'access_mode': 'walk'})
            trace_attributes = ', '.join([f'{k}={v}' for k, v in scalar_attributes.items()])
            chunk_trace_label = tracing.extend_trace_label(trace_label, trace_attributes)

            choosers_df = od_choosers(network_los, uid_calculator, scalar_attributes)

            chunk.log_df('chunk_trace_label', 'choosers_df', choosers_df)

            assert (choosers_df.index.values == offset * len(choosers_df) + np.arange(len(choosers_df))).all()

            model_settings = network_los.setting(f'TVPB_SETTINGS.tour_mode_choice.tap_tap_settings')

            model_constants = network_los.setting(f'TVPB_SETTINGS.tour_mode_choice.CONSTANTS').copy()
            model_constants.update(scalar_attributes)

            utilities_df = \
                pathbuilder.compute_utilities(network_los,
                                              model_settings=model_settings,
                                              choosers=choosers_df,
                                              model_constants=model_constants,
                                              trace_label=chunk_trace_label)

            chunk.log_df(chunk_trace_label, f'{trace_attributes} utilities_df', utilities_df)

            assert utilities_df.values.shape == data[offset].shape
            assert not np.any(utilities_df.values == pathbuilder_cache.UNINITIALIZED)
            with lock_data(lock):
                data[offset, :, :] = utilities_df.values

            # FIXME do we care about offset?
            assert (uid_calculator.get_skim_offset(scalar_attributes) == offset)

            logger.debug(f"{trace_label} updated utilities for offset {offset}")

    # if multiprocessing, we have to wait for all processes to fully populate share data
    if multiprocess:
        logger.info(f"{trace_label} Waiting for other processes to fully populate cache.")
        while any_uninitialized(data, lock):
            assert multiprocess  # if single_process, we should have fully populated data
            print(f"{trace_label}.{multiprocessing.current_process().name} "
                  f" waiting for other process to fully populate pathbuilder_cache..")
            logger.debug(f"{trace_label}. waiting for other process to complete..")
            time.sleep(5)

    if write_results:

        logger.info(f"{trace_label} writing cache for_rebuild.")

        with lock_data(lock):
            final_df = \
                pd.DataFrame(data=data.reshape(uid_calculator.fully_populated_shape),
                             columns=uid_calculator.set_names,
                             index=uid_calculator.fully_populated_uids)
            final_df.index.name = 'uid'

            tap_cache.open(for_rebuild=True)
            tap_cache.extend_table(final_df)
            assert tap_cache.is_fully_populated
            tap_cache.close()
