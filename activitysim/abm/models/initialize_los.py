# ActivitySim
# See full license in LICENSE.txt.
import logging
import os
import time

from contextlib import contextmanager

import pandas as pd
import numpy as np

from activitysim.core import tracing
from activitysim.core import chunk
from activitysim.core import inject
from activitysim.core import cache
from activitysim.core import config

from activitysim.core import transit_virtual_path_builder as tvpb

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

    trace_label = 'initialize_los'

    tap_cache = network_los.tvpb.tap_cache
    uid_calculator = network_los.tvpb.uid_calculator

    assert not tap_cache.is_open

    if os.path.isfile(tap_cache.cache_path(cache.STATIC)):
        # otherwise should have been deleted by TVPBCache.cleanup at start of run
        assert not network_los.get('rebuild_tvpb_cache', cache.REBUILD_DEFAULT)
        logger.info(f"initialize_los skipping rebuild of STATIC cache because rebuild_tvpb_cache setting is False"
                    f" and cache already exists: {tap_cache.cache_path}")
        return

    assert not os.path.isfile(tap_cache.cache_path)

    attribute_combinations_df = uid_calculator.scalar_attribute_combinations()
    shared_data = cache.multiprocess()

    if shared_data:
        # multiprocess - we will compute 'skim' chunks at offsets specified by our slice of attribute_combinations_df
        data, lock = tap_cache.get_data_and_lock_from_buffers()
        write_results = inject.get_injectable('locutor', False)
    else:
        # singleprocess - we will compute all data
        assert not cache.multiprocess()
        data = tap_cache.allocate_data_buffer(shared=False)
        write_results = True

    data = data.reshape(uid_calculator.skim_shape)

    with chunk.chunk_log(trace_label):

        for offset, scalar_attributes in attribute_combinations_df.to_dict('index').items():

            print(f"compute utilities for offset {offset} scalar_attributes: {scalar_attributes}")

            # compute utilities for this 'skim' with a single full set of scalar attributes

            assert (offset == uid_calculator.get_skim_offset(scalar_attributes)).all()

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
                tvpb.compute_utilities(network_los,
                                       model_settings=model_settings,
                                       choosers=choosers_df,
                                       model_constants=model_constants,
                                       trace_label=chunk_trace_label)

            chunk.log_df(chunk_trace_label, f'{trace_attributes} utilities_df', utilities_df)

            assert utilities_df.values.shape == data[offset].shape
            if shared_data:
                with lock:
                    data[offset, :, :] = utilities_df.values
            else:
                data[offset, :, :] = utilities_df.values

            # FIXME do we care about offset?
            assert (offset == uid_calculator.get_skim_offset(scalar_attributes)).all()

    if write_results:

        if shared_data:
            def populated():
                with lock:
                    return np.any(data == cache.UNINITIALIZED)
            while not populated():
                print(f"initialize_los locutor waiting on lock")
                time.sleep(1)

        data = data.reshape(uid_calculator.fully_populated_shape)
        final_df = pd.DataFrame(data, columns=uid_calculator.set_names, index=uid_calculator.fully_populated_uids)
        final_df.index.name = 'uid'

        tap_cache.open(for_rebuild=True)
        tap_cache.extend_table(final_df)
        tap_cache.close()

