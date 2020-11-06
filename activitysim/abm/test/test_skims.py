from collections import OrderedDict


import numpy as np
import pytest

from activitysim.core import los


@pytest.fixture(scope="session")
def matrix_dimension():
    return 5922


@pytest.fixture(scope="session")
def num_of_matrices():
    return 845


@pytest.fixture(scope="session")
def skim_info(num_of_matrices, matrix_dimension):
    time_periods = ['EA', 'AM', 'MD', 'PM', 'NT']

    omx_keys = OrderedDict()
    omx_key1_block_offsets = OrderedDict()
    omx_block_offsets = OrderedDict()
    omx_blocks = OrderedDict()
    omx_blocks['skim_arc_skims_0'] = num_of_matrices

    for i in range(0, num_of_matrices + 1):
        key1_name = 'm{}'.format(i // len(time_periods) + 1)
        time_period = time_periods[i % len(time_periods)]

        omx_keys[(key1_name, time_period)] = '{}__{}'.format(key1_name, time_period)
        omx_block_offsets[(key1_name, time_period)] = (0, i)

        if 0 == i % len(time_periods):
            omx_key1_block_offsets[key1_name] = (0, i)

    skim_info = {
        'skim_tag': 'arc_skims',
        'omx_shape': (matrix_dimension, matrix_dimension),
        'num_skims': num_of_matrices,
        'dtype': np.float32,
        'omx_keys': omx_keys,
        'key1_block_offsets': omx_key1_block_offsets,
        'block_offsets': omx_block_offsets,
        'blocks': omx_blocks
    }

    return skim_info
