from collections import OrderedDict
from future.utils import iteritems

import numpy as np
import pytest

from activitysim.abm.tables import skims


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
        'omx_name': 'arc_skims',
        'omx_shape': (matrix_dimension, matrix_dimension),
        'num_skims': num_of_matrices,
        'dtype': np.float32,
        'omx_keys': omx_keys,
        'key1_block_offsets': omx_key1_block_offsets,
        'block_offsets': omx_block_offsets,
        'blocks': omx_blocks
    }

    return skim_info


def test_multiply_large_numbers(skim_info, num_of_matrices, matrix_dimension):
    omx_shape = skim_info['omx_shape']
    blocks = skim_info['blocks']

    for block_name, block_size in iteritems(blocks):
        # If overflow, this number will go negative
        assert int(skims.multiply_large_numbers(omx_shape) * block_size) == \
               num_of_matrices * matrix_dimension ** 2


def test_multiple_large_floats():
    calculated_value = skims.multiply_large_numbers([6205.1, 5423.2, 932.4, 15.4])
    actual_value = 483200518316.9472
    assert abs(calculated_value - actual_value) < 0.0001
