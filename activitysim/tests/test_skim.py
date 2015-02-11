# ActivitySim
# Copyright (C) 2014-2015 Synthicity, LLC
# See full license in LICENSE.txt.

import numpy as np
import numpy.testing as npt
import pytest

from .. import skim


@pytest.fixture
def data():
    return np.arange(100, dtype='int').reshape((10, 10))


def test_basic(data):
    sk = skim.Skim(data)

    orig = [5, 9, 1]
    dest = [2, 9, 6]

    npt.assert_array_equal(
        sk.get(orig, dest),
        [52, 99, 16])


def test_offset(data):
    sk = skim.Skim(data, offset=-1)

    orig = [6, 10, 2]
    dest = [3, 10, 7]

    npt.assert_array_equal(
        sk.get(orig, dest),
        [52, 99, 16])
