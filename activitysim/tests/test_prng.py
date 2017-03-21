# ActivitySim
# See full license in LICENSE.txt.

import numpy as np
import pandas as pd
import numpy.testing as npt
import pandas.util.testing as pdt
import pytest

from .. import prng
from .. import pipeline


def test_basic():

    rng = pipeline.get_rn_generator()

    rng.set_base_seed(0)

    gprng = rng.get_global_prng()

    npt.assert_almost_equal(gprng.rand(1), [0.5488135])

    with pytest.raises(AssertionError) as excinfo:
        npt.assert_almost_equal(gprng.rand(1), [0.5488135])
    assert "Arrays are not almost equal" in str(excinfo.value)

    # set_base_seed should reseed
    rng.set_base_seed(0)
    npt.assert_almost_equal(gprng.rand(1), [0.5488135])

    # set_global_prng_offset should reseed
    rng.set_global_prng_offset(0)
    npt.assert_almost_equal(gprng.rand(1), [0.5488135])
