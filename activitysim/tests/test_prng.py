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

    r = rng.get_global_prng()
