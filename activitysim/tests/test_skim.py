# ActivitySim
# See full license in LICENSE.txt.

import numpy as np
import pandas as pd
import numpy.testing as npt
import pandas.util.testing as pdt
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


def test_skim_nans(data):
    sk = skim.Skim(data)

    orig = [5, np.nan, 1, 2]
    dest = [np.nan, 9, 6, 4]

    npt.assert_array_equal(
        sk.get(orig, dest),
        [np.nan, np.nan, 16, 24])


def test_skims(data):

    skims = skim.Skims()
    skims.set_keys("taz_l", "taz_r")

    sk = skim.Skim(data)
    sk2 = skim.Skim(data)

    skims['AM'] = sk
    skims['PM'] = sk2

    df = pd.DataFrame({
        "taz_l": [1, 9, 4],
        "taz_r": [2, 3, 7],
    })

    skims.set_df(df)

    pdt.assert_series_equal(
        skims["AM"],
        pd.Series(
            [12, 93, 47],
            index=[0, 1, 2]
        ).astype('float64')
    )

    pdt.assert_series_equal(
        skims["PM"],
        pd.Series(
            [12, 93, 47],
            index=[0, 1, 2]
        ).astype('float64')
    )


def test_3dskims(data):

    skims = skim.Skims()
    skims.set_keys("taz_l", "taz_r")

    sk = skim.Skim(data)
    sk2 = skim.Skim(data)

    skims.set_3d("SOV", "AM", sk)
    skims.set_3d("SOV", "PM", sk2)

    skims3d = skim.Skims3D(skims, "period")

    df = pd.DataFrame({
        "taz_l": [1, 9, 4],
        "taz_r": [2, 3, 7],
        "period": ["AM", "PM", "AM"]
    })

    skims3d.set_df(df)

    pdt.assert_series_equal(
        skims3d["SOV"],
        pd.Series(
            [12, 93, 47],
            index=[0, 1, 2]
        ),
        check_dtype=False
    )
