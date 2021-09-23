# ActivitySim
# See full license in LICENSE.txt.

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest

from .. import skim_dictionary


@pytest.fixture
def data():
    return np.arange(100, dtype="int").reshape((10, 10))


class FakeSkimInfo(object):
    def __init__(self):
        self.offset_map = None


def test_skims(data):

    # ROW_MAJOR_LAYOUT
    omx_shape = (10, 10)
    num_skims = 2
    skim_data_shape = (num_skims,) + omx_shape
    skim_data = np.zeros(skim_data_shape, dtype=int)
    skim_data[0, :, :] = data
    skim_data[1, :, :] = data * 10

    skim_info = FakeSkimInfo()
    skim_info.block_offsets = {"AM": 0, "PM": 1}
    skim_info.omx_shape = omx_shape
    skim_info.dtype_name = "int"

    skim_dict = skim_dictionary.SkimDict("taz", skim_info, skim_data)
    skim_dict.offset_mapper.set_offset_int(0)  # default is -1
    skims = skim_dict.wrap("taz_l", "taz_r")

    df = pd.DataFrame(
        {
            "taz_l": [1, 9, 4],
            "taz_r": [2, 3, 7],
        }
    )

    skims.set_df(df)

    pdt.assert_series_equal(
        skims["AM"], pd.Series([12, 93, 47], index=[0, 1, 2]).astype(data.dtype)
    )

    pdt.assert_series_equal(
        skims["PM"], pd.Series([120, 930, 470], index=[0, 1, 2]).astype(data.dtype)
    )


def test_3dskims(data):

    # ROW_MAJOR_LAYOUT
    omx_shape = (10, 10)
    num_skims = 2
    skim_data_shape = (num_skims,) + omx_shape
    skim_data = np.zeros(skim_data_shape, dtype=int)
    skim_data[0, :, :] = data
    skim_data[1, :, :] = data * 10

    skim_info = FakeSkimInfo()
    skim_info.block_offsets = {("SOV", "AM"): 0, ("SOV", "PM"): 1}
    skim_info.omx_shape = omx_shape
    skim_info.dtype_name = "int"
    skim_info.key1_block_offsets = {"SOV": 0}

    skim_dict = skim_dictionary.SkimDict("taz", skim_info, skim_data)
    skim_dict.offset_mapper.set_offset_int(0)  # default is -1
    skims3d = skim_dict.wrap_3d(orig_key="taz_l", dest_key="taz_r", dim3_key="period")

    df = pd.DataFrame(
        {"taz_l": [1, 9, 4], "taz_r": [2, 3, 7], "period": ["AM", "PM", "AM"]}
    )

    skims3d.set_df(df)

    pdt.assert_series_equal(
        skims3d["SOV"], pd.Series([12, 930, 47], index=[0, 1, 2]), check_dtype=False
    )
