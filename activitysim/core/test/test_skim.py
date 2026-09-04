# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import secrets

import numpy as np
import numpy.testing as npt
import openmatrix
import pandas as pd
import pandas.testing as pdt
import pytest
import sharrow as sh

from activitysim.core import skim_dictionary, workflow


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

    skim_dict = skim_dictionary.SkimDict(
        workflow.State().default_settings(), "taz", skim_info, skim_data
    )
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

    skim_dict = skim_dictionary.SkimDict(
        workflow.State().default_settings(), "taz", skim_info, skim_data
    )
    skim_dict.offset_mapper.set_offset_int(0)  # default is -1
    skims3d = skim_dict.wrap_3d(orig_key="taz_l", dest_key="taz_r", dim3_key="period")

    df = pd.DataFrame(
        {"taz_l": [1, 9, 4], "taz_r": [2, 3, 7], "period": ["AM", "PM", "AM"]}
    )

    skims3d.set_df(df)

    pdt.assert_series_equal(
        skims3d["SOV"], pd.Series([12, 930, 47], index=[0, 1, 2]), check_dtype=False
    )


# ---------------------------------------------------------------------------
# Helpers for the reload-path tests
# ---------------------------------------------------------------------------

N_ZONES = 5
TIME_PERIODS = ["AM", "PM"]
ZONE_IDS = np.array([100, 200, 300, 400, 500])


def _create_test_omx(
    filepath, n_zones=N_ZONES, time_periods=TIME_PERIODS, zone_ids=None
):
    """Write a small OMX file with 3-D and 2-D skim matrices.

    Matrices written (all float32, shape n_zones x n_zones):
        SOV_TIME__AM, SOV_TIME__PM   - 3-D (time-dependent)
        HOV_COST__AM, HOV_COST__PM   - 3-D (time-dependent)
        DIST                         - 2-D (time-agnostic)
        UNUSED_VAR__AM, UNUSED_VAR__PM - 3-D, will be dropped
        UNUSED_2D                    - 2-D, will be dropped

    Values are deterministic so assertions are reproducible.
    """
    if zone_ids is None:
        zone_ids = np.arange(n_zones, dtype=int)
    with openmatrix.open_file(str(filepath), mode="w") as out:
        shp = np.empty(2, dtype=int)
        shp[0] = n_zones
        shp[1] = n_zones
        out.root._v_attrs.SHAPE = shp

        out.create_carray("/lookup", "zone_id", obj=zone_ids)

        # 3-D variable: SOV_TIME
        for i, tp in enumerate(time_periods):
            mat = (
                np.arange(n_zones * n_zones).reshape(n_zones, n_zones) * (i + 1)
            ).astype(np.float32)
            out.create_carray("/data", f"SOV_TIME__{tp}", obj=mat)

        # 3-D variable: HOV_COST
        for i, tp in enumerate(time_periods):
            mat = (
                np.arange(n_zones * n_zones).reshape(n_zones, n_zones) * (i + 10)
            ).astype(np.float32)
            out.create_carray("/data", f"HOV_COST__{tp}", obj=mat)

        # 2-D variable: DIST
        dist = (
            np.arange(n_zones * n_zones).reshape(n_zones, n_zones).astype(np.float32)
            * 0.5
        )
        out.create_carray("/data", "DIST", obj=dist)

        # 3-D variable that will be "unused": UNUSED_VAR
        for i, tp in enumerate(time_periods):
            mat = np.ones((n_zones, n_zones), dtype=np.float32) * 999
            out.create_carray("/data", f"UNUSED_VAR__{tp}", obj=mat)

        # 2-D variable that will be "unused": UNUSED_2D
        out.create_carray(
            "/data",
            "UNUSED_2D",
            obj=np.ones((n_zones, n_zones), dtype=np.float32) * 888,
        )


def _load_omx_dataset(omx_path, time_periods=TIME_PERIODS):
    """Load and compute a full dataset from OMX via sharrow.

    Returns a numpy-backed dataset (dask arrays are computed before
    the OMX handle is closed).
    """
    with openmatrix.open_file(str(omx_path), mode="r") as f:
        ds = sh.dataset.from_omx_3d(
            f,
            index_names=("otaz", "dtaz", "time_period"),
            time_periods=time_periods,
            max_float_precision=32,
        )
        # HDF5/PyTables is not thread-safe; force single-threaded compute
        # to avoid H5ARRAYOreadSlice errors on Linux CI.
        ds = ds.compute(scheduler="synchronous")
    return ds


def _drop_vars(ds, drop_names):
    """Simulate _drop_unused_names by dropping specified variables."""
    return ds.drop_vars([n for n in drop_names if n in ds])


@pytest.fixture
def omx_env(tmp_path):
    """OMX file with zero-based zone IDs."""
    omx_path = tmp_path / "test_skims.omx"
    _create_test_omx(omx_path)
    # shm tokens are process-global, so we generate a unique one per test to avoid collisions
    token = "test_reload_" + secrets.token_hex(5)
    return omx_path, token


@pytest.fixture
def omx_env_with_zones(tmp_path):
    """OMX file with non-trivial zone IDs (100, 200, …)."""
    omx_path = tmp_path / "test_skims_zones.omx"
    _create_test_omx(omx_path, zone_ids=ZONE_IDS)
    # shm tokens are process-global, so we generate a unique one per test to avoid collisions
    token = "test_zones_" + secrets.token_hex(5)
    return omx_path, token


# ---------------------------------------------------------------------------
# Tests for the load=False skim reload path via _finalize_skim_dataset
# ---------------------------------------------------------------------------


class TestFinalizeSkimDataset:
    """Unit tests that call ``_finalize_skim_dataset`` directly,
    exercising the real production code paths for zone alignment,
    shared-memory reload, and digital encoding.
    """

    @staticmethod
    def _finalize(
        d,
        omx_path,
        *,
        store_skims_in_shm,
        backing,
        land_use_zone_id=None,
        land_use_index=None,
        zone_system=1,
        skim_digital_encoding=None,
        omx_ignore_patterns=None,
    ):
        """Convenience wrapper around the extracted production function."""
        from activitysim.core.skim_dataset import _finalize_skim_dataset

        omx_handles = [openmatrix.open_file(str(omx_path), mode="r")]
        return _finalize_skim_dataset(
            d,
            omx_file_paths=[omx_path],
            omx_file_handles=omx_handles,
            time_periods=TIME_PERIODS,
            land_use_zone_id=land_use_zone_id,
            land_use_index=land_use_index,
            zone_system=zone_system,
            store_skims_in_shm=store_skims_in_shm,
            backing=backing,
            skim_digital_encoding=skim_digital_encoding or [],
            omx_ignore_patterns=omx_ignore_patterns,
        )

    # -- store_skims_in_shm=False (local memory path) ---------------------

    def test_local_memory_path(self, omx_env):
        """When store_skims_in_shm is False, _finalize returns an
        in-process dataset with digital encoding applied."""
        omx_path, token = omx_env

        d = _load_omx_dataset(omx_path)
        d = _drop_vars(d, ["UNUSED_VAR", "UNUSED_2D"])

        result = self._finalize(
            d,
            omx_path,
            store_skims_in_shm=False,
            backing=token,
            land_use_zone_id=np.arange(N_ZONES),
            land_use_index=np.arange(N_ZONES),
        )

        assert not result.shm.is_shared_memory
        assert "SOV_TIME" in result
        assert "DIST" in result
        assert "UNUSED_VAR" not in result

    def test_local_memory_path_materializes_before_closing_omx(self, omx_env):
        """Lazy OMX skims remain usable after finalization closes their handle."""
        omx_path, token = omx_env

        from activitysim.core.skim_dataset import _finalize_skim_dataset

        omx_handle = openmatrix.open_file(str(omx_path), mode="r")
        d = sh.dataset.from_omx_3d(
            omx_handle,
            index_names=("otaz", "dtaz", "time_period"),
            time_periods=TIME_PERIODS,
            max_float_precision=32,
        )

        result = _finalize_skim_dataset(
            d,
            omx_file_paths=[omx_path],
            omx_file_handles=[omx_handle],
            time_periods=TIME_PERIODS,
            land_use_zone_id=np.arange(N_ZONES),
            land_use_index=np.arange(N_ZONES),
            zone_system=1,
            store_skims_in_shm=False,
            backing=token,
            skim_digital_encoding=[],
        )

        assert not omx_handle.isopen
        npt.assert_array_equal(
            result["DIST"].values,
            np.arange(N_ZONES * N_ZONES).reshape(N_ZONES, N_ZONES) * 0.5,
        )

    # -- store_skims_in_shm=True, no realignment (reload path) ------------

    def test_shm_reload_3d_skims(self, omx_env):
        """3-D skims are correctly loaded into shared memory via the
        deferred reload path (load=False + reload_from_omx_3d)."""
        omx_path, token = omx_env

        d = _load_omx_dataset(omx_path)
        ref = d.copy(deep=True)  # reference before finalization

        result = self._finalize(
            d,
            omx_path,
            store_skims_in_shm=True,
            backing=token,
            land_use_zone_id=np.arange(N_ZONES),
            land_use_index=np.arange(N_ZONES),
        )

        # Data in shared memory must match the reference
        for var in ["SOV_TIME", "HOV_COST", "DIST", "UNUSED_VAR", "UNUSED_2D"]:
            npt.assert_array_equal(
                result[var].values,
                ref[var].values,
                err_msg=f"{var} mismatch after shm reload",
            )

    def test_shm_reload_with_dropped_skims(self, omx_env):
        """Dropped (unused) skims are excluded during reload via
        the auto-generated reload_ignore list."""
        omx_path, token = omx_env

        d = _load_omx_dataset(omx_path)
        d = _drop_vars(d, ["UNUSED_VAR", "UNUSED_2D"])
        ref = d.copy(deep=True)

        result = self._finalize(
            d,
            omx_path,
            store_skims_in_shm=True,
            backing=token,
            land_use_zone_id=np.arange(N_ZONES),
            land_use_index=np.arange(N_ZONES),
        )

        # Dropped variables must not appear
        assert "UNUSED_VAR" not in result
        assert "UNUSED_2D" not in result

        # Kept variables must match the reference
        for var in ["SOV_TIME", "HOV_COST", "DIST"]:
            npt.assert_array_equal(
                result[var].values,
                ref[var].values,
                err_msg=f"{var} data mismatch after reload with drops",
            )

    def test_shm_reload_digital_encoding_after(self, omx_env):
        """Digital encoding is applied AFTER reload so raw OMX values
        are written first, then compressed in place."""
        omx_path, token = omx_env

        d = _load_omx_dataset(omx_path)
        d = _drop_vars(d, ["UNUSED_VAR", "UNUSED_2D"])

        result = self._finalize(
            d,
            omx_path,
            store_skims_in_shm=True,
            backing=token,
            land_use_zone_id=np.arange(N_ZONES),
            land_use_index=np.arange(N_ZONES),
            skim_digital_encoding=[{"regex": "SOV_TIME", "bitwidth": 16}],
        )

        # SOV_TIME should carry digital_encoding metadata
        assert "digital_encoding" in result["SOV_TIME"].attrs
        # Underlying dtype should have changed
        assert result["SOV_TIME"].dtype != np.float32

        # HOV_COST was not matched by regex — should be unaffected
        assert "digital_encoding" not in result["HOV_COST"].attrs
        npt.assert_array_equal(
            result["HOV_COST"].values,
            d["HOV_COST"].values,
        )

    # -- Zone alignment (land_use zone IDs ≠ OMX zone IDs) ----------------

    def test_zone_alignment_remaps_coords(self, omx_env_with_zones):
        """When OMX zones are [100,200,300,400,500] but land_use expects
        zero-based indices, _finalize rewrites otaz/dtaz coordinates."""
        omx_path, token = omx_env_with_zones

        d = _load_omx_dataset(omx_path)
        d = _drop_vars(d, ["UNUSED_VAR", "UNUSED_2D"])
        land_use_index = np.arange(N_ZONES)

        result = self._finalize(
            d,
            omx_path,
            store_skims_in_shm=False,
            backing=token,
            land_use_zone_id=ZONE_IDS,
            land_use_index=land_use_index,
        )

        # Coordinates should now be zero-based
        npt.assert_array_equal(result["otaz"].values, land_use_index)
        npt.assert_array_equal(result["dtaz"].values, land_use_index)
        assert result["otaz"].attrs["preprocessed"] == "zero-based-contiguous"

    # -- Lookup equivalence: local vs shared memory -----------------------

    def test_lookup_equivalence_2d(self, omx_env_with_zones):
        """SkimDataset.lookup on a 2-D skim yields identical results
        for the local-memory and shared-memory paths."""
        omx_path, token_local = omx_env_with_zones
        token_shm = token_local + "_shm"

        from activitysim.core.skim_dataset import SkimDataset

        d = _load_omx_dataset(omx_path)
        d = _drop_vars(d, ["UNUSED_VAR", "UNUSED_2D"])

        d_local = self._finalize(
            d.copy(deep=True),
            omx_path,
            store_skims_in_shm=False,
            backing=token_local,
            land_use_zone_id=ZONE_IDS,
            land_use_index=np.arange(N_ZONES),
        )
        d_shm = self._finalize(
            d.copy(deep=True),
            omx_path,
            store_skims_in_shm=True,
            backing=token_shm,
            land_use_zone_id=ZONE_IDS,
            land_use_index=np.arange(N_ZONES),
        )

        orig = np.array([0, 1, 3, 4, 2])
        dest = np.array([1, 3, 0, 2, 4])

        result_local = SkimDataset(d_local).lookup(orig, dest, "DIST")
        result_shm = SkimDataset(d_shm).lookup(orig, dest, "DIST")

        pdt.assert_series_equal(result_local, result_shm, check_names=False)
        # Sanity: DIST = arange(25).reshape(5,5) * 0.5
        npt.assert_array_almost_equal(result_local.values, [0.5, 4.0, 7.5, 11.0, 7.0])

    def test_lookup_equivalence_3d(self, omx_env_with_zones):
        """SkimDataset.lookup on a 3-D skim (with time period key) yields
        identical results for local-memory and shared-memory paths."""
        omx_path, token_local = omx_env_with_zones
        token_shm = token_local + "_shm"

        from activitysim.core.skim_dataset import SkimDataset

        d = _load_omx_dataset(omx_path)
        d = _drop_vars(d, ["UNUSED_VAR", "UNUSED_2D"])

        d_local = self._finalize(
            d.copy(deep=True),
            omx_path,
            store_skims_in_shm=False,
            backing=token_local,
            land_use_zone_id=ZONE_IDS,
            land_use_index=np.arange(N_ZONES),
        )
        d_shm = self._finalize(
            d.copy(deep=True),
            omx_path,
            store_skims_in_shm=True,
            backing=token_shm,
            land_use_zone_id=ZONE_IDS,
            land_use_index=np.arange(N_ZONES),
        )

        orig = np.array([0, 2, 4])
        dest = np.array([1, 3, 0])

        for tp in TIME_PERIODS:
            result_local = SkimDataset(d_local).lookup(orig, dest, ("SOV_TIME", tp))
            result_shm = SkimDataset(d_shm).lookup(orig, dest, ("SOV_TIME", tp))
            pdt.assert_series_equal(result_local, result_shm, check_names=False)

        # Spot-check AM: SOV_TIME__AM = arange(25).reshape(5,5) * 1
        am = SkimDataset(d_local).lookup(orig, dest, ("SOV_TIME", "AM"))
        npt.assert_array_almost_equal(am.values, [1.0, 13.0, 20.0])

    def test_wrapper_3d_equivalence(self, omx_env_with_zones):
        """DatasetWrapper (wrap_3d) lookups with mixed time periods yield
        identical results for local-memory and shared-memory paths."""
        omx_path, token_local = omx_env_with_zones
        token_shm = token_local + "_shm"

        from activitysim.core.skim_dataset import SkimDataset

        d = _load_omx_dataset(omx_path)
        d = _drop_vars(d, ["UNUSED_VAR", "UNUSED_2D"])

        d_local = self._finalize(
            d.copy(deep=True),
            omx_path,
            store_skims_in_shm=False,
            backing=token_local,
            land_use_zone_id=ZONE_IDS,
            land_use_index=np.arange(N_ZONES),
        )
        d_shm = self._finalize(
            d.copy(deep=True),
            omx_path,
            store_skims_in_shm=True,
            backing=token_shm,
            land_use_zone_id=ZONE_IDS,
            land_use_index=np.arange(N_ZONES),
        )

        df = pd.DataFrame(
            {
                "otaz": [0, 2, 4, 1, 3],
                "dtaz": [1, 3, 0, 4, 2],
                "time_period": ["AM", "PM", "AM", "PM", "AM"],
            }
        )

        wrap_local = SkimDataset(d_local).wrap_3d("otaz", "dtaz", "time_period")
        wrap_shm = SkimDataset(d_shm).wrap_3d("otaz", "dtaz", "time_period")
        wrap_local.set_df(df)
        wrap_shm.set_df(df)

        for var in ["SOV_TIME", "HOV_COST"]:
            pdt.assert_series_equal(wrap_local[var], wrap_shm[var])

        # Spot-check mixed time periods:
        # row 0: SOV_TIME__AM at (0,1) = 1*1 = 1.0
        # row 1: SOV_TIME__PM at (2,3) = 13*2 = 26.0
        # row 2: SOV_TIME__AM at (4,0) = 20*1 = 20.0
        # row 3: SOV_TIME__PM at (1,4) = 9*2 = 18.0
        # row 4: SOV_TIME__AM at (3,2) = 17*1 = 17.0
        npt.assert_array_almost_equal(
            wrap_local["SOV_TIME"].values,
            [1.0, 26.0, 20.0, 18.0, 17.0],
        )
