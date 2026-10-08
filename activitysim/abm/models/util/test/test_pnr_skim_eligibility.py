"""Exercise park-and-ride eligibility with both concrete skim backends."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from activitysim.abm.models.park_and_ride_lot_choice import (
    ParkAndRideLotChoiceSettings,
    filter_chooser_to_transit_accessible_destinations,
)
from activitysim.core import workflow
from activitysim.core.skim_dataset import SkimDataset
from activitysim.core.skim_dictionary import SkimDict


@pytest.fixture(params=["numpy", "sharrow"])
def eligibility_skims(request):
    """Use identical zero-based zones and accessibility in both backends."""
    walk = np.zeros((5, 5), dtype=np.float32)
    transit = np.zeros((5, 5, 2), dtype=np.float32)
    walk[0, 2] = 10
    # Missing and negative skim values must not count as transit access.
    walk[0, 3] = np.nan
    walk[1, 4] = -1
    transit[1, 3, 0] = 20
    transit[0, 4, 1] = 30

    if request.param == "sharrow":
        return SkimDataset(
            xr.Dataset(
                {
                    "WALK": (("otaz", "dtaz"), walk),
                    "TRANSIT": (("otaz", "dtaz", "time_period"), transit),
                },
                coords={"time_period": ["AM", "PM"]},
            )
        )

    skim_info = SimpleNamespace(
        offset_map=None,
        omx_shape=(5, 5),
        dtype_name="float32",
        block_offsets={"WALK": 0, ("TRANSIT", "AM"): 1, ("TRANSIT", "PM"): 2},
    )
    skims = SkimDict(
        workflow.State().default_settings(),
        "taz",
        skim_info,
        np.stack([walk, transit[:, :, 0], transit[:, :, 1]]),
    )
    skims.offset_mapper.set_offset_int(0)
    return skims


@pytest.mark.parametrize(
    "key, expected",
    [
        ("WALK", True),
        (("TRANSIT", "AM"), True),
        (("TRANSIT", "PM"), True),
        ("MISSING", False),
        (("MISSING", "AM"), False),
        (("TRANSIT", "MD"), False),
        ("TRANSIT", False),
        (("WALK", "AM"), False),
        (("TRANSIT", "AM", "extra"), False),
        ("time_period", False),
    ],
)
def test_skim_membership(eligibility_skims, key, expected):
    """Validation agrees across backends without reading or recording skim usage."""
    assert (key in eligibility_skims) is expected
    assert eligibility_skims.get_skim_usage() == set()


def _filter_choosers(skims, skim_names, destinations=None):
    """Filter repeated, unordered destinations from two possible lot locations."""
    choosers = pd.DataFrame(
        {"destination": [4, 2, 3, 2, 0] if destinations is None else destinations},
        index=pd.Index([100, 101, 102, 103, 104], name="tour_id"),
    )
    lots = pd.DataFrame(index=pd.Index([0, 1], name="zone_id"))
    settings = ParkAndRideLotChoiceSettings(
        SPEC="unused.csv",
        LANDUSE_PNR_SPACES_COLUMN="pnr_spaces",
        TRANSIT_SKIMS_FOR_ELIGIBILITY=skim_names,
    )
    return filter_chooser_to_transit_accessible_destinations(
        state=None,
        choosers=choosers,
        land_use=lots,
        pnr_alts=lots,
        network_los=SimpleNamespace(get_default_skim_dict=lambda: skims),
        model_settings=settings,
        choosers_dest_col_name="destination",
    )


@pytest.mark.parametrize(
    "skim_names, expected_ids",
    [
        (["WALK"], [101, 103]),
        (["TRANSIT__AM"], [102]),
        (["TRANSIT__PM"], [100]),
        (["WALK", "TRANSIT__AM"], [101, 102, 103]),
        (["TRANSIT__AM", "TRANSIT__PM"], [100, 102]),
    ],
)
def test_filter_skim_backends(eligibility_skims, skim_names, expected_ids):
    """Accept access from any lot or configured skim, preserving chooser order."""
    result = _filter_choosers(eligibility_skims, skim_names)
    expected = pd.DataFrame(
        {"destination": [4, 2, 3, 2, 0]},
        index=pd.Index([100, 101, 102, 103, 104], name="tour_id"),
    ).loc[expected_ids]
    pd.testing.assert_frame_equal(result, expected)


def test_no_accessible_destinations(eligibility_skims):
    """An available skim with no positive lot-to-destination values rejects all tours."""
    result = _filter_choosers(eligibility_skims, ["WALK"], [0, 1, 3, 4, 0])
    expected = pd.DataFrame(
        {"destination": pd.Series(dtype="int64")},
        index=pd.Index([], dtype="int64", name="tour_id"),
    )
    pd.testing.assert_frame_equal(result, expected)


def test_invalid_skim_after_valid_skim(eligibility_skims):
    """An earlier accessible skim must not bypass validation of later settings."""
    with pytest.raises(ValueError, match="Skim 'MISSING' not found"):
        _filter_choosers(eligibility_skims, ["WALK", "MISSING"])


@pytest.mark.parametrize(
    "skim_name", ["MISSING", "MISSING__AM", "TRANSIT__MD", "TRANSIT", "WALK__AM"]
)
def test_invalid_eligibility_skim(eligibility_skims, skim_name):
    """Missing cores and invalid time periods retain an actionable settings error."""
    with pytest.raises(ValueError, match="TRANSIT_SKIMS_FOR_ELIGIBILITY"):
        _filter_choosers(eligibility_skims, [skim_name])
