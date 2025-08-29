import numpy as np
import pandas as pd
import openmatrix as omx
import os
from pathlib import Path

from activitysim.abm.models.trip_matrices import (
    write_matrices,
    WriteTripMatricesSettings,
    MatrixSettings,
    MatrixTableSettings,
)
from activitysim.core import workflow


def get_settings(omx_file_name):
    # Settings: same table name twice, second time with custom destination column
    settings = WriteTripMatricesSettings(
        MATRICES=[
            MatrixSettings(
                file_name=Path(omx_file_name),
                is_tap=False,
                tables=[
                    MatrixTableSettings(
                        name="DRIVEALONE_VOT1",
                        data_field="DRIVEALONE_VOT1",
                    ),
                    MatrixTableSettings(
                        name="DRIVEALONE_VOT1",
                        data_field="PNR_DRIVEALONE_OUT",
                        origin="origin",
                        destination="pnr_zone_id",
                    ),
                    MatrixTableSettings(
                        name="WALK_TRN_OUT",
                        data_field="WALK_TRN_OUT",
                        origin="origin",
                        destination="destination",
                    ),
                    MatrixTableSettings(
                        name="WALK_TRN_OUT",
                        data_field="PNR_TRN_OUT",
                        origin="pnr_zone_id",
                        destination="destination",
                    ),
                ],
            )
        ]
    )
    return settings


def test_write_matrices_one_zone():
    state = workflow.State.make_default(__file__)

    # Zones domain (TAZ)
    zone_index = pd.Index([1, 2, 3, 4], name="TAZ")

    # Trips with both standard drive and PNR drive legs
    trips_df = pd.DataFrame(
        {
            "origin": [1, 1, 2],
            "destination": [2, 2, 4],
            "pnr_zone_id": [3, -1, 4],
            # we have more flags below than trips, but that's ok for testing
            "DRIVEALONE_VOT1": [0, 1, 1],
            "PNR_DRIVEALONE_OUT": [1, 0, 1],
            "WALK_TRN_OUT": [1, 0, 1],
            "PNR_TRN_OUT": [1, 0, 1],
            "sample_rate": [0.5, 0.5, 0.5],
        }
    )

    # Run writer
    write_matrices(
        state=state,
        trips_df=trips_df,
        zone_index=zone_index,
        model_settings=get_settings("trips_one_zone.omx"),
        is_tap=False,
    )

    # Validate output
    omx_path = os.path.join(os.path.dirname(__file__), "output", "trips_one_zone.omx")
    assert os.path.exists(omx_path), "OMX file not created"

    with omx.open_file(omx_path, mode="r") as f:
        # Mapping should reflect zone_index order [1,2,3,4]
        mapping = f.mapping("TAZ")
        assert list(mapping) == [1, 2, 3, 4]

        # --- checking driving output
        assert "DRIVEALONE_VOT1" in f.list_matrices(), "Expected matrix not found"
        data = f["DRIVEALONE_VOT1"][:]

        # Expected totals:
        # - zones (1->2): 1 drive = 1
        # - zones (1->3): 1 PNR drive leg = 1
        # - zones (2->4): 1 PNR drive leg + 1 drive = 2
        # and double them based on sample rate
        assert data.shape == (4, 4)
        # indices are zero-based positions for labels [1,2,3,4]
        assert data[0, 1] == 2  # 1->2
        assert data[0, 2] == 2  # 1->3
        assert data[1, 3] == 4  # 2->4
        # everything else remains zero
        zero_mask = np.ones_like(data, dtype=bool)
        zero_mask[0, 1] = False
        zero_mask[0, 2] = False
        zero_mask[1, 3] = False
        assert np.all(data[zero_mask] == 0.0)

        # ---- checking transit output
        assert "WALK_TRN_OUT" in f.list_matrices(), "Expected matrix not found"
        data = f["WALK_TRN_OUT"][:]

        # Expected totals:
        # - zones (1->2): 1 wlk trn
        # - zones (2->4): 1 wlk trn
        # - zones (3->2): 1 PNR trn leg
        # - zones (4->4): 1 PNR trn leg
        # and double them based on sample rate
        assert data.shape == (4, 4)
        assert data[0, 1] == 2  # 1->2
        assert data[1, 3] == 2  # 2->4
        assert data[2, 1] == 2  # 3->2
        assert data[3, 3] == 2  # 4->4
        # everything else remains zero
        zero_mask = np.ones_like(data, dtype=bool)
        zero_mask[0, 1] = False
        zero_mask[1, 3] = False
        zero_mask[2, 1] = False
        zero_mask[3, 3] = False
        assert np.all(data[zero_mask] == 0.0)


# ...existing code...
def test_write_matrices_two_zone():
    state = workflow.State.make_default(__file__)

    # land_use table for MAZ -> TAZ mapping
    # MAZ: 101,102 -> TAZ 1; 103,104 -> TAZ 2
    land_use = pd.DataFrame(
        {"TAZ": [1, 1, 2, 2]},
        index=pd.Index([101, 102, 103, 104], name="MAZ"),
    )
    state.add_table("land_use", land_use)

    # TAZ domain for output
    zone_index = pd.Index([1, 2], name="TAZ")

    # Trips in MAZ space, with both standard drive and PNR drive legs
    trips_df = pd.DataFrame(
        {
            "origin": [101, 102, 103],
            "destination": [102, 104, 104],
            "pnr_zone_id": [103, 0, 104],
            # we have more flags below than trips, but that's ok for testing
            "DRIVEALONE_VOT1": [0, 1, 1],
            "PNR_DRIVEALONE_OUT": [1, 0, 1],
            "WALK_TRN_OUT": [1, 0, 1],
            "PNR_TRN_OUT": [1, 0, 1],
            "sample_rate": [0.5, 0.5, 0.5],
        }
    )

    # Run writer
    write_matrices(
        state=state,
        trips_df=trips_df,
        zone_index=zone_index,
        model_settings=get_settings("trips_two_zone.omx"),
        is_tap=False,
    )

    # Validate output
    omx_path = os.path.join(os.path.dirname(__file__), "output", "trips_two_zone.omx")
    assert os.path.exists(omx_path), "OMX file not created"

    with omx.open_file(omx_path, mode="r") as f:
        mapping = f.mapping("TAZ")
        assert list(mapping) == [1, 2]

        # ---- checking drive output
        assert "DRIVEALONE_VOT1" in f.list_matrices(), "Expected matrix not found"
        data = f["DRIVEALONE_VOT1"][:]

        # Expected before expansion weighting:
        # Standard drive (MAZ->TAZ):
        # 102->104 => 1->2 : 1
        # 103->104 => 2->2 : 1
        # PNR drive legs (origin->pnr_zone_id):
        # 101->103 => 1->2 : +1
        # 103->104 => 2->2 : +1
        # Totals: (1,2)=2, (2,2)=2
        # With sample_rate=0.5, values are doubled:
        assert data.shape == (2, 2)
        assert data[0, 1] == 4  # 1->2
        assert data[1, 1] == 4  # 2->2

        # All other cells zero
        zero_mask = np.ones_like(data, dtype=bool)
        zero_mask[0, 1] = False
        zero_mask[1, 1] = False
        assert np.all(data[zero_mask] == 0.0)

        # ---- checking transit output
        assert "WALK_TRN_OUT" in f.list_matrices(), "Expected matrix not found"
        data = f["WALK_TRN_OUT"][:]

        # Expected before expansion weighting:
        # regular transit
        # 101->102 => 1->1 : 1
        # 103->104 => 2->2 : 1
        # pnr transit legs
        # 103->102 => 2->1 : +1
        # 104->104 => 2->2 : +1
        # Totals: (1,1)=1, (2,1)=1, (2,2)=2
        # With sample_rate=0.5, values are doubled:
        assert data.shape == (2, 2)
        assert data[0, 0] == 2  # 1->1
        assert data[1, 0] == 2  # 2->1
        assert data[1, 1] == 4  # 2->2

        # All other cells zero
        zero_mask = np.ones_like(data, dtype=bool)
        zero_mask[0, 0] = False
        zero_mask[1, 0] = False
        zero_mask[1, 1] = False
        assert np.all(data[zero_mask] == 0.0)
