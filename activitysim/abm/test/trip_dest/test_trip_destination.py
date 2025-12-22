from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from activitysim import abm  # noqa: F401
from activitysim.core import workflow as wf


def test_trip_destination(tmp_path: Path):
    shutil.copytree(
        Path(__file__).parent.joinpath("configs"), tmp_path.joinpath("configs")
    )
    shutil.copytree(Path(__file__).parent.joinpath("data"), tmp_path.joinpath("data"))

    state = wf.State.make_default(working_dir=tmp_path)

    # init tours
    tours = pd.read_csv(
        tmp_path / state.filesystem.data_dir[0] / "tours.csv"
    ).set_index("tour_id")
    state.add_table("tours", tours)
    state.tracing.register_traceable_table("tours", tours)
    state.get_rn_generator().add_channel("tours", tours)

    # init trips
    trips = pd.read_csv(
        tmp_path / state.filesystem.data_dir[0] / "trips.csv"
    ).set_index("trip_id")
    state.add_table("trips", trips)
    state.tracing.register_traceable_table("trips", trips)
    state.get_rn_generator().add_channel("trips", trips)

    state.run.all()

    out_trips = state.get_dataframe("trips")

    # logsums are generated for intermediate trips only
    assert out_trips["destination_logsum"].isna().tolist() == [
        True,
        False,
        True,
        True,
        False,
        True,
    ]
