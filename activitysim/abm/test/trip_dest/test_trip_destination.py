from __future__ import annotations

import shutil
from unittest.mock import Mock
from pathlib import Path

import pandas as pd
import pytest

from activitysim.abm.models import trip_destination as td
from activitysim.core import chunk

from activitysim import abm  # noqa: F401
from activitysim.core import workflow as wf


def run_trip_destination(
    tmp_path: Path,
    explicit_chunk: float | None = None,
    repeat_work_tours: int = 1,
    chunk_training_mode: str = chunk.MODE_EXPLICIT,
):
    shutil.copytree(
        Path(__file__).parent.joinpath("configs"), tmp_path.joinpath("configs")
    )
    shutil.copytree(Path(__file__).parent.joinpath("data"), tmp_path.joinpath("data"))

    if explicit_chunk is not None:
        with (tmp_path / "configs" / "trip_destination.yaml").open("a") as stream:
            stream.write(f"\nexplicit_chunk: {explicit_chunk}\n")

    state = wf.State.make_default(working_dir=tmp_path)

    state.settings.chunk_training_mode = chunk_training_mode
    if chunk_training_mode != chunk.MODE_EXPLICIT:
        state.settings.chunk_size = 1_000_000

    # init tours
    tours = pd.read_csv(
        tmp_path / state.filesystem.data_dir[0] / "tours.csv"
    ).set_index("tour_id")
    base_tour = tours.loc[[500]]
    tours = pd.concat(
        [tours]
        + [base_tour.rename(index={500: 510 + i}) for i in range(1, repeat_work_tours)]
    ).sort_index()
    state.add_table("tours", tours)
    state.tracing.register_traceable_table("tours", tours)
    state.get_rn_generator().add_channel("tours", tours)

    # init trips
    trips = pd.read_csv(
        tmp_path / state.filesystem.data_dir[0] / "trips.csv"
    ).set_index("trip_id")
    base_trips = trips[trips.tour_id == 500]
    repeated_trips = [trips]
    for i in range(1, repeat_work_tours):
        trip_copy = base_trips.copy()
        trip_copy.index += 100_000 * i
        trip_copy["tour_id"] = 510 + i
        repeated_trips.append(trip_copy)
    trips = pd.concat(repeated_trips).sort_index()
    state.add_table("trips", trips)
    state.tracing.register_traceable_table("trips", trips)
    state.get_rn_generator().add_channel("trips", trips)

    state.run.all()

    return state.get_dataframe("trips")


def test_trip_destination(tmp_path: Path):
    out_trips = run_trip_destination(tmp_path)

    # logsums are generated for intermediate trips only
    assert out_trips["destination_logsum"].isna().tolist() == [
        True,
        False,
        True,
        True,
        False,
        True,
    ]


def test_trip_destination_chunked_logsums_match_unchunked(tmp_path: Path):
    # Repeat one tour so each work-purpose presample has enough chooser rows to
    # exercise the outer TAZ-to-MAZ pipeline chunker.
    unchunked = run_trip_destination(tmp_path / "unchunked", repeat_work_tours=4)
    chunked = run_trip_destination(
        tmp_path / "chunked",
        explicit_chunk=0.5,
        repeat_work_tours=4,
    )

    pd.testing.assert_frame_equal(chunked, unchunked)


@pytest.mark.parametrize("mode", chunk.TRAINING_MODES)
@pytest.mark.parametrize("explicit_chunk", [0, 0.5, 2])
@pytest.mark.parametrize("legacy_reason", [None, "estimator", "sample_table"])
def test_destination_pipeline_chunk_boundary(
    tmp_path, monkeypatch, mode, explicit_chunk, legacy_reason
):
    state = wf.State.make_default(
        working_dir=tmp_path,
        configs_dir=Path(__file__).parent / "configs",
        data_dir=Path(__file__).parent / "data",
    )
    state.settings.chunk_training_mode = mode
    state.settings.chunk_size = 1_000_000
    settings = td.TripDestinationSettings.model_construct(explicit_chunk=explicit_chunk)
    trips = pd.DataFrame({"value": range(4)}, index=pd.Index(range(4), name="trip_id"))
    calls = []

    def choose(
        _state, purpose, chooser, alternatives, tours, model_settings, *args, **kwargs
    ):
        calls.append((chooser.index.tolist(), model_settings))
        return chooser.copy(), None

    monkeypatch.setattr(td, "_choose_trip_destination_unchunked", choose)
    monkeypatch.setattr(td.mem, "release_memory", lambda: False)
    use_outer = mode == chunk.MODE_EXPLICIT and explicit_chunk and legacy_reason is None
    if not use_outer:
        outer = Mock(
            side_effect=AssertionError("legacy path must not create an outer chunker")
        )
        monkeypatch.setattr(td.chunk, "adaptive_chunked_choosers", outer)
    result, sample = td.choose_trip_destination(
        state,
        "work",
        trips,
        None,
        None,
        settings,
        False,
        legacy_reason == "sample_table",
        None,
        None,
        object() if legacy_reason == "estimator" else None,
        1_000_000,
        "test",
    )
    pd.testing.assert_frame_equal(result, trips)
    assert sample is None
    assert settings.explicit_chunk == explicit_chunk
    if use_outer:
        assert [rows for rows, _ in calls] == [[0, 1], [2, 3]]
        assert all(
            inner.explicit_chunk == 0 and inner is not settings for _, inner in calls
        )
    else:
        assert calls == [([0, 1, 2, 3], settings)]


def test_trip_destination_training_ignores_explicit_chunk(tmp_path):
    unchunked = run_trip_destination(
        tmp_path / "default",
        repeat_work_tours=4,
        chunk_training_mode=chunk.MODE_RETRAIN,
    )
    explicit = run_trip_destination(
        tmp_path / "explicit",
        explicit_chunk=0.5,
        repeat_work_tours=4,
        chunk_training_mode=chunk.MODE_RETRAIN,
    )
    pd.testing.assert_frame_equal(explicit, unchunked)
