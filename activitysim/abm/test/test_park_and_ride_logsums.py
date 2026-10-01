from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from activitysim.abm.models import location_choice, park_and_ride_lot_choice as pnr
from activitysim.abm.models.util import logsums, tour_destination
from activitysim.core import interaction_simulate, workflow


@pytest.fixture
def pnr_context(monkeypatch):
    state = workflow.State.make_temp()
    state.settings.chunk_size = 1_000_000_000
    state.settings.sharrow = False
    state.rng().begin_step("destination")
    land_use = pd.DataFrame({"pnr_spaces": [10, 10]}, index=[1, 2])
    state.add_table("land_use", land_use)
    settings = SimpleNamespace(
        SPEC="unused.csv",
        CONSTANTS={},
        LANDUSE_PNR_SPACES_COLUMN="pnr_spaces",
        preprocessor=None,
        CHOOSER_FILTER_EXPR=None,
        explicit_chunk=0.5,
        compute_settings=None,
    )
    spec = pd.DataFrame({"coefficient": [1.0]}, index=["1"])
    fs = type(state.filesystem)
    monkeypatch.setattr(fs, "read_model_spec", lambda *a, **k: spec)
    monkeypatch.setattr(fs, "read_model_coefficients", lambda *a, **k: {})
    monkeypatch.setattr(fs, "get_segment_coefficients", lambda *a, **k: {})
    monkeypatch.setattr(
        pnr.ParkAndRideLotChoiceSettings, "read_settings_file", lambda *a, **k: settings
    )
    monkeypatch.setattr(
        pnr.simulate, "eval_coefficients", lambda state, spec, *a, **k: spec
    )
    monkeypatch.setattr(
        pnr,
        "filter_chooser_to_transit_accessible_destinations",
        lambda state, choosers, *a: choosers,
    )
    monkeypatch.setattr(logsums, "setup_skims", lambda *a, **k: {})
    monkeypatch.setattr(pnr.expressions, "annotate_preprocessors", lambda *a, **k: None)
    monkeypatch.setattr(pnr.util, "drop_unused_columns", lambda df, *a, **k: df)
    network = SimpleNamespace(
        skim_time_period_label=lambda period, as_cat, broadcast_to: pd.Series(
            "AM", index=broadcast_to
        )
    )
    return state, land_use, network, settings


@pytest.mark.parametrize("mode", ["training", "adaptive", "explicit"])
@pytest.mark.parametrize("component", ["tour", "location"])
def test_pnr_logsums_share_outer_chunk_boundary(
    pnr_context, monkeypatch, mode, component
):
    state, land_use, network, pnr_settings = pnr_context
    state.settings.chunk_training_mode = mode
    persons = pd.DataFrame(
        {"home_zone_id": [1, 2]}, index=pd.Index([1, 2], name="person_id")
    )
    sample = pd.DataFrame(
        {"person_id": [1, 1, 2, 2], "alt_dest": [1, 2, 1, 2]},
        index=pd.Index(
            [1, 1, 2, 2], name="person_id" if component == "location" else "tour_id"
        ),
    )
    if component == "location":
        sample = sample.drop(columns="person_id")
    model = SimpleNamespace(
        LOGSUM_SETTINGS="unused.yaml",
        LOGSUM_TOUR_PURPOSE="work",
        CHOOSER_ID_COLUMN="person_id",
        explicit_chunk=0.5,
        CHOOSER_ORIG_COL_NAME="home_zone_id",
        ALT_DEST_COL_NAME="alt_dest",
        IN_PERIOD=17,
        OUT_PERIOD=8,
        LOGSUM_PREPROCESSOR="preprocessor",
    )
    logsum_settings = SimpleNamespace(
        include_pnr_for_logsums=True,
        SPEC="unused.csv",
        preprocessor=None,
        compute_settings=None,
    )
    monkeypatch.setattr(
        type(state.filesystem), "read_model_settings", lambda *a, **k: logsum_settings
    )
    monkeypatch.setattr(
        location_choice.TourModeComponentSettings,
        "read_settings_file",
        lambda *a, **k: logsum_settings,
    )
    monkeypatch.setattr(logsums.config, "get_logit_model_settings", lambda *a: None)
    monkeypatch.setattr(logsums.config, "get_model_constants", lambda *a: {})
    monkeypatch.setattr(
        logsums.simulate,
        "simple_simulate_logsums",
        lambda state, choosers, *a, **k: choosers.pnr_zone_id.astype(float),
    )
    sizes = []

    def evaluate(state, choosers, *args, **kwargs):
        sizes.append(len(choosers))
        return pd.Series(1, index=choosers.index)

    # Keep the actual interaction simulator and its ChunkSizer, replacing only
    # utility evaluation so the test exercises the full ledger call chain.
    monkeypatch.setattr(interaction_simulate, "_interaction_simulate", evaluate)
    if component == "tour":
        result = tour_destination.run_destination_logsums(
            state,
            "work",
            persons,
            sample,
            model,
            network,
            state.settings.chunk_size,
            "test",
        )
    else:
        result = location_choice.run_location_logsums(
            state,
            "work",
            persons,
            network,
            sample,
            model,
            state.settings.chunk_size,
            "logsums",
            "test",
        )
    assert result.mode_choice_logsum.tolist() == [1.0] * 4
    assert sum(sizes) == 4
    if mode == "explicit":
        assert sizes == [2, 2]  # PNR's fractional setting must not split these again.
    assert pnr_settings.explicit_chunk == 0.5
    assert not state.chunk.CHUNK_SIZERS
    assert not state.chunk.CHUNK_LEDGERS


@pytest.mark.parametrize("component", ["tour", "location"])
def test_pnr_random_draws_and_logsums_do_not_depend_on_chunk_size(
    pnr_context, monkeypatch, component
):
    state, land_use, network, pnr_settings = pnr_context
    state.settings.chunk_training_mode = "explicit"
    persons = pd.DataFrame(
        {"home_zone_id": [1, 2, 1]}, index=pd.Index([11, 22, 33], name="person_id")
    )
    # Different multiplicities cross a rounding boundary; the final chooser
    # also exercises a chunk whose index is unique despite the segment's not
    # being unique. Every segment uses the same canonical PNR random channel.
    ids = [11] * 11 + [22] * 2 + [33]
    sample = pd.DataFrame(
        {"person_id": ids, "alt_dest": list(range(11)) + [1, 2, 1]},
        index=pd.Index(ids, name="person_id" if component == "location" else "tour_id"),
    )
    if component == "location":
        sample = sample.drop(columns="person_id")
    model = SimpleNamespace(
        LOGSUM_SETTINGS="unused.yaml",
        LOGSUM_TOUR_PURPOSE="work",
        CHOOSER_ID_COLUMN="person_id",
        explicit_chunk=0,
        CHOOSER_ORIG_COL_NAME="home_zone_id",
        ALT_DEST_COL_NAME="alt_dest",
        IN_PERIOD=17,
        OUT_PERIOD=8,
        LOGSUM_PREPROCESSOR="preprocessor",
    )
    logsum_settings = SimpleNamespace(
        include_pnr_for_logsums=True,
        SPEC="unused.csv",
        preprocessor=None,
        compute_settings=None,
    )
    monkeypatch.setattr(
        type(state.filesystem), "read_model_settings", lambda *a, **k: logsum_settings
    )
    monkeypatch.setattr(
        location_choice.TourModeComponentSettings,
        "read_settings_file",
        lambda *a, **k: logsum_settings,
    )
    monkeypatch.setattr(logsums.config, "get_logit_model_settings", lambda *a: None)
    monkeypatch.setattr(logsums.config, "get_model_constants", lambda *a: {})
    monkeypatch.setattr(
        logsums.simulate,
        "simple_simulate_logsums",
        lambda state, choosers, *a, **k: choosers.pnr_zone_id.astype(float),
    )
    draws = []

    def evaluate(state, choosers, *args, **kwargs):
        rands = state.rng().random_for_df(choosers).ravel()
        draws.extend(zip(choosers.index, rands))
        return pd.Series(1 + (rands > 0.5).astype(int), index=choosers.index)

    monkeypatch.setattr(interaction_simulate, "_interaction_simulate", evaluate)
    baseline = baseline_draws = None
    for size in [0, 0.5, 1, 2]:
        model.explicit_chunk = size
        draws.clear()
        if component == "tour":
            result = tour_destination.run_destination_logsums(
                state, "work", persons, sample.copy(), model, network, 0, "test"
            )
        else:
            result = location_choice.run_location_logsums(
                state,
                "work",
                persons,
                network,
                sample.copy(),
                model,
                0,
                "logsums",
                "test",
            )
        if baseline is None:
            baseline, baseline_draws = result, list(draws)
            assert [idx for idx, rand in draws] == list(range(220, 231)) + [
                440,
                441,
                660,
            ]
        else:
            pd.testing.assert_frame_equal(result, baseline)
            assert draws == baseline_draws
        assert "pnr_lot_choice" not in state.rng().channels
