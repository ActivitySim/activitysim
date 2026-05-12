from __future__ import annotations

import pandas as pd

from activitysim.abm.models.util import tour_destination
from activitysim.core import workflow


class _DummySkimDict:
    def wrap(self, orig_key, dest_key):
        return type("WrappedSkims", (), {"orig_key": orig_key, "dest_key": dest_key})()


class _DummyNetworkLos:
    zone_system = 2

    def __init__(self, maz_to_taz):
        self._maz_to_taz = maz_to_taz

    def map_maz_to_taz(self, maz_index):
        return pd.Index([self._maz_to_taz[maz] for maz in maz_index], name="TAZ")

    def get_default_skim_dict(self):
        return _DummySkimDict()

    def get_skim_dict(self, layer):
        assert layer == "taz"
        return _DummySkimDict()


def test_destination_presample_uses_taz_stable_mapping(monkeypatch):
    captured = {}

    def fake_destination_sample(
        _state,
        _spec_segment_name,
        _choosers,
        destination_size_terms,
        _skims,
        _estimator,
        _model_settings,
        alt_dest_col_name,
        chunk_tag,
        trace_label,
        zone_layer=None,
        stable_alt_positions=None,
        n_total_alts=None,
    ):
        captured["alt_dest_col_name"] = alt_dest_col_name
        captured["zone_layer"] = zone_layer
        captured["active_taz_index"] = destination_size_terms.index.copy()
        captured["stable_alt_positions"] = stable_alt_positions.copy()
        captured["n_total_alts"] = n_total_alts
        captured["chunk_tag"] = chunk_tag
        captured["trace_label"] = trace_label
        return pd.DataFrame(
            {tour_destination.DEST_TAZ: [1]},
            index=pd.Index([7001], name="tour_id"),
        )

    def fake_choose_maz_for_taz(
        _state, _taz_sample, _maz_size_terms, _trace_label, _model_settings
    ):
        return pd.DataFrame(
            {tour_destination.DEST_MAZ: [101]},
            index=pd.Index([7001], name="tour_id"),
        )

    monkeypatch.setattr(
        tour_destination, "_destination_sample", fake_destination_sample
    )
    monkeypatch.setattr(tour_destination, "choose_MAZ_for_TAZ", fake_choose_maz_for_taz)

    state = workflow.State().default_settings()
    choosers = pd.DataFrame(
        {"origin": [101]},
        index=pd.Index([7001], name="tour_id"),
    )
    model_settings = type(
        "ModelSettings",
        (),
        {
            "ALT_DEST_COL_NAME": "zone_id",
            "CHOOSER_ORIG_COL_NAME": "origin",
        },
    )()
    network_los = _DummyNetworkLos({101: 1, 102: 2, 103: 3})

    active_destination_size_terms = pd.DataFrame(
        {"size_term": [1.0, 2.0]},
        index=pd.Index([101, 103], name="zone_id"),
    )
    full_destination_size_terms = pd.DataFrame(
        {"size_term": [1.0, 0.0, 2.0]},
        index=pd.Index([101, 102, 103], name="zone_id"),
    )

    out = tour_destination.destination_presample(
        state,
        "segment",
        choosers,
        model_settings,
        network_los,
        active_destination_size_terms,
        full_destination_size_terms,
        estimator=None,
        trace_label="test_trace",
    )

    pd.testing.assert_frame_equal(
        out,
        pd.DataFrame({"zone_id": [101]}, index=pd.Index([7001], name="tour_id")),
    )
    pd.testing.assert_index_equal(
        captured["active_taz_index"],
        pd.Index([1, 3], name=tour_destination.DEST_TAZ),
    )
    assert captured["alt_dest_col_name"] == tour_destination.DEST_TAZ
    assert captured["zone_layer"] == "taz"
    assert captured["n_total_alts"] == 3
    assert list(captured["stable_alt_positions"]) == [0, 2]


def test_destination_sample_uses_maz_stable_mapping(monkeypatch):
    captured = {}

    def fake_destination_sample(
        _state,
        _spec_segment_name,
        _choosers,
        destination_size_terms,
        _skims,
        _estimator,
        _model_settings,
        alt_dest_col_name,
        chunk_tag,
        trace_label,
        zone_layer=None,
        stable_alt_positions=None,
        n_total_alts=None,
    ):
        captured["active_maz_index"] = destination_size_terms.index.copy()
        captured["stable_alt_positions"] = stable_alt_positions.copy()
        captured["n_total_alts"] = n_total_alts
        captured["alt_dest_col_name"] = alt_dest_col_name
        captured["zone_layer"] = zone_layer
        return pd.DataFrame(
            {"zone_id": [101], "person_id": [55]},
            index=pd.Index([7001], name="tour_id"),
        )

    monkeypatch.setattr(
        tour_destination, "_destination_sample", fake_destination_sample
    )

    state = workflow.State().default_settings()
    choosers = pd.DataFrame(
        {"origin": [101], "person_id": [55]},
        index=pd.Index([7001], name="tour_id"),
    )
    model_settings = type(
        "ModelSettings",
        (),
        {
            "ALT_DEST_COL_NAME": "zone_id",
            "CHOOSER_ORIG_COL_NAME": "origin",
            "CHOOSER_ID_COLUMN": "person_id",
        },
    )()
    network_los = _DummyNetworkLos({101: 1, 102: 2, 103: 3})

    active_destination_size_terms = pd.DataFrame(
        {"size_term": [1.0, 2.0]},
        index=pd.Index([101, 103], name="zone_id"),
    )
    full_destination_size_terms = pd.DataFrame(
        {"size_term": [1.0, 0.0, 2.0]},
        index=pd.Index([101, 102, 103], name="zone_id"),
    )

    out = tour_destination.destination_sample(
        state,
        "segment",
        choosers,
        model_settings,
        network_los,
        active_destination_size_terms,
        full_destination_size_terms,
        estimator=None,
        chunk_size=0,
        trace_label="test_trace",
    )

    pd.testing.assert_frame_equal(
        out,
        pd.DataFrame(
            {"zone_id": [101], "person_id": [55]},
            index=pd.Index([7001], name="tour_id"),
        ),
    )
    pd.testing.assert_index_equal(
        captured["active_maz_index"],
        pd.Index([101, 103], name="zone_id"),
    )
    assert list(captured["stable_alt_positions"]) == [0, 2]
    assert captured["n_total_alts"] == 3
    assert captured["alt_dest_col_name"] == "zone_id"
    assert captured["zone_layer"] is None
