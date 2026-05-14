from __future__ import annotations

from types import SimpleNamespace

import numpy as np
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


class _DummyRng:
    def __init__(self, draws):
        self._draws = np.asarray(draws)
        self.calls = []

    def random_for_df(self, df, n):
        self.calls.append(n)
        assert self._draws.shape == (len(df), n)
        return self._draws.copy()


class _DummyState:
    def __init__(self, draws, use_explicit_error_terms=False):
        self.settings = SimpleNamespace(
            trace_hh_id=None,
            use_explicit_error_terms=use_explicit_error_terms,
        )
        self._rng = _DummyRng(draws)

    def get_rn_generator(self):
        return self._rng


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
        _state,
        _taz_sample,
        _maz_size_terms,
        _trace_label,
        _model_settings,
        full_taz_index=None,
    ):
        captured["full_taz_index"] = full_taz_index
        return pd.DataFrame(
            {tour_destination.DEST_MAZ: [101]},
            index=pd.Index([7001], name="tour_id"),
        )

    monkeypatch.setattr(
        tour_destination, "_destination_sample", fake_destination_sample
    )
    monkeypatch.setattr(tour_destination, "choose_MAZ_for_TAZ", fake_choose_maz_for_taz)

    state = workflow.State().default_settings()
    state.settings.use_explicit_error_terms = True
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
    pd.testing.assert_index_equal(
        captured["full_taz_index"],
        pd.Index([1, 2, 3], name=tour_destination.DEST_TAZ),
    )


def test_choose_maz_for_taz_supports_variable_taz_counts():
    state = _DummyState([[0.2, 0.81], [0.1, 0.9]])

    taz_sample = pd.DataFrame(
        {
            tour_destination.DEST_TAZ: [1, 2, 2],
            "prob": [0.4, 0.6, 1.0],
            "pick_count": [1, 1, 1],
        },
        index=pd.Index([7001, 7001, 7002], name="tour_id"),
    )
    maz_size_terms = pd.DataFrame(
        {
            "zone_id": [101, 102, 201, 202],
            tour_destination.DEST_TAZ: [1, 1, 2, 2],
            "size_term": [1.0, 3.0, 4.0, 1.0],
        }
    )

    out = tour_destination.choose_MAZ_for_TAZ(
        state,
        taz_sample,
        maz_size_terms,
        "test_trace",
        SimpleNamespace(ESTIMATION_SAMPLE_SIZE=0, SAMPLE_SIZE=0),
    )

    pd.testing.assert_frame_equal(
        out,
        pd.DataFrame(
            {
                tour_destination.DEST_MAZ: [101, 202, 201],
                "prob": [0.10, 0.12, 0.80],
                "pick_count": [1, 1, 1],
            },
            index=pd.Index([7001, 7001, 7002], name="tour_id"),
        ),
    )


def test_choose_maz_for_taz_preserves_fixed_width_path():
    state = _DummyState([[0.2, 0.81], [0.1, 0.9]])

    taz_sample = pd.DataFrame(
        {
            tour_destination.DEST_TAZ: [1, 2, 1, 2],
            "prob": [0.4, 0.6, 0.25, 0.75],
            "pick_count": [1, 1, 1, 1],
        },
        index=pd.Index([7001, 7001, 7002, 7002], name="tour_id"),
    )
    maz_size_terms = pd.DataFrame(
        {
            "zone_id": [101, 102, 201, 202],
            tour_destination.DEST_TAZ: [1, 1, 2, 2],
            "size_term": [1.0, 3.0, 4.0, 1.0],
        }
    )

    out = tour_destination.choose_MAZ_for_TAZ(
        state,
        taz_sample,
        maz_size_terms,
        "test_trace",
        SimpleNamespace(ESTIMATION_SAMPLE_SIZE=0, SAMPLE_SIZE=0),
    )

    pd.testing.assert_frame_equal(
        out,
        pd.DataFrame(
            {
                tour_destination.DEST_MAZ: [101, 202, 101, 202],
                "prob": [0.10, 0.12, 0.0625, 0.15],
                "pick_count": [1, 1, 1, 1],
            },
            index=pd.Index([7001, 7001, 7002, 7002], name="tour_id"),
        ),
    )


def test_choose_maz_for_taz_eet_poisson_uses_full_taz_positions():
    state = _DummyState([[0.99, 0.2, 0.99, 0.99, 0.8]])

    taz_sample = pd.DataFrame(
        {
            tour_destination.DEST_TAZ: [2, 5],
            "prob": [0.5, 0.25],
            "pick_count": [1, 1],
        },
        index=pd.Index([7001, 7001], name="tour_id"),
    )
    maz_size_terms = pd.DataFrame(
        {
            "zone_id": [201, 202, 501, 502],
            tour_destination.DEST_TAZ: [2, 2, 5, 5],
            "size_term": [3.0, 1.0, 3.0, 1.0],
        }
    )

    out = tour_destination.choose_MAZ_for_TAZ(
        state,
        taz_sample,
        maz_size_terms,
        "test_trace",
        SimpleNamespace(ESTIMATION_SAMPLE_SIZE=0, SAMPLE_SIZE=0),
        full_taz_index=pd.Index([1, 2, 3, 4, 5], name=tour_destination.DEST_TAZ),
    )

    pd.testing.assert_frame_equal(
        out,
        pd.DataFrame(
            {
                tour_destination.DEST_MAZ: [201, 502],
                "prob": [0.375, 0.0625],
                "pick_count": [1, 1],
            },
            index=pd.Index([7001, 7001], name="tour_id"),
        ),
    )
    assert state.get_rn_generator().calls == [5]


def test_choose_maz_for_taz_uses_sample_width_when_full_taz_index_omitted():
    state = _DummyState([[0.2, 0.81]])

    taz_sample = pd.DataFrame(
        {
            tour_destination.DEST_TAZ: [2, 5],
            "prob": [0.5, 0.25],
            "pick_count": [1, 1],
        },
        index=pd.Index([7001, 7001], name="tour_id"),
    )
    maz_size_terms = pd.DataFrame(
        {
            "zone_id": [201, 202, 501, 502],
            tour_destination.DEST_TAZ: [2, 2, 5, 5],
            "size_term": [3.0, 1.0, 3.0, 1.0],
        }
    )

    out = tour_destination.choose_MAZ_for_TAZ(
        state,
        taz_sample,
        maz_size_terms,
        "test_trace",
        SimpleNamespace(ESTIMATION_SAMPLE_SIZE=0, SAMPLE_SIZE=0),
    )

    pd.testing.assert_frame_equal(
        out,
        pd.DataFrame(
            {
                tour_destination.DEST_MAZ: [201, 502],
                "prob": [0.375, 0.0625],
                "pick_count": [1, 1],
            },
            index=pd.Index([7001, 7001], name="tour_id"),
        ),
    )
    assert state.get_rn_generator().calls == [2]


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
    state.settings.use_explicit_error_terms = True
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
