from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from activitysim.abm.models import trip_destination
from activitysim.core import workflow
from activitysim.core.skim_dictionary import DataFrameMatrix


class _DummySkimHotel:
    def sample_skims(self, presample):
        return {"presample": presample}


class _DummyNetworkLos:
    zone_system = 2

    def __init__(self, maz_to_taz):
        self._maz_to_taz = maz_to_taz

    def map_maz_to_taz(self, maz_index):
        return pd.Index([self._maz_to_taz[maz] for maz in maz_index], name="zone_id")

    def get_maz_to_taz_series(self, _state):
        return pd.Series(self._maz_to_taz)


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


def test_destination_sample_retains_full_maz_universe(monkeypatch):
    captured = {}

    def fake_destination_sample(
        _state,
        _primary_purpose,
        _trips,
        alternatives,
        _model_settings,
        _size_term_matrix,
        skims,
        alt_dest_col_name,
        _estimator,
        chunk_tag,
        trace_label,
        zone_layer=None,
    ):
        captured["alternatives_index"] = alternatives.index.copy()
        captured["alt_dest_col_name"] = alt_dest_col_name
        captured["chunk_tag"] = chunk_tag
        captured["trace_label"] = trace_label
        captured["zone_layer"] = zone_layer
        captured["presample"] = skims["presample"]
        return pd.DataFrame(
            {"dest_taz": [101]},
            index=pd.Index([7001], name="trip_id"),
        )

    monkeypatch.setattr(
        trip_destination, "_destination_sample", fake_destination_sample
    )

    state = workflow.State().default_settings()
    trips = pd.DataFrame(index=pd.Index([7001], name="trip_id"))
    model_settings = type("ModelSettings", (), {"ALT_DEST_COL_NAME": "dest_taz"})()

    alternatives = pd.DataFrame(
        {"eatout": [1.0, 0.0, 2.0]},
        index=pd.Index([101, 102, 103], name="dest_taz"),
    )
    size_term_matrix = DataFrameMatrix(alternatives)

    out = trip_destination.destination_sample(
        state,
        "eatout",
        trips,
        alternatives,
        model_settings,
        size_term_matrix,
        _DummySkimHotel(),
        estimator=None,
        chunk_size=0,
        trace_label="test_trace",
    )

    pd.testing.assert_frame_equal(
        out,
        pd.DataFrame({"dest_taz": [101]}, index=pd.Index([7001], name="trip_id")),
    )
    pd.testing.assert_index_equal(
        captured["alternatives_index"],
        pd.Index([101, 102, 103], name="dest_taz"),
    )
    assert captured["alt_dest_col_name"] == "dest_taz"
    assert captured["chunk_tag"] == "trip_destination.sample.eatout"
    assert captured["zone_layer"] is None
    assert captured["presample"] is False


def test_destination_presample_retains_full_taz_universe(monkeypatch):
    captured = {}

    def fake_destination_sample(
        _state,
        _primary_purpose,
        _trips,
        alternatives,
        _model_settings,
        size_term_matrix,
        skims,
        alt_dest_col_name,
        _estimator,
        chunk_tag,
        trace_label,
        zone_layer=None,
    ):
        captured["alternatives_index"] = alternatives.index.copy()
        captured["size_term_index"] = size_term_matrix.df.index.copy()
        captured["alt_dest_col_name"] = alt_dest_col_name
        captured["chunk_tag"] = chunk_tag
        captured["trace_label"] = trace_label
        captured["zone_layer"] = zone_layer
        captured["presample"] = skims["presample"]
        return pd.DataFrame(
            {"dest_taz": [1]},
            index=pd.Index([7001], name="trip_id"),
        )

    def fake_choose_maz_for_taz(
        _state,
        _taz_sample,
        _maz_size_terms,
        _trips,
        _network_los,
        _alt_dest_col_name,
        _trace_label,
        _model_settings,
        full_taz_index=None,
    ):
        captured["full_taz_index"] = full_taz_index
        return pd.DataFrame(
            {"dest_taz": [101]},
            index=pd.Index([7001], name="trip_id"),
        )

    monkeypatch.setattr(
        trip_destination, "_destination_sample", fake_destination_sample
    )
    monkeypatch.setattr(trip_destination, "choose_MAZ_for_TAZ", fake_choose_maz_for_taz)

    state = workflow.State().default_settings()
    trips = pd.DataFrame(
        {"origin": [101], "tour_leg_dest": [103]},
        index=pd.Index([7001], name="trip_id"),
    )
    model_settings = type(
        "ModelSettings",
        (),
        {
            "ALT_DEST_COL_NAME": "dest_taz",
            "TRIP_ORIGIN": "origin",
            "PRIMARY_DEST": "tour_leg_dest",
        },
    )()
    network_los = _DummyNetworkLos({101: 1, 102: 2, 103: 3})

    alternatives = pd.DataFrame(
        {"eatout": [1.0, 0.0, 2.0]},
        index=pd.Index([101, 102, 103], name="dest_taz"),
    )
    size_term_matrix = DataFrameMatrix(alternatives)

    out = trip_destination.destination_presample(
        state,
        "eatout",
        trips,
        alternatives,
        model_settings,
        size_term_matrix,
        _DummySkimHotel(),
        network_los,
        estimator=None,
        trace_label="test_trace",
    )

    pd.testing.assert_frame_equal(
        out,
        pd.DataFrame({"dest_taz": [101]}, index=pd.Index([7001], name="trip_id")),
    )
    pd.testing.assert_index_equal(
        captured["alternatives_index"],
        pd.Index([1, 2, 3], name="zone_id"),
    )
    pd.testing.assert_index_equal(
        captured["size_term_index"],
        pd.Index([1, 2, 3], name="zone_id"),
    )
    assert captured["alt_dest_col_name"] == "dest_taz"
    assert captured["chunk_tag"] == "trip_destination.presample.eatout"
    assert captured["zone_layer"] == "taz"
    assert captured["presample"] is True
    assert captured["full_taz_index"] is None


def test_destination_presample_passes_full_taz_index_for_eet_poisson(monkeypatch):
    captured = {}

    def fake_destination_sample(
        _state,
        _primary_purpose,
        _trips,
        alternatives,
        _model_settings,
        size_term_matrix,
        skims,
        alt_dest_col_name,
        _estimator,
        chunk_tag,
        trace_label,
        zone_layer=None,
    ):
        captured["alternatives_index"] = alternatives.index.copy()
        captured["size_term_index"] = size_term_matrix.df.index.copy()
        captured["alt_dest_col_name"] = alt_dest_col_name
        captured["chunk_tag"] = chunk_tag
        captured["trace_label"] = trace_label
        captured["zone_layer"] = zone_layer
        captured["presample"] = skims["presample"]
        return pd.DataFrame(
            {"dest_taz": [1]},
            index=pd.Index([7001], name="trip_id"),
        )

    def fake_choose_maz_for_taz(
        _state,
        _taz_sample,
        _maz_size_terms,
        _trips,
        _network_los,
        _alt_dest_col_name,
        _trace_label,
        _model_settings,
        full_taz_index=None,
    ):
        captured["full_taz_index"] = full_taz_index
        return pd.DataFrame(
            {"dest_taz": [101]},
            index=pd.Index([7001], name="trip_id"),
        )

    monkeypatch.setattr(
        trip_destination, "_destination_sample", fake_destination_sample
    )
    monkeypatch.setattr(trip_destination, "choose_MAZ_for_TAZ", fake_choose_maz_for_taz)

    state = workflow.State().default_settings()
    state.settings.use_explicit_error_terms = True
    state.add_table(
        "land_use_taz",
        pd.DataFrame(index=pd.Index([1, 2, 3], name="zone_id")),
    )
    trips = pd.DataFrame(
        {"origin": [101], "tour_leg_dest": [103]},
        index=pd.Index([7001], name="trip_id"),
    )
    model_settings = type(
        "ModelSettings",
        (),
        {
            "ALT_DEST_COL_NAME": "dest_taz",
            "TRIP_ORIGIN": "origin",
            "PRIMARY_DEST": "tour_leg_dest",
        },
    )()
    network_los = _DummyNetworkLos({101: 1, 102: 2, 103: 3})

    alternatives = pd.DataFrame(
        {"eatout": [1.0, 0.0, 2.0]},
        index=pd.Index([101, 102, 103], name="dest_taz"),
    )
    size_term_matrix = DataFrameMatrix(alternatives)

    out = trip_destination.destination_presample(
        state,
        "eatout",
        trips,
        alternatives,
        model_settings,
        size_term_matrix,
        _DummySkimHotel(),
        network_los,
        estimator=None,
        trace_label="test_trace",
    )

    pd.testing.assert_frame_equal(
        out,
        pd.DataFrame({"dest_taz": [101]}, index=pd.Index([7001], name="trip_id")),
    )
    pd.testing.assert_index_equal(
        captured["full_taz_index"],
        pd.Index([1, 2, 3], name="dest_taz_TAZ"),
    )


def test_choose_maz_for_taz_eet_poisson_uses_full_taz_positions():
    state = _DummyState([[0.99, 0.2, 0.99, 0.99, 0.8]])
    network_los = _DummyNetworkLos({201: 2, 202: 2, 501: 5, 502: 5})

    taz_sample = pd.DataFrame(
        {
            "dest_taz": [2, 5],
            "prob": [0.5, 0.25],
            "pick_count": [1, 1],
        },
        index=pd.Index([7001, 7001], name="trip_id"),
    )
    maz_size_terms = DataFrameMatrix(
        pd.DataFrame(
            {"eatout": [3.0, 1.0, 3.0, 1.0]},
            index=pd.Index([201, 202, 501, 502], name="dest_taz"),
        )
    )
    trips = pd.DataFrame(
        {"purpose": ["eatout"]},
        index=pd.Index([7001], name="trip_id"),
    )

    out = trip_destination.choose_MAZ_for_TAZ(
        state,
        taz_sample,
        maz_size_terms,
        trips,
        network_los,
        "dest_taz",
        "test_trace",
        SimpleNamespace(ESTIMATION_SAMPLE_SIZE=0, SAMPLE_SIZE=0),
        full_taz_index=pd.Index([1, 2, 3, 4, 5], name="dest_taz_TAZ"),
    )

    pd.testing.assert_frame_equal(
        out,
        pd.DataFrame(
            {
                "dest_taz": [201, 502],
                "prob": [0.375, 0.0625],
                "pick_count": [1, 1],
            },
            index=pd.Index([7001, 7001], name="trip_id"),
        ),
    )
    assert state.get_rn_generator().calls == [5]


def test_choose_maz_for_taz_uses_sample_width_when_full_taz_index_omitted():
    state = _DummyState([[0.2, 0.81]])
    network_los = _DummyNetworkLos({201: 2, 202: 2, 501: 5, 502: 5})

    taz_sample = pd.DataFrame(
        {
            "dest_taz": [2, 5],
            "prob": [0.5, 0.25],
            "pick_count": [1, 1],
        },
        index=pd.Index([7001, 7001], name="trip_id"),
    )
    maz_size_terms = DataFrameMatrix(
        pd.DataFrame(
            {"eatout": [3.0, 1.0, 3.0, 1.0]},
            index=pd.Index([201, 202, 501, 502], name="dest_taz"),
        )
    )
    trips = pd.DataFrame(
        {"purpose": ["eatout"]},
        index=pd.Index([7001], name="trip_id"),
    )

    out = trip_destination.choose_MAZ_for_TAZ(
        state,
        taz_sample,
        maz_size_terms,
        trips,
        network_los,
        "dest_taz",
        "test_trace",
        SimpleNamespace(ESTIMATION_SAMPLE_SIZE=0, SAMPLE_SIZE=0),
    )

    pd.testing.assert_frame_equal(
        out,
        pd.DataFrame(
            {
                "dest_taz": [201, 502],
                "prob": [0.375, 0.0625],
                "pick_count": [1, 1],
            },
            index=pd.Index([7001, 7001], name="trip_id"),
        ),
    )
    assert state.get_rn_generator().calls == [2]
