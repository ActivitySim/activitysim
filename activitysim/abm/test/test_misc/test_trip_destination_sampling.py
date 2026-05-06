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

    monkeypatch.setattr(trip_destination, "_destination_sample", fake_destination_sample)

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
    assert captured["chunk_tag"] == "trip_destination.sample"
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
    ):
        return pd.DataFrame(
            {"dest_taz": [101]},
            index=pd.Index([7001], name="trip_id"),
        )

    monkeypatch.setattr(trip_destination, "_destination_sample", fake_destination_sample)
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
    assert captured["chunk_tag"] == "trip_destination.presample"
    assert captured["zone_layer"] == "taz"
    assert captured["presample"] is True