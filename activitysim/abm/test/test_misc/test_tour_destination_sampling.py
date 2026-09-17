from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd

from activitysim.abm.models.util import tour_destination
from activitysim.core import workflow


def test_destination_logsums_join_persons_inside_chunks(monkeypatch):
    persons = pd.DataFrame(
        {"income": [10, 20]}, index=pd.Index([1, 2], name="person_id")
    )
    sample = pd.DataFrame(
        {
            "person_id": [1, 1, 2, 2, 2],
            "alt_dest": [101, 102, 201, 202, 203],
        },
        index=pd.Index([11, 11, 22, 22, 22], name="tour_id"),
    )
    model_settings = SimpleNamespace(
        LOGSUM_SETTINGS="tour_mode_choice.yaml",
        CHOOSER_ID_COLUMN="person_id",
        explicit_chunk=0.5,
    )
    state = SimpleNamespace(
        filesystem=SimpleNamespace(
            read_model_settings=lambda *args, **kwargs: SimpleNamespace()
        ),
        tracing=SimpleNamespace(dump_df=lambda *args, **kwargs: None),
    )
    chunk_sizer = Mock()

    def chunked(*args, **kwargs):
        assert args[1].index.tolist() == [11, 22]
        assert args[2] is sample
        assert kwargs["chunk_size"] == 123
        assert kwargs["explicit_chunk_size"] == 0.5
        yield 1, args[1].iloc[:1], sample.iloc[:2], "logsums.i1", chunk_sizer
        yield 2, args[1].iloc[1:], sample.iloc[2:], "logsums.i2", chunk_sizer

    monkeypatch.setattr(
        tour_destination.chunk, "adaptive_chunked_choosers_and_alts", chunked
    )
    chooser_lengths = []

    def compute_logsums(_state, choosers, *args, **kwargs):
        chooser_lengths.append(len(choosers))
        assert args[4] == 0
        assert kwargs["explicit_chunk_size"] == 0
        return choosers["alt_dest"] + choosers["income"]

    monkeypatch.setattr(
        tour_destination.logsum,
        "compute_location_choice_logsums",
        compute_logsums,
    )

    result = tour_destination.run_destination_logsums(
        state,
        "shopping",
        persons,
        sample,
        model_settings,
        Mock(),
        chunk_size=123,
        trace_label="non_mandatory.shopping.logsums",
    )

    assert chooser_lengths == [2, 3]
    assert result["mode_choice_logsum"].tolist() == [111, 112, 221, 222, 223]


def test_destination_simulate_forwards_explicit_chunk_size(monkeypatch):
    tours = pd.DataFrame(
        {"person_id": [1, 2], "home_zone_id": [10, 20]},
        index=pd.Index([11, 22], name="tour_id"),
    )
    persons = pd.DataFrame(
        {"income": [100, 200]}, index=pd.Index([1, 2], name="person_id")
    )
    sample = pd.DataFrame(
        {"alt_dest": [101, 102, 201]},
        index=pd.Index([11, 11, 22], name="tour_id"),
    )
    destination_size_terms = pd.DataFrame(
        {"size_term": [1.0, 2.0, 3.0]},
        index=pd.Index([101, 102, 201], name="alt_dest"),
    )
    model_settings = SimpleNamespace(
        SPEC="destination.csv",
        COEFFICIENTS="coefficients.csv",
        CHOOSER_ID_COLUMN="person_id",
        ALT_DEST_COL_NAME="alt_dest",
        CHOOSER_ORIG_COL_NAME="home_zone_id",
        CONSTANTS=None,
        explicit_chunk=0.25,
        compute_settings=SimpleNamespace(),
    )
    state = SimpleNamespace(
        settings=SimpleNamespace(log_alt_losers=False, use_explicit_error_terms=False),
        tracing=SimpleNamespace(dump_df=lambda *args, **kwargs: None),
    )
    network_los = SimpleNamespace(get_default_skim_dict=lambda: _DummySkimDict())
    monkeypatch.setattr(
        tour_destination.simulate,
        "spec_for_segment",
        lambda *args, **kwargs: pd.DataFrame({"coefficient": [1.0]}),
    )
    monkeypatch.setattr(
        tour_destination.expressions,
        "annotate_preprocessors",
        lambda *args, **kwargs: None,
    )
    captured = {}

    def simulate_sampled(*args, **kwargs):
        captured.update(kwargs)
        return pd.DataFrame(
            {"choice": [101, 201], "logsum": [1.0, 2.0]}, index=tours.index
        )

    monkeypatch.setattr(
        tour_destination, "interaction_sample_simulate", simulate_sampled
    )

    result = tour_destination.run_destination_simulate(
        state,
        "shopping",
        tours,
        persons,
        sample,
        want_logsums=True,
        model_settings=model_settings,
        network_los=network_los,
        destination_size_terms=destination_size_terms,
        estimator=None,
        chunk_size=0,
        trace_label="non_mandatory.shopping.simulate",
    )

    assert captured["explicit_chunk_size"] == 0.25
    assert result["choice"].tolist() == [101, 201]


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
        choosers,
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
        captured["origin_taz"] = choosers[tour_destination.ORIG_TAZ].copy()
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
        {
            "origin": [101],
            # A merged person table may carry a home TAZ that does not match
            # the configured tour origin.
            tour_destination.ORIG_TAZ: [99],
        },
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
    pd.testing.assert_series_equal(
        captured["origin_taz"],
        pd.Series([1], index=choosers.index, name=tour_destination.ORIG_TAZ),
    )
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
