from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from activitysim.abm.models.util import tour_od
from activitysim.core import logit, workflow


class _DummySkimDict:
    def wrap(self, orig_key, dest_key):
        return type("WrappedSkims", (), {"orig_key": orig_key, "dest_key": dest_key})()


class _DummyNetworkLos:
    zone_system = 2

    def __init__(self, maz_to_taz):
        self._maz_to_taz = maz_to_taz

    def map_maz_to_taz(self, maz_index):
        return pd.Index([self._maz_to_taz[maz] for maz in maz_index], name="TAZ")

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
    @staticmethod
    def make(draws, use_explicit_error_terms=False):
        state = workflow.State().default_settings()
        state.settings.trace_hh_id = None
        state.settings.use_explicit_error_terms = use_explicit_error_terms
        rng = _DummyRng(draws)
        state._dummy_rng = rng
        state.get_rn_generator = lambda: rng
        return state


def test_od_presample_passes_full_taz_index_for_eet_poisson(monkeypatch):
    captured = {}

    def fake_od_sample(
        _state,
        _spec_segment_name,
        _choosers,
        _network_los,
        destination_size_terms,
        _origin_id_col,
        _dest_id_col,
        _skims,
        _estimator,
        _model_settings,
        alt_od_col_name,
        _chunk_size,
        chunk_tag,
        trace_label,
    ):
        captured["active_taz_index"] = destination_size_terms.index.copy()
        captured["alt_od_col_name"] = alt_od_col_name
        captured["chunk_tag"] = chunk_tag
        captured["trace_label"] = trace_label
        return pd.DataFrame(
            {
                alt_od_col_name: ["101_1", "101_3"],
                "prob": [0.5, 0.25],
                "pick_count": [1, 1],
            },
            index=pd.Index([7001, 7001], name="tour_id"),
        )

    def fake_choose_maz_for_taz(
        _state,
        _taz_sample,
        _maz_size_terms,
        _trace_label,
        addtl_col_for_unique_key=None,
        dest_maz_id_col=tour_od.DEST_MAZ,
        full_taz_index=None,
    ):
        captured["addtl_col_for_unique_key"] = addtl_col_for_unique_key
        captured["dest_maz_id_col"] = dest_maz_id_col
        captured["full_taz_index"] = full_taz_index
        return pd.DataFrame(
            {
                dest_maz_id_col: [101],
                tour_od.ORIG_MAZ: [101],
                "prob": [0.5],
                "pick_count": [1],
            },
            index=pd.Index([7001], name="tour_id"),
        )

    monkeypatch.setattr(tour_od, "_od_sample", fake_od_sample)
    monkeypatch.setattr(tour_od, "choose_MAZ_for_TAZ", fake_choose_maz_for_taz)

    state = workflow.State().default_settings()
    state.settings.use_explicit_error_terms = True
    choosers = pd.DataFrame(
        {tour_od.ORIG_TAZ: [1]},
        index=pd.Index([7001], name="tour_id"),
    )
    model_settings = type(
        "ModelSettings",
        (),
        {
            "ALT_DEST_COL_NAME": "alt_dest",
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

    out = tour_od.od_presample(
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
            {"alt_dest": [101], "origin": [101], "prob": [0.5], "pick_count": [1]},
            index=pd.Index([7001], name="tour_id"),
        ),
    )
    pd.testing.assert_index_equal(
        captured["active_taz_index"],
        pd.Index([1, 3], name=tour_od.DEST_TAZ),
    )
    assert captured["alt_od_col_name"] == tour_od.get_od_id_col(
        tour_od.ORIG_MAZ, tour_od.DEST_TAZ
    )
    assert captured["chunk_tag"] == "tour_od.presample"
    assert captured["addtl_col_for_unique_key"] == tour_od.ORIG_MAZ
    pd.testing.assert_index_equal(
        captured["full_taz_index"],
        pd.Index([1, 2, 3], name=tour_od.DEST_TAZ),
    )


@pytest.mark.parametrize("use_explicit_error_terms", [False, True], ids=["mnl", "eet"])
def test_od_presample_returns_empty_sample_when_every_chooser_failed(
    monkeypatch, use_explicit_error_terms
):
    """An empty OD sample gives an empty result with the usual columns.

    Under skip_failed_choices, the OD sample is empty when every chooser in the
    segment failed sampling. choose_MAZ_for_TAZ raises on an empty sample.
    """
    alt_od_col_name = tour_od.get_od_id_col(tour_od.ORIG_MAZ, tour_od.DEST_TAZ)

    def fake_od_sample(*_args, **_kwargs):
        # interaction_sample's result when every chooser failed
        return pd.DataFrame(
            {
                alt_od_col_name: pd.Series(dtype="object"),
                "prob": pd.Series(dtype="float32"),
                "pick_count": pd.Series(dtype="uint32"),
            },
            index=pd.Index([], name="tour_id", dtype="int64"),
        )

    monkeypatch.setattr(tour_od, "_od_sample", fake_od_sample)

    state = workflow.State().default_settings()
    state.settings.skip_failed_choices = True
    state.settings.use_explicit_error_terms = use_explicit_error_terms
    choosers = pd.DataFrame(
        {tour_od.ORIG_TAZ: [1, 3]},
        index=pd.Index([7001, 7002], name="tour_id"),
    )
    model_settings = type(
        "ModelSettings",
        (),
        {
            "ALT_DEST_COL_NAME": "alt_dest",
            "CHOOSER_ORIG_COL_NAME": "origin",
        },
    )()
    network_los = _DummyNetworkLos({101: 1, 102: 2, 103: 3})
    destination_size_terms = pd.DataFrame(
        {"size_term": [1.0, 2.0, 3.0]},
        index=pd.Index([101, 102, 103], name="zone_id"),
    )

    out = tour_od.od_presample(
        state,
        "segment",
        choosers,
        model_settings,
        network_los,
        destination_size_terms,
        destination_size_terms,
        estimator=None,
        chunk_size=0,
        trace_label="test_trace",
    )

    assert out.empty
    assert out.index.name == "tour_id"
    assert list(out.columns) == ["alt_dest", "origin", "prob", "pick_count"]


def test_od_presample_omits_full_taz_index_for_eet_non_poisson(monkeypatch):
    captured = {}

    def fake_od_sample(
        _state,
        _spec_segment_name,
        _choosers,
        _network_los,
        destination_size_terms,
        _origin_id_col,
        _dest_id_col,
        _skims,
        _estimator,
        _model_settings,
        alt_od_col_name,
        _chunk_size,
        chunk_tag,
        trace_label,
    ):
        captured["active_taz_index"] = destination_size_terms.index.copy()
        captured["alt_od_col_name"] = alt_od_col_name
        captured["chunk_tag"] = chunk_tag
        captured["trace_label"] = trace_label
        return pd.DataFrame(
            {
                alt_od_col_name: ["101_1", "101_3"],
                "prob": [0.5, 0.25],
                "pick_count": [1, 1],
            },
            index=pd.Index([7001, 7001], name="tour_id"),
        )

    def fake_choose_maz_for_taz(
        _state,
        _taz_sample,
        _maz_size_terms,
        _trace_label,
        addtl_col_for_unique_key=None,
        dest_maz_id_col=tour_od.DEST_MAZ,
        full_taz_index=None,
    ):
        captured["addtl_col_for_unique_key"] = addtl_col_for_unique_key
        captured["dest_maz_id_col"] = dest_maz_id_col
        captured["full_taz_index"] = full_taz_index
        return pd.DataFrame(
            {
                dest_maz_id_col: [101],
                tour_od.ORIG_MAZ: [101],
                "prob": [0.5],
                "pick_count": [1],
            },
            index=pd.Index([7001], name="tour_id"),
        )

    monkeypatch.setattr(tour_od, "_od_sample", fake_od_sample)
    monkeypatch.setattr(tour_od, "choose_MAZ_for_TAZ", fake_choose_maz_for_taz)

    state = workflow.State().default_settings()
    state.settings.use_explicit_error_terms = True
    state.settings.sample_method = "eet"
    choosers = pd.DataFrame(
        {tour_od.ORIG_TAZ: [1]},
        index=pd.Index([7001], name="tour_id"),
    )
    model_settings = type(
        "ModelSettings",
        (),
        {
            "ALT_DEST_COL_NAME": "alt_dest",
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

    out = tour_od.od_presample(
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
            {"alt_dest": [101], "origin": [101], "prob": [0.5], "pick_count": [1]},
            index=pd.Index([7001], name="tour_id"),
        ),
    )
    assert captured["full_taz_index"] is None


def test_choose_maz_for_taz_eet_uses_full_taz_positions_with_origin_key():
    state = _DummyState.make([[0.99, 0.2, 0.99, 0.99, 0.8]])

    taz_sample = pd.DataFrame(
        {
            tour_od.DEST_TAZ: [2, 5],
            "prob": [0.5, 0.25],
            "pick_count": [1, 1],
            tour_od.ORIG_MAZ: [9001, 9001],
        },
        index=pd.Index([7001, 7001], name="tour_id"),
    )
    maz_size_terms = pd.DataFrame(
        {
            "zone_id": [201, 202, 501, 502],
            tour_od.DEST_TAZ: [2, 2, 5, 5],
            "size_term": [3.0, 1.0, 3.0, 1.0],
        }
    )

    out = tour_od.choose_MAZ_for_TAZ(
        state,
        taz_sample,
        maz_size_terms,
        "test_trace",
        addtl_col_for_unique_key=tour_od.ORIG_MAZ,
        full_taz_index=pd.Index([1, 2, 3, 4, 5], name=tour_od.DEST_TAZ),
    )

    pd.testing.assert_frame_equal(
        out,
        pd.DataFrame(
            {
                tour_od.DEST_MAZ: [201, 502],
                tour_od.ORIG_MAZ: [9001, 9001],
                "prob": [0.375, 0.0625],
                "pick_count": [1, 1],
            },
            index=pd.Index([7001, 7001], name="tour_id"),
        ),
    )
    assert state.get_rn_generator().calls == [5]


class _SizeTerms:
    def __init__(self, state, size_term_selector):
        self.destination_size_terms = pd.DataFrame(
            {"work": [1.0, 1.0]}, index=pd.Index([101, 202], name="zone_id")
        )

    def dest_size_terms_df(self, segment_name, trace_label):
        return self.destination_size_terms[[segment_name]].rename(
            columns={segment_name: "size_term"}
        )


@pytest.mark.parametrize(
    "sampled_tours", [[], [2, 2]], ids=["all_failed", "one_failed"]
)
def test_run_tour_od_skips_failed_samples(monkeypatch, sampled_tours):
    """Tours without an OD sample skip logsums and simulation; their households are recorded.

    With every tour failed, run_tour_od returns an empty choice table instead of
    raising UnboundLocalError.
    """
    tours = pd.DataFrame(
        {"household_id": [11, 12], "person_id": [1, 2], "tour_type": ["work", "work"]},
        index=pd.Index([1, 2], name="tour_id"),
    )
    persons = pd.DataFrame(
        {"is_university": [False, False], "demographic_segment": [0, 0]},
        index=pd.Index([1, 2], name="person_id"),
    )
    sample = pd.DataFrame(
        {"alt_od": ["101_1", "202_2"][: len(sampled_tours)]},
        index=pd.Index(sampled_tours, name="tour_id", dtype="int64"),
    )
    expected_tours = tours.loc[tours.index.isin(sampled_tours)]
    logsums_model = Mock(return_value=sample.assign(tour_mode_choice_logsum=0.0))
    simulate_model = Mock(
        side_effect=lambda state, segment, choosers, *args, **kwargs: pd.DataFrame(
            {"choice": "101_1", "origin": 1, "destination": 101}, index=choosers.index
        )
    )
    monkeypatch.setattr(tour_od, "SizeTermCalculator", _SizeTerms)
    monkeypatch.setattr(
        tour_od.expressions, "annotate_preprocessors", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(tour_od, "run_od_sample", Mock(return_value=sample))
    monkeypatch.setattr(tour_od, "run_od_logsums", logsums_model)
    monkeypatch.setattr(tour_od, "run_od_simulate", simulate_model)
    state = workflow.State().default_settings()
    state.settings.skip_failed_choices = True
    model_settings = SimpleNamespace(
        SIZE_TERM_SELECTOR="od",
        ORIG_COL_NAME="origin",
        DEST_COL_NAME="destination",
        CHOOSER_SEGMENT_COLUMN_NAME="tour_type",
        SEGMENTS=["work"],
        ORIG_FILTER="original_MAZ > 0",
        LOGSUM_SETTINGS="tour_mode_choice.yaml",
        ALT_DEST_COL_NAME="alt_od",
    )

    choices, _ = tour_od.run_tour_od(
        state,
        tours,
        persons,
        want_logsums=False,
        want_sample_table=False,
        model_settings=model_settings,
        network_los=None,
        estimator=None,
        chunk_size=0,
        trace_hh_id=None,
        trace_label="tour_od_choice",
    )

    failed = tours.loc[~tours.index.isin(sampled_tours), "household_id"]
    assert logit.skipped_household_ids(state) == set(failed)
    assert choices.index.tolist() == expected_tours.index.tolist()
    assert {"choice", "origin", "destination"} <= set(choices.columns)
    if expected_tours.empty:
        logsums_model.assert_not_called()
        simulate_model.assert_not_called()
    else:
        assert simulate_model.call_args.args[2].index.tolist() == [2]
