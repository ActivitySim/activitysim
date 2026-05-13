from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from activitysim.abm.models.util import tour_od
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


def test_od_presample_passes_full_taz_index_for_eet(monkeypatch):
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
