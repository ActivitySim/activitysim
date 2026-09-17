from __future__ import annotations

import weakref
from contextlib import nullcontext
from types import SimpleNamespace

import pandas as pd
import pytest

from activitysim.abm.models import trip_mode_choice as trip_mode_choice_module


class _DummySkimWrapper:
    df = None

    def set_df(self, df):
        self.df = df


class _DummySkimDict:
    def wrap_3d(self, **_kwargs):
        return _DummySkimWrapper()

    def wrap(self, *_args):
        return _DummySkimWrapper()

    def map_time_periods_from_series(self, periods):
        return periods.map({"AM": 0, "PM": 1})


class _DummyState:
    current_model_name = "test_trip_mode_choice"

    def __init__(self, trips):
        self.settings = SimpleNamespace(
            downcast_int=False,
            downcast_float=False,
            skip_failed_choices=False,
            trace_hh_id=None,
        )
        self.filesystem = SimpleNamespace(
            read_model_spec=lambda **_kwargs: pd.DataFrame(),
            get_segment_coefficients=lambda *_args: {},
        )
        self.tables = {"trips": trips}

    def add_table(self, name, df):
        self.tables[name] = df

    def get_dataframe(self, name, columns=None, as_copy=True):
        df = self.tables[name]
        if columns is not None:
            df = df[columns]
        return df.copy() if as_copy else df

    def is_table(self, _name):
        return False


@pytest.mark.parametrize("keep_trip_period", [False, True])
@pytest.mark.parametrize("copy_annotation_table", [False, True])
@pytest.mark.parametrize("annotation_error", [False, True])
def test_post_choice_annotations_preserve_requested_trip_period(
    monkeypatch, keep_trip_period, copy_annotation_table, annotation_error
):
    trips = pd.DataFrame(
        {
            "tour_id": [11, 12, 13],
            "household_id": [1, 2, 3],
            "primary_purpose": ["work", "shopping", "work"],
            "depart": [8, 17, 9],
            "origin": [1, 2, 3],
            "destination": [2, 3, 1],
        },
        index=pd.Index([101, 102, 103], name="trip_id"),
    )
    state = _DummyState(trips)
    skim_dict = _DummySkimDict()
    network_los = SimpleNamespace(
        skim_time_periods=SimpleNamespace(period_minutes=60),
        skim_time_period_label=lambda depart, as_cat=True: depart.map(
            lambda value: "AM" if value < 12 else "PM"
        ),
        get_default_skim_dict=lambda: skim_dict,
    )
    model_settings = SimpleNamespace(
        MODE_CHOICE_LOGSUM_COLUMN_NAME="mode_choice_logsum",
        TOURS_MERGED_CHOOSER_COLUMNS=[],
        CHOOSER_COLS_TO_KEEP=["trip_period"] if keep_trip_period else [],
        FORCE_ESCORTEE_CHAUFFEUR_MODE_MATCH=False,
        SPEC="trip_mode_choice.csv",
        explicit_chunk=None,
        compute_settings=None,
    )

    monkeypatch.setattr(
        trip_mode_choice_module.tracing, "print_summary", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        trip_mode_choice_module.config, "get_model_constants", lambda *_args: {}
    )
    monkeypatch.setattr(
        trip_mode_choice_module.config, "get_logit_model_settings", lambda *_args: None
    )
    monkeypatch.setattr(
        trip_mode_choice_module.simulate,
        "eval_coefficients",
        lambda _state, spec, *_args: spec,
    )
    monkeypatch.setattr(
        trip_mode_choice_module.simulate,
        "eval_nest_coefficients",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        trip_mode_choice_module.expressions,
        "annotate_preprocessors",
        lambda *_args, **_kwargs: None,
    )

    chooser_refs = []
    wrappers = []

    def choose_mode(_state, choosers, **_kwargs):
        assert all(ref() is None for ref in chooser_refs)
        chooser_refs.append(weakref.ref(choosers))
        wrappers[:] = _kwargs["skims"].values()
        trip_mode_choice_module.simulate.set_skim_wrapper_targets(
            choosers, _kwargs["skims"]
        )
        return pd.DataFrame(
            {
                "trip_mode": "DRIVE",
                "mode_choice_logsum": 1.0,
            },
            index=choosers.index,
        )

    monkeypatch.setattr(trip_mode_choice_module, "mode_choice_simulate", choose_mode)

    def annotate_tables(_state, **_kwargs):
        annotated = _state.get_dataframe("trips", as_copy=copy_annotation_table)
        assert annotated.index.equals(trips.index)
        assert annotated["trip_period"].tolist() == [0, 1, 0]
        annotated["post_choice_skim_value"] = [10.0, 20.0, 30.0]
        _state.add_table("trips", annotated)
        if annotation_error:
            raise RuntimeError("annotation failed")

    monkeypatch.setattr(
        trip_mode_choice_module.expressions, "annotate_tables", annotate_tables
    )

    with pytest.raises(
        RuntimeError, match="annotation failed"
    ) if annotation_error else nullcontext():
        trip_mode_choice_module.trip_mode_choice(
            state,
            trips,
            network_los,
            model_settings=model_settings,
        )

    assert all(ref() is None for ref in chooser_refs)
    assert all(wrapper.df.empty for wrapper in wrappers)
    result = state.get_dataframe("trips", as_copy=False)
    assert ("trip_period" in trips) == keep_trip_period
    assert ("trip_period" in result) == keep_trip_period
    if keep_trip_period:
        assert result["trip_period"].tolist() == [0, 1, 0]
    assert result["post_choice_skim_value"].tolist() == [10.0, 20.0, 30.0]
