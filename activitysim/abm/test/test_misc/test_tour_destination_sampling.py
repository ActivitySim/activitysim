from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from activitysim.abm.models.util import tour_destination
from activitysim.core import logit, los, workflow


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


@pytest.mark.parametrize("use_explicit_error_terms", [False, True], ids=["mnl", "eet"])
def test_destination_presample_returns_empty_sample_when_every_chooser_failed(
    monkeypatch, use_explicit_error_terms
):
    """An empty TAZ sample is returned as is, without choosing MAZs.

    Under skip_failed_choices, the TAZ sample is empty when every chooser in the
    segment failed sampling. choose_MAZ_for_TAZ raises on an empty sample.
    """

    def fake_destination_sample(*_args, **_kwargs):
        # interaction_sample's result when every chooser failed
        return pd.DataFrame(
            {
                tour_destination.DEST_TAZ: pd.Series(dtype="int64"),
                "prob": pd.Series(dtype="float32"),
                "pick_count": pd.Series(dtype="uint32"),
                "person_id": pd.Series(dtype="int64"),
            },
            index=pd.Index([], name="tour_id", dtype="int64"),
        )

    monkeypatch.setattr(
        tour_destination, "_destination_sample", fake_destination_sample
    )

    state = workflow.State().default_settings()
    state.settings.skip_failed_choices = True
    state.settings.use_explicit_error_terms = use_explicit_error_terms
    choosers = pd.DataFrame(
        {"origin": [101, 103]},
        index=pd.Index([7001, 7002], name="tour_id"),
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
    destination_size_terms = pd.DataFrame(
        {"size_term": [1.0, 2.0, 3.0]},
        index=pd.Index([101, 102, 103], name="zone_id"),
    )

    out = tour_destination.destination_presample(
        state,
        "segment",
        choosers,
        model_settings,
        network_los,
        destination_size_terms,
        destination_size_terms,
        estimator=None,
        trace_label="test_trace",
    )

    assert out.empty
    assert out.index.name == "tour_id"
    assert list(out.columns) == ["zone_id", "prob", "pick_count", "person_id"]


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


@pytest.mark.parametrize("sampled_tours", [[], [5, 5, 6]], ids=["empty", "sampled"])
def test_run_destination_sample_copies_person_id_without_adding_rows(
    monkeypatch, sampled_tours
):
    """person_id is broadcast onto sampled rows; an empty sample stays empty."""
    tours = pd.DataFrame({"person_id": [7, 8]}, index=pd.Index([5, 6], name="tour_id"))
    persons_merged = pd.DataFrame(
        {"home_zone_id": [1, 2]}, index=pd.Index([7, 8], name="person_id")
    )
    sample = pd.DataFrame(
        {"alt_dest": [10, 20, 10][: len(sampled_tours)]},
        index=pd.Index(sampled_tours, name="tour_id", dtype="int64"),
    )
    monkeypatch.setattr(
        tour_destination, "destination_sample", lambda *args, **kwargs: sample.copy()
    )

    result = tour_destination.run_destination_sample(
        SimpleNamespace(settings=SimpleNamespace(want_dest_choice_presampling=True)),
        "shopping",
        tours,
        persons_merged,
        SimpleNamespace(CHOOSER_ID_COLUMN="person_id"),
        SimpleNamespace(zone_system=los.ONE_ZONE),
        None,
        None,
        None,
        chunk_size=0,
        trace_label="test",
    )

    assert result.index.tolist() == sampled_tours
    assert result["person_id"].tolist() == [7, 7, 8][: len(sampled_tours)]
    assert result["alt_dest"].notna().all()


class _SizeTerms:
    def __init__(self, state, size_term_selector):
        self.destination_size_terms = pd.DataFrame(
            {"shopping": [1.0, 1.0]}, index=pd.Index([101, 202], name="zone_id")
        )

    def dest_size_terms_df(self, segment_name, trace_label):
        return self.destination_size_terms[[segment_name]].rename(
            columns={segment_name: "size_term"}
        )


@pytest.mark.parametrize(
    "sampled_tours", [[], [2, 2]], ids=["all_failed", "one_failed"]
)
def test_run_tour_destination_skips_failed_samples(monkeypatch, sampled_tours):
    """Tours without a sample skip logsums and simulation; their households are recorded.

    The sample choosers carry no household_id, so sampling cannot record them.
    """
    tours = pd.DataFrame(
        {
            "household_id": [11, 12],
            "person_id": [1, 2],
            "tour_type": ["shopping", "shopping"],
        },
        index=pd.Index([1, 2], name="tour_id"),
    )
    sample = pd.DataFrame(
        {"alt_dest": [101, 202][: len(sampled_tours)]},
        index=pd.Index(sampled_tours, name="tour_id", dtype="int64"),
    )
    expected_tours = tours.loc[tours.index.isin(sampled_tours)]
    logsums_model = Mock(return_value=sample.assign(mode_choice_logsum=0.0))
    simulate_model = Mock(
        side_effect=lambda state, segment, choosers, *args, **kwargs: pd.DataFrame(
            {"choice": 101}, index=choosers.index
        )
    )
    monkeypatch.setattr(tour_destination, "SizeTermCalculator", _SizeTerms)
    monkeypatch.setattr(
        tour_destination, "run_destination_sample", Mock(return_value=sample)
    )
    monkeypatch.setattr(tour_destination, "run_destination_logsums", logsums_model)
    monkeypatch.setattr(tour_destination, "run_destination_simulate", simulate_model)
    state = workflow.State().default_settings()
    state.settings.skip_failed_choices = True
    model_settings = SimpleNamespace(
        SIZE_TERM_SELECTOR="non_mandatory",
        SEGMENTS=["shopping"],
        CHOOSER_SEGMENT_COLUMN_NAME="tour_type",
        LOGSUM_SETTINGS="tour_mode_choice.yaml",
        ALT_DEST_COL_NAME="alt_dest",
    )

    choices, _ = tour_destination.run_tour_destination(
        state,
        tours,
        persons_merged=pd.DataFrame(),
        want_logsums=False,
        want_sample_table=False,
        model_settings=model_settings,
        network_los=None,
        estimator=None,
        trace_label="non_mandatory_tour_destination",
    )

    failed = tours.loc[~tours.index.isin(sampled_tours), "household_id"]
    assert logit.skipped_household_ids(state) == set(failed)
    assert choices.index.tolist() == expected_tours.index.tolist()
    if expected_tours.empty:
        logsums_model.assert_not_called()
        simulate_model.assert_not_called()
    else:
        pd.testing.assert_frame_equal(simulate_model.call_args.args[2], expected_tours)
