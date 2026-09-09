from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pandas.testing as pdt

from activitysim.abm.models import location_choice


def test_estimation_override_preserves_destination_choice_logsum(monkeypatch):
    """Survey overrides should only change the chosen destination's mode logsum."""
    person_index = pd.Index([1], name="person_id")
    location_sample = pd.DataFrame(
        {
            "alt_dest": [101, 202],
            location_choice.ALT_LOGSUM: [5.0, 7.0],
        },
        index=pd.Index([1, 1], name="person_id"),
    )
    modeled_choices = pd.DataFrame(
        {"choice": [101], "logsum": [-1.5]}, index=person_index
    )

    # Keep the test focused on the estimation override and final logsum merge.
    monkeypatch.setattr(
        location_choice,
        "run_location_sample",
        lambda *args, **kwargs: location_sample.copy(),
    )
    monkeypatch.setattr(
        location_choice,
        "run_location_logsums",
        lambda *args, **kwargs: location_sample.copy(),
    )
    monkeypatch.setattr(
        location_choice,
        "run_location_simulate",
        lambda *args, **kwargs: modeled_choices.copy(),
    )

    estimator = Mock()
    estimator.get_survey_values.return_value = pd.Series(
        [202], index=person_index, name="choice"
    )
    shadow_price_calculator = Mock()
    shadow_price_calculator.dest_size_terms.return_value = pd.Series(
        [1.0, 1.0], index=[101, 202]
    )
    model_settings = SimpleNamespace(
        ALT_DEST_COL_NAME="alt_dest",
        CHOOSER_SEGMENT_COLUMN_NAME="segment",
        DEST_CHOICE_COLUMN_NAME="workplace_zone_id",
        LOGSUM_SETTINGS="tour_mode_choice.yaml",
        SEGMENT_IDS={"workers": 1},
    )
    state = SimpleNamespace(
        settings=SimpleNamespace(
            sample_method="inverse_cdf",
            trace_hh_id=None,
            use_explicit_error_terms=False,
        )
    )
    persons = pd.DataFrame({"segment": [1]}, index=person_index)

    choices, sample = location_choice.run_location_choice(
        state=state,
        persons_merged_df=persons,
        network_los=Mock(),
        shadow_price_calculator=shadow_price_calculator,
        want_logsums=True,
        want_sample_table=False,
        estimator=estimator,
        model_settings=model_settings,
        chunk_size=0,
        chunk_tag="workplace_location",
        trace_label="workplace_location",
    )

    expected = pd.DataFrame(
        {
            "choice": [202],
            "logsum": [-1.5],
            location_choice.ALT_LOGSUM: [7.0],
        },
        index=person_index,
    )
    pdt.assert_frame_equal(choices, expected)
    assert sample is None


def test_location_logsums_join_person_attributes_inside_chunks(monkeypatch):
    person_index = pd.Index([1, 2], name="person_id")
    persons = pd.DataFrame({"income": [10, 20]}, index=person_index)
    sample = pd.DataFrame(
        {"alt_dest": [101, 102, 201, 202, 203]},
        index=pd.Index([1, 1, 2, 2, 2], name="person_id"),
    )
    model_settings = SimpleNamespace(
        LOGSUM_SETTINGS="tour_mode_choice.yaml",
        LOGSUM_TOUR_PURPOSE="work",
        explicit_chunk=0.5,
    )
    state = SimpleNamespace(filesystem=Mock())
    chunk_sizer = Mock()

    monkeypatch.setattr(
        location_choice.TourModeComponentSettings,
        "read_settings_file",
        lambda *args, **kwargs: SimpleNamespace(),
    )

    def chunked(*args, **kwargs):
        assert args[1] is persons
        assert args[2] is sample
        assert kwargs["chunk_size"] == 123
        assert kwargs["explicit_chunk_size"] == 0.5
        yield 1, persons.iloc[:1], sample.iloc[:2], "logsums.i1", chunk_sizer
        yield 2, persons.iloc[1:], sample.iloc[2:], "logsums.i2", chunk_sizer

    monkeypatch.setattr(
        location_choice.chunk, "adaptive_chunked_choosers_and_alts", chunked
    )
    chooser_lengths = []

    def compute_logsums(_state, choosers, *args, **kwargs):
        chooser_lengths.append(len(choosers))
        assert args[4] == 0
        assert kwargs["explicit_chunk_size"] == 0
        return choosers["alt_dest"] + choosers["income"]

    monkeypatch.setattr(
        location_choice.logsum, "compute_location_choice_logsums", compute_logsums
    )

    result = location_choice.run_location_logsums(
        state,
        "work",
        persons,
        Mock(),
        sample,
        model_settings,
        chunk_size=123,
        chunk_tag="school_location.logsums",
        trace_label="school_location.logsums.work",
    )

    assert chooser_lengths == [2, 3]
    assert result[location_choice.ALT_LOGSUM].tolist() == [111, 112, 221, 222, 223]
