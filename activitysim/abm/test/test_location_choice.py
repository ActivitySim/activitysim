from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pandas.testing as pdt
import pytest

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
            skip_failed_choices=False,
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


@pytest.mark.parametrize("sampled_person_ids", [[], [2, 2]])
def test_location_choice_skips_failed_samples(monkeypatch, sampled_person_ids):
    """Only persons with sampled alternatives reach logsums and final choice."""
    persons = pd.DataFrame(
        {"segment": [1, 1]}, index=pd.Index([1, 2], name="person_id")
    )
    location_sample = pd.DataFrame(
        {"alt_dest": [101, 202][: len(sampled_person_ids)]},
        index=pd.Index(sampled_person_ids, name="person_id"),
    )
    expected_persons = persons.loc[persons.index.isin(sampled_person_ids)]
    sample_model = Mock(return_value=location_sample)
    logsums_model = Mock(
        return_value=location_sample.assign(**{location_choice.ALT_LOGSUM: 0.0})
    )
    simulate_model = Mock(
        return_value=pd.DataFrame({"choice": 101}, index=expected_persons.index)
    )
    monkeypatch.setattr(location_choice, "run_location_sample", sample_model)
    monkeypatch.setattr(location_choice, "run_location_logsums", logsums_model)
    monkeypatch.setattr(location_choice, "run_location_simulate", simulate_model)
    shadow_price_calculator = Mock()
    shadow_price_calculator.dest_size_terms.return_value = pd.Series(
        [1.0, 1.0], index=[101, 202]
    )
    model_settings = SimpleNamespace(
        ALT_DEST_COL_NAME="alt_dest",
        CHOOSER_SEGMENT_COLUMN_NAME="segment",
        LOGSUM_SETTINGS="tour_mode_choice.yaml",
        SEGMENT_IDS={"workers": 1},
    )
    state = SimpleNamespace(
        settings=SimpleNamespace(
            skip_failed_choices=True, use_explicit_error_terms=True
        )
    )

    choices, sample = location_choice.run_location_choice(
        state=state,
        persons_merged_df=persons,
        network_los=Mock(),
        shadow_price_calculator=shadow_price_calculator,
        want_logsums=False,
        want_sample_table=True,
        estimator=None,
        model_settings=model_settings,
        chunk_size=0,
        chunk_tag="workplace_location",
        trace_label="workplace_location",
    )

    if expected_persons.empty:
        assert choices.empty
        assert sample is None
        logsums_model.assert_not_called()
        simulate_model.assert_not_called()
    else:
        pdt.assert_index_equal(choices.index, expected_persons.index)
        pdt.assert_frame_equal(logsums_model.call_args.args[2], expected_persons)
        pdt.assert_frame_equal(simulate_model.call_args.args[2], expected_persons)
        pdt.assert_index_equal(
            sample.index.get_level_values("person_id").unique(),
            expected_persons.index,
        )
