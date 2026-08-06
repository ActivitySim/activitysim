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
            sample_method="monte_carlo",
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
