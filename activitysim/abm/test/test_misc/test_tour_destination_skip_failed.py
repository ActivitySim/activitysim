from __future__ import annotations

import pandas as pd
import pytest

from activitysim.abm.models import joint_tour_destination, non_mandatory_destination
from activitysim.core import logit, workflow
from activitysim.core.configuration.logit import TourLocationComponentSettings


def _tour_1_failed_sampling(state, tours, *args, **kwargs):
    """Stand-in for run_tour_destination when tour 1 had no sample.

    The real function records the household of a tour that failed sampling
    and returns choices for the other tours only.
    """
    logit.record_skipped_households(state, tours.loc[[1], "household_id"], "sample")
    choices = pd.DataFrame(
        {"choice": [202], "logsum": [1.5]}, index=pd.Index([2], name="tour_id")
    )
    return choices, None


@pytest.mark.parametrize(
    "step_module, step_name, tour_category, logsum_column",
    [
        (
            joint_tour_destination,
            "joint_tour_destination",
            "joint",
            "destination_logsum",
        ),
        (
            non_mandatory_destination,
            "non_mandatory_tour_destination",
            "non_mandatory",
            None,
        ),
        (
            non_mandatory_destination,
            "non_mandatory_tour_destination",
            "non_mandatory",
            "destination_logsum",
        ),
    ],
    ids=["joint_with_logsums", "non_mandatory", "non_mandatory_with_logsums"],
)
def test_destination_step_drops_tours_of_skipped_households(
    monkeypatch, tmp_path, step_module, step_name, tour_category, logsum_column
):
    """A tour skipped during sampling has no destination; add_table drops it.

    Before this was handled, joint_tour_destination saved the tours table
    before assigning logsums, and assign_in_place then failed on the dropped
    tour. non_mandatory_tour_destination asserted that every tour has a
    destination.
    """
    (tmp_path / "configs").mkdir()
    (tmp_path / "data").mkdir()
    state = workflow.State().default_settings()
    state.initialize_filesystem(working_dir=tmp_path, configs_dir=("configs",))
    state.settings.skip_failed_choices = True
    tours = pd.DataFrame(
        {
            "household_id": [11, 12],
            "person_id": [1, 2],
            "tour_category": [tour_category, tour_category],
            "destination": [-1, -1],
        },
        index=pd.Index([1, 2], name="tour_id"),
    )
    state.add_table("tours", tours)
    monkeypatch.setattr(
        step_module.tour_destination, "run_tour_destination", _tour_1_failed_sampling
    )
    model_settings = TourLocationComponentSettings.model_construct(
        DEST_CHOICE_LOGSUM_COLUMN_NAME=logsum_column,
        DEST_CHOICE_SAMPLE_TABLE_NAME=None,
        ALT_DEST_COL_NAME="alt_dest",
    )

    getattr(step_module, step_name)(
        state,
        tours=state.get_dataframe("tours"),
        persons_merged=pd.DataFrame(),
        network_los=None,
        model_settings=model_settings,
    )

    result = state.get_dataframe("tours")
    assert result.index.tolist() == [2]
    assert result.loc[2, "destination"] == 202
    if logsum_column:
        assert result.loc[2, logsum_column] == 1.5
