# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from activitysim.core import config, workflow

logger = logging.getLogger(__name__)


@workflow.step
def reassign_tour_purpose_by_poe(state: workflow.State, tours: pd.DataFrame) -> None:
    """
    Simulates tour purpose choices after tour origin has been assigned. This
    is useful when the original tour purposes are assigned randomly
    from an aggregate distribution that was not segmented by tour origin.
    """

    trace_label = "reassign_tour_purpose_by_poe"
    probs_df = pd.read_csv(
        state.filesystem.get_config_file_path("tour_purpose_probs_by_poe.csv")
    )
    probs_df.columns = [
        col if col in ["Purpose", "Description"] else int(col)
        for col in probs_df.columns
    ]

    tours_df = tours[["tour_type", "poe_id"]]
    tour_types = probs_df[["Purpose", "Description"]].set_index("Purpose")[
        "Description"
    ]

    tours_df["purpose_id"] = None
    for poe, group in tours_df.groupby("poe_id"):
        num_tours = len(group)
        purpose_probs = probs_df[poe]
        purpose_cum_probs = purpose_probs.values.cumsum()
        rands = state.get_rn_generator().random_for_df(group)
        purpose_scaled_probs = np.subtract(purpose_cum_probs, rands)
        purpose = np.argmax((purpose_scaled_probs + 1.0).astype("i4"), axis=1)
        tours_df.loc[group.index, "purpose_id"] = purpose
    tours_df["new_tour_type"] = tours_df["purpose_id"].map(tour_types)

    tours["tour_type"] = tours_df["new_tour_type"].reindex(tours.index)
    tours["purpose_id"] = tours_df["purpose_id"].reindex(tours.index)
    tours["tour_category"] = "non_mandatory"
    tours.loc[tours["tour_type"].isin(["home", "work"]), "tour_category"] = "mandatory"

    state.add_table("tours", tours)

    return
