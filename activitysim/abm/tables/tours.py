# ActivitySim
# See full license in LICENSE.txt.
import logging

import pandas as pd

from activitysim.core import inject, workflow

logger = logging.getLogger(__name__)


@workflow.temp_table
def tours_merged(
    whale: workflow.Whale, tours: pd.DataFrame, persons_merged: pd.DataFrame
):
    # return inject.merge_tables(tours.name, tables=[tours, persons_merged])
    def join(left, right, left_on):
        intersection = set(left.columns).intersection(right.columns)
        intersection.discard(left_on)  # intersection is ok if it's the join key
        right = right.drop(intersection, axis=1)
        return pd.merge(
            left,
            right,
            left_on=left_on,
            right_index=True,
        )

    return join(
        tours,
        persons_merged,
        left_on="person_id",
    )


# inject.broadcast("persons_merged", "tours", cast_index=True, onto_on="person_id")
