# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from activitysim.core import estimation, workflow
from activitysim.core.interaction_sample import resolve_sample_method

logger = logging.getLogger(__name__)


def maybe_bias_logsums(state: workflow.State, choices_df: pd.DataFrame, model_settings):
    """Check for temporary fix to bias logsums for Poisson sampling results to align with MC/eet sampling."""

    if estimation.manager.enabled:
        raise RuntimeError("maybe_bias_logsums should not be called during estimation.")

    sample_method = resolve_sample_method(state, model_settings)
    if (
        (sample_method == "poisson")
        and (model_settings.SAMPLE_SIZE > 0)
        and not state.settings.disable_destination_sampling
    ):
        # Only apply for sample size > 0, for unsampled disagg acc the MC/eet results are unbiased and we
        # want to stay consistent.
        if state.settings.bias_location_choice_logsums_for_poisson_sampling:
            logger.warning(
                "Applying bias correction to location logsums with Poisson sampling to align with MC/eet sampling."
            )
            # it looks like the logsum column can be named either "logsum" or "logsums", depending on if choices get skipped.
            if "logsum" in choices_df.columns:
                choices_df["logsum"] += np.log(model_settings.SAMPLE_SIZE)
            if "logsums" in choices_df.columns:
                choices_df["logsums"] += np.log(model_settings.SAMPLE_SIZE)
        else:
            logger.warning(
                "Using Poisson sampling method for location choice logsum calculations. Currently the logsums results will"
                + " differ from those obtained with monte_carlo or eet sampling by a constant shift of"
                + f" log({model_settings.SAMPLE_SIZE}) if using the common correction factor"
                + " `log(pick_count / prob)` in location choice specs. The results of the Poisson method are unbiased,"
                + " i.e., they agree with the results obtained with a full destination sample, unlike those for"
                + " monte_carlo or eet sampling."
            )

    return choices_df
