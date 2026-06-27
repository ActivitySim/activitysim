# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import numpy as np
import pandas as pd

from activitysim.core import workflow


def draw_maz_rands(
    state: workflow.State,
    chooser_df: pd.DataFrame,
    taz_choices: pd.DataFrame,
    taz_choice_counts: pd.Series,
    taz_sample_size: int,
    maz_probs: np.ndarray,
    max_maz_count: int,
    uniform_taz_choice_counts: bool,
    dest_taz_col: str,
    full_taz_index: pd.Index | None = None,
) -> np.ndarray:
    """
    Draw uniform random numbers for the MAZ-within-TAZ choice step.

    Three modes, selected by the inputs:

    - `full_taz_index is not None` (EET-stable / Poisson MAZ-for-TAZ): draw
      `len(full_taz_index)` uniforms per chooser keyed to the fixed TAZ
      universe, then project to the active TAZ rows via
      `full_taz_index.get_indexer(taz_choices[dest_taz_col])`. Gives
      cross-scenario stability when the TAZ universe is the same.
    - `uniform_taz_choice_counts` (MC / EET with identical per-chooser TAZ
      sample size): draw one uniform per (chooser, TAZ-rank).
    - otherwise (MC / EET with variable per-chooser TAZ sample size): draw
      `taz_sample_size` uniforms per chooser, then mask to each chooser's
      actual TAZ count.

    Returns a 2-D array of shape `(maz_probs.shape[0], 1)` with the per-TAZ-row
    uniform draw used to pick a MAZ within that TAZ.
    """
    if full_taz_index is not None:
        full_taz_index = pd.Index(full_taz_index, name=dest_taz_col)
        taz_positions = full_taz_index.get_indexer(taz_choices[dest_taz_col])
        assert (taz_positions >= 0).all()
        chooser_rands = np.asarray(
            state.get_rn_generator().random_for_df(chooser_df, n=len(full_taz_index))
        )
        chooser_row_positions = np.repeat(
            np.arange(len(chooser_df)), taz_choice_counts.to_numpy()
        )
        rands = chooser_rands[chooser_row_positions, taz_positions].reshape(-1, 1)
        assert len(rands) == len(taz_choices)
    elif uniform_taz_choice_counts:
        assert maz_probs.shape == (len(chooser_df) * taz_sample_size, max_maz_count)
        rands = (
            state.get_rn_generator()
            .random_for_df(chooser_df, n=taz_sample_size)
            .reshape(-1, 1)
        )
        assert len(rands) == len(chooser_df) * taz_sample_size
    else:
        assert maz_probs.shape == (len(taz_choices), max_maz_count)
        chooser_rands = np.asarray(
            state.get_rn_generator().random_for_df(chooser_df, n=taz_sample_size)
        )
        chooser_rand_mask = (
            np.arange(taz_sample_size) < taz_choice_counts.to_numpy()[:, np.newaxis]
        )
        rands = chooser_rands[chooser_rand_mask].reshape(-1, 1)
        assert len(rands) == len(taz_choices)
    assert len(rands) == maz_probs.shape[0]
    return rands
