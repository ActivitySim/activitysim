# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import io
import logging

import pandas as pd

from activitysim.abm.misc import override_hh_ids
from activitysim.abm.tables.util import simple_table_join
from activitysim.core import tracing, workflow
from activitysim.core.input import read_input_table

logger = logging.getLogger(__name__)


@workflow.table
def households(state: workflow.State) -> pd.DataFrame:
    households_sample_size = state.settings.households_sample_size
    _override_hh_ids = override_hh_ids(state)
    _trace_hh_id = state.settings.trace_hh_id

    df_full = read_input_table(state, "households")
    tot_households = df_full.shape[0]

    logger.info("full household list contains %s households" % tot_households)

    households_sliced = False

    # only using households listed in override_hh_ids
    if _override_hh_ids is not None:
        # trace_hh_id will not used if it is not in list of override_hh_ids
        logger.info(
            "override household list containing %s households" % len(_override_hh_ids)
        )

        df = df_full[df_full.index.isin(_override_hh_ids)]
        households_sliced = True

        if df.shape[0] < len(_override_hh_ids):
            logger.info(
                "found %s of %s households in override household list"
                % (df.shape[0], len(_override_hh_ids))
            )

        if df.shape[0] == 0:
            raise RuntimeError("No override households found in store")

    # if we are tracing hh exclusively
    elif _trace_hh_id and households_sample_size == 1:
        # df contains only trace_hh (or empty if not in full store)
        df = tracing.slice_ids(df_full, _trace_hh_id)
        households_sliced = True

    # if we need a subset of full store
    elif tot_households > households_sample_size > 0:
        logger.info(
            "sampling %s of %s households" % (households_sample_size, tot_households)
        )

        """
        Because random seed is set differently for each step, sampling of households using
        Random.global_rng would sample differently depending upon which step it was called from.
        We use a one-off rng seeded with the pseudo step name 'sample_households' to provide
        repeatable sampling no matter when the table is loaded.

        Note that the external_rng is also seeded with base_seed so the sample will (rightly) change
        if the pipeline rng's base_seed is changed
        """

        prng = state.get_rn_generator().get_external_rng("sample_households")
        df = df_full.take(
            prng.choice(len(df_full), size=households_sample_size, replace=False)
        )
        households_sliced = True

        # if tracing and we missed trace_hh in sample, but it is in full store
        if (
            _trace_hh_id
            and _trace_hh_id not in df.index
            and _trace_hh_id in df_full.index
        ):
            # replace first hh in sample with trace_hh
            logger.debug(
                "replacing household %s with %s in household sample"
                % (df.index[0], _trace_hh_id)
            )
            df_hh = df_full.loc[[_trace_hh_id]]
            df = pd.concat([df_hh, df[1:]])

    else:
        df = df_full

    state.set("households_sliced", households_sliced)

    if "sample_rate" not in df.columns:
        if households_sample_size == 0:
            sample_rate = 1
        else:
            sample_rate = round(households_sample_size / tot_households, 3)

        df["sample_rate"] = sample_rate

    logger.info("loaded households %s" % (df.shape,))
    buffer = io.StringIO()
    df.info(buf=buffer)
    logger.debug("households.info:\n" + buffer.getvalue())

    # replace table function with dataframe
    state.add_table("households", df)

    state.get_rn_generator().add_channel("households", df)

    state.tracing.register_traceable_table("households", df)
    if _trace_hh_id:
        state.tracing.trace_df(df, "raw.households", warn_if_empty=True)

    return df


# this is a common merge so might as well define it once here and use it
@workflow.temp_table
def households_merged(
    state: workflow.State,
    households: pd.DataFrame,
    land_use: pd.DataFrame,
    accessibility: pd.DataFrame,
) -> pd.DataFrame:
    households = simple_table_join(
        households,
        land_use,
        left_on="home_zone_id",
    )
    households = simple_table_join(
        households,
        accessibility,
        left_on="home_zone_id",
    )
    return households
