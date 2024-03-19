# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import io
import logging

import pandas as pd

from activitysim.abm.tables.util import simple_table_join
from activitysim.core import workflow
from activitysim.core.input import read_input_table

logger = logging.getLogger(__name__)


def read_raw_persons(state, households):
    df = read_input_table(state, "persons")

    if state.get_injectable("households_sliced", False):
        # keep only persons in the sampled households
        df = df[df.household_id.isin(households.index)]

    return df


@workflow.table
def persons(state: workflow.State) -> pd.DataFrame:
    households = state.get_dataframe("households")
    trace_hh_id = state.settings.trace_hh_id
    df = read_raw_persons(state, households)

    logger.info("loaded persons %s" % (df.shape,))
    buffer = io.StringIO()
    df.info(buf=buffer)
    logger.debug("persons.info:\n" + buffer.getvalue())

    # replace table function with dataframe
    state.add_table("persons", df)

    state.get_rn_generator().add_channel("persons", df)

    state.tracing.register_traceable_table("persons", df)
    if trace_hh_id:
        state.tracing.trace_df(df, "raw.persons", warn_if_empty=True)

    logger.debug(f"{len(df.household_id.unique())} unique household_ids in persons")
    logger.debug(f"{len(households.index.unique())} unique household_ids in households")
    assert not households.index.duplicated().any()
    assert not df.index.duplicated().any()

    persons_without_households = ~df.household_id.isin(households.index)
    if persons_without_households.any():
        logger.error(
            f"{persons_without_households.sum()} persons out of {len(df)} without households\n"
            f"{pd.Series({'person_id': persons_without_households.index.values})}"
        )
        raise RuntimeError(
            f"{persons_without_households.sum()} persons with bad household_id"
        )

    households_without_persons = (
        df.groupby("household_id").size().reindex(households.index).isnull()
    )
    if households_without_persons.any():
        logger.error(
            f"{households_without_persons.sum()} households out of {len(households.index)} without  persons\n"
            f"{pd.Series({'household_id': households_without_persons.index.values})}"
        )
        raise RuntimeError(
            f"{households_without_persons.sum()} households with no persons"
        )

    return df


@workflow.temp_table
def persons_merged(
    state: workflow.State,
    persons: pd.DataFrame,
    land_use: pd.DataFrame,
    households: pd.DataFrame,
    accessibility: pd.DataFrame,
    disaggregate_accessibility: pd.DataFrame = None,
):
    n_persons = len(persons)
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
    persons = simple_table_join(
        persons,
        households,
        left_on="household_id",
    )
    if state.is_table("disaggregate_accessibility"):
        disaggregate_accessibility = state.get_table("disaggregate_accessibility")
        persons = simple_table_join(
            persons,
            disaggregate_accessibility,
            left_on="person_id",
        )
    if n_persons != len(persons):
        raise RuntimeError("number of persons changed")
    return persons
