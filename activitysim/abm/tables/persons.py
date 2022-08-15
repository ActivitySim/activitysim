# ActivitySim
# See full license in LICENSE.txt.
import logging

import pandas as pd

from activitysim.core import inject, mem, pipeline, tracing
from activitysim.core.input import read_input_table

logger = logging.getLogger(__name__)


def read_raw_persons(households):

    df = read_input_table("persons")

    if inject.get_injectable("households_sliced", False):
        # keep only persons in the sampled households
        df = df[df.household_id.isin(households.index)]

    return df


@inject.table()
def persons(households, trace_hh_id):

    df = read_raw_persons(households)

    logger.info("loaded persons %s" % (df.shape,))

    # replace table function with dataframe
    inject.add_table("persons", df)

    pipeline.get_rn_generator().add_channel("persons", df)

    tracing.register_traceable_table("persons", df)
    if trace_hh_id:
        tracing.trace_df(df, "raw.persons", warn_if_empty=True)

    logger.debug(f"{len(df.household_id.unique())} unique household_ids in persons")
    logger.debug(f"{len(households.index.unique())} unique household_ids in households")
    assert not households.index.duplicated().any()
    assert not df.index.duplicated().any()

    persons_without_households = ~df.household_id.isin(households.index)
    if persons_without_households.any():
        logger.error(
            f"{persons_without_households.sum()} persons out of {len(persons)} without households\n"
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


# another common merge for persons
@inject.table()
def persons_merged(persons, households, land_use, accessibility):

    return inject.merge_tables(
        persons.name, tables=[persons, households, land_use, accessibility]
    )
