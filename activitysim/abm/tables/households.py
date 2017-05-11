# ActivitySim
# See full license in LICENSE.txt.

import logging

import numpy as np
import orca
import pandas as pd

from activitysim.core import simulate as asim
from activitysim.core import tracing
from activitysim.core import pipeline


logger = logging.getLogger(__name__)


@orca.table()
def households(store, households_sample_size, trace_hh_id):

    df_full = store["households"]

    # if we are tracing hh exclusively
    if trace_hh_id and households_sample_size == 1:

        # df contains only trace_hh (or empty if not in full store)
        df = tracing.slice_ids(df_full, trace_hh_id)

    # if we need sample a subset of full store
    elif households_sample_size > 0 and len(df_full.index) > households_sample_size:

        # take the requested random sample
        df = asim.random_rows(df_full, households_sample_size)

        # if tracing and we missed trace_hh in sample, but it is in full store
        if trace_hh_id and trace_hh_id not in df.index and trace_hh_id in df_full.index:
                # replace first hh in sample with trace_hh
                logger.debug("replacing household %s with %s in household sample" %
                             (df.index[0], trace_hh_id))
                df_hh = tracing.slice_ids(df_full, trace_hh_id)
                df = pd.concat([df_hh, df[1:]])

    else:
        df = df_full

    logger.info("loaded households %s" % (df.shape,))

    # replace table function with dataframe
    orca.add_table('households', df)

    pipeline.get_rn_generator().add_channel(df, 'households')

    if trace_hh_id:
        tracing.register_traceable_table('households', df)
        tracing.trace_df(df, "households", warn_if_empty=True)

    return df


# this assigns a chunk_id to each household based on the chunk_size setting
@orca.column("households", cache=True)
def chunk_id(households, hh_chunk_size):

    chunk_ids = pd.Series(range(len(households)), households.index)

    if hh_chunk_size > 0:
        chunk_ids = np.floor(chunk_ids.div(hh_chunk_size)).astype(int)

    return chunk_ids


@orca.column('households')
def work_tour_auto_time_savings(households):
    # FIXME - fix this variable from auto ownership model
    return pd.Series(0, households.index)


# this is the placeholder for all the columns to update after the
# workplace location choice model
@orca.table()
def households_cdap(households):
    return pd.DataFrame(index=households.index)


@orca.column("households_cdap")
def num_under16_not_at_school(persons, households):
    return persons.under16_not_at_school.groupby(persons.household_id).size().\
        reindex(households.index).fillna(0)


# this is a placeholder table for columns that get computed after the
# auto ownership model
@orca.table()
def households_autoown(households):
    return pd.DataFrame(index=households.index)


@orca.column('households_autoown')
def no_cars(households):
    return (households.auto_ownership == 0)


@orca.column('households_autoown')
def car_sufficiency(households, persons):
    return households.auto_ownership - persons.household_id.value_counts()


# this is a common merge so might as well define it once here and use it
@orca.table()
def households_merged(households, land_use, accessibility):
    return orca.merge_tables(households.name, tables=[
        households, land_use, accessibility])


orca.broadcast('households', 'persons', cast_index=True, onto_on='household_id')
