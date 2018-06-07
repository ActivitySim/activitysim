# ActivitySim
# See full license in LICENSE.txt.

import logging
import os

import pandas as pd

from activitysim.core import simulate as asim
from activitysim.core import tracing
from activitysim.core import pipeline

from activitysim.core import inject

logger = logging.getLogger(__name__)


@inject.table()
def households(store, households_sample_size, trace_hh_id, override_hh_ids):

    df_full = store["households"]

    # only using households listed in override_hh_ids
    if override_hh_ids is not None:

        # trace_hh_id will not used if it is not in list of override_hh_ids
        logger.info("override household list containing %s households" % len(override_hh_ids))
        df = tracing.slice_ids(df_full, override_hh_ids)

    # if we are tracing hh exclusively
    elif trace_hh_id and households_sample_size == 1:

        # df contains only trace_hh (or empty if not in full store)
        df = tracing.slice_ids(df_full, trace_hh_id)

    # if we need a subset of full store
    elif households_sample_size > 0 and df_full.shape[0] > households_sample_size:

        logger.info("sampling %s of %s households" % (households_sample_size, df_full.shape[0]))

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

    # FIXME - pathological knowledge of name of chunk_id column used by chunked_choosers_by_chunk_id
    assert 'chunk_id' not in df.columns
    df['chunk_id'] = pd.Series(range(len(df)), df.index)

    # replace table function with dataframe
    inject.add_table('households', df)

    pipeline.get_rn_generator().add_channel(df, 'households')

    if trace_hh_id:
        tracing.register_traceable_table('households', df)
        tracing.trace_df(df, "households", warn_if_empty=True)

    return df


# this is a common merge so might as well define it once here and use it
@inject.table()
def households_merged(households, land_use, accessibility):
    return inject.merge_tables(households.name, tables=[
        households, land_use, accessibility])


inject.broadcast('households', 'persons', cast_index=True, onto_on='household_id')

# this would be accessibility around the household location - be careful with
# this one as accessibility at some other location can also matter
inject.broadcast('accessibility', 'households', cast_index=True, onto_on='TAZ')
