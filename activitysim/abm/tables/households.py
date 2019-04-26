# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402
from builtins import range

import logging

import pandas as pd
import numpy as np

from activitysim.core import tracing
from activitysim.core import pipeline
from activitysim.core import inject

from .input_store import read_input_table

logger = logging.getLogger(__name__)


@inject.table()
def households(households_sample_size, override_hh_ids, trace_hh_id):

    df_full = read_input_table("households")
    households_sliced = False

    logger.info("full household list contains %s households" % df_full.shape[0])

    # only using households listed in override_hh_ids
    if override_hh_ids is not None:

        # trace_hh_id will not used if it is not in list of override_hh_ids
        logger.info("override household list containing %s households" % len(override_hh_ids))

        df = df_full[df_full.index.isin(override_hh_ids)]
        households_sliced = True

        if df.shape[0] < len(override_hh_ids):
            logger.info("found %s of %s households in override household list" %
                        (df.shape[0], len(override_hh_ids)))

        if df.shape[0] == 0:
            raise RuntimeError('No override households found in store')

    # if we are tracing hh exclusively
    elif trace_hh_id and households_sample_size == 1:

        # df contains only trace_hh (or empty if not in full store)
        df = tracing.slice_ids(df_full, trace_hh_id)
        households_sliced = True

    # if we need a subset of full store
    elif households_sample_size > 0 and df_full.shape[0] > households_sample_size:

        logger.info("sampling %s of %s households" % (households_sample_size, df_full.shape[0]))

        """
        Because random seed is set differently for each step, sampling of households using
        Random.global_rng would sample differently depending upon which step it was called from.
        We use a one-off rng seeded with the pseudo step name 'sample_households' to provide
        repeatable sampling no matter when the table is loaded.

        Note that the external_rng is also seeded with base_seed so the sample will (rightly) change
        if the pipeline rng's base_seed is changed
        """

        prng = pipeline.get_rn_generator().get_external_rng('sample_households')
        df = df_full.take(prng.choice(len(df_full), size=households_sample_size, replace=False))
        households_sliced = True

        # if tracing and we missed trace_hh in sample, but it is in full store
        if trace_hh_id and trace_hh_id not in df.index and trace_hh_id in df_full.index:
            # replace first hh in sample with trace_hh
            logger.debug("replacing household %s with %s in household sample" %
                         (df.index[0], trace_hh_id))
            df_hh = df_full.loc[[trace_hh_id]]
            df = pd.concat([df_hh, df[1:]])

    else:
        df = df_full

    # persons table
    inject.add_injectable('households_sliced', households_sliced)

    logger.info("loaded households %s" % (df.shape,))

    df.index.name = 'household_id'

    # FIXME - pathological knowledge of name of chunk_id column used by chunked_choosers_by_chunk_id
    assert 'chunk_id' not in df.columns
    df['chunk_id'] = pd.Series(list(range(len(df))), df.index)

    # replace table function with dataframe
    inject.add_table('households', df)

    pipeline.get_rn_generator().add_channel('households', df)

    if trace_hh_id:
        tracing.register_traceable_table('households', df)
        tracing.trace_df(df, "raw.households", warn_if_empty=True)

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
