# ActivitySim
# See full license in LICENSE.txt.

import logging

import pandas as pd

from activitysim.core import pipeline
from activitysim.core import inject
from activitysim.core import tracing
from activitysim.core.util import other_than, reindex

from constants import *

logger = logging.getLogger(__name__)


@inject.table()
def persons(store, households_sample_size, households, trace_hh_id):

    df = store["persons"]

    if households_sample_size > 0:
        # keep all persons in the sampled households
        df = df[df.household_id.isin(households.index)]

    logger.info("loaded persons %s" % (df.shape,))

    # replace table function with dataframe
    inject.add_table('persons', df)

    pipeline.get_rn_generator().add_channel(df, 'persons')

    if trace_hh_id:
        tracing.register_traceable_table('persons', df)
        tracing.trace_df(df, "persons", warn_if_empty=True)

    return df


# another common merge for persons
@inject.table()
def persons_merged(persons, households, land_use, accessibility):
    return inject.merge_tables(persons.name, tables=[
        persons, households, land_use, accessibility])
