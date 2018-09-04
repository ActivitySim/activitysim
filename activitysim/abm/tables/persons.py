# ActivitySim
# See full license in LICENSE.txt.

import logging

import pandas as pd

from activitysim.core import pipeline
from activitysim.core import inject
from activitysim.core import tracing
from activitysim.core.util import other_than, reindex

from input_store import read_input_table

logger = logging.getLogger(__name__)


def read_raw_persons(households):

    # we only use these to know whether we need to slice persons
    households_sample_size = inject.get_injectable('households_sample_size')
    override_hh_ids = inject.get_injectable('override_hh_ids')

    df = read_input_table("persons")

    if (households_sample_size > 0) or override_hh_ids:
        # keep all persons in the sampled households
        df = df[df.household_id.isin(households.index)]

    return df


@inject.table()
def persons(households, trace_hh_id):

    df = read_raw_persons(households)

    logger.info("loaded persons %s" % (df.shape,))

    df.index.name = 'person_id'

    # replace table function with dataframe
    inject.add_table('persons', df)

    pipeline.get_rn_generator().add_channel(df, 'persons')

    if trace_hh_id:
        tracing.register_traceable_table('persons', df)
        tracing.trace_df(df, "raw.persons", warn_if_empty=True)

    return df


# another common merge for persons
@inject.table()
def persons_merged(persons, households, land_use, accessibility):
    return inject.merge_tables(persons.name, tables=[
        persons, households, land_use, accessibility])
