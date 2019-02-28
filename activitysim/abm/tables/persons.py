# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402

import logging

from activitysim.core import pipeline
from activitysim.core import inject
from activitysim.core import tracing

from .input_store import read_input_table

logger = logging.getLogger(__name__)


def read_raw_persons(households):

    df = read_input_table("persons")

    if inject.get_injectable('households_sliced', False):
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

    pipeline.get_rn_generator().add_channel('persons', df)

    if trace_hh_id:
        tracing.register_traceable_table('persons', df)
        tracing.trace_df(df, "raw.persons", warn_if_empty=True)

    return df


# another common merge for persons
@inject.table()
def persons_merged(persons, households, land_use, accessibility):
    return inject.merge_tables(persons.name, tables=[
        persons, households, land_use, accessibility])
