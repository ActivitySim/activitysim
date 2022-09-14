# ActivitySim
# See full license in LICENSE.txt.
import logging

import pandas as pd

from activitysim.core import inject, pipeline, tracing
from activitysim.core.input import read_input_table

logger = logging.getLogger(__name__)

def read_disaggregate_accessibility(table_name):
    """
    Generic disaggregate accessibility table import function to recycle within table specific injection function calls.
    If '*destination model*_accessibilities' is in input_tables list, then read it in.
    """
    df = read_input_table(table_name, required=False)
    inject.add_table(table_name, df)
    return df


@inject.table()
def workplace_location_accessibility(table_name='workplace_location_accessibility'):
    """
    This allows loading of pre-computed accessibility table.
    """
    df = read_disaggregate_accessibility(table_name)
    return df

@inject.table()
def school_location_accessibility(table_name='school_location_accessibility'):
    """
    This allows loading of pre-computed accessibility table
    """
    df = read_disaggregate_accessibility(table_name)
    return df

@inject.table()
def non_mandatory_tour_destination_accessibility(table_name='non_mandatory_tour_destination_accessibility'):
    """
    This allows loading of pre-computed accessibility table
    """
    df = read_disaggregate_accessibility(table_name)
    return df

@inject.table()
def proto_persons(households, trace_hh_id):
    df = pd.DataFrame()
    # df = inject.get_table('persons')
    logger.info("loaded proto_persons %s" % (df.shape,))
    # replace table function with dataframe
    inject.add_table("proto_persons", df)
    pipeline.get_rn_generator().add_channel("proto_persons", df)
    tracing.register_traceable_table("proto_persons", df)
    return df


# another common merge for persons
@inject.table()
def proto_persons_merged(proto_persons, households, land_use):

    return inject.merge_tables(
        proto_persons.name, tables=[proto_persons, households, land_use]
    )


@inject.table()
def proto_households(trace_hh_id):

    df = pd.DataFrame()
    logger.info("loaded proto_households %s" % (df.shape,))
    # replace table function with dataframe
    inject.add_table("proto_households", df)
    pipeline.get_rn_generator().add_channel("proto_households", df)
    tracing.register_traceable_table("proto_households", df)
    if trace_hh_id:
        tracing.trace_df(df, "raw.proto_households", warn_if_empty=True)

    return df


@inject.table()
def proto_tours(trace_hh_id):
    df = pd.DataFrame()
    logger.info("loaded proto_tours %s" % (df.shape,))
    # replace table function with dataframe
    inject.add_table("proto_tours", df)
    pipeline.get_rn_generator().add_channel("proto_tours", df)
    tracing.register_traceable_table("proto_tours", df)
    if trace_hh_id:
        tracing.trace_df(df, "raw.proto_tours", warn_if_empty=True)

    return df

# this is a common merge so might as well define it once here and use it
# @inject.table()
# def proto_households_merged(proto_households, land_use):
#     return inject.merge_tables(
#         proto_households.name, tables=[proto_households, land_use]
#     )


inject.broadcast("proto_households", "proto_persons", cast_index=True, onto_on="household_id")

# this would be accessibility around the household location - be careful with
# this one as accessibility at some other location can also matter
inject.broadcast("workplace_location_accessibility", "proto_households", cast_index=True, onto_on="home_zone_id")
inject.broadcast("school_location_accessibility", "proto_households", cast_index=True, onto_on="home_zone_id")
inject.broadcast("non_mandatory_tour_destination_accessibility", "proto_households", cast_index=True, onto_on="home_zone_id")
