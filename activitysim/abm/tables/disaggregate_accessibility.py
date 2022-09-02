# ActivitySim
# See full license in LICENSE.txt.
import logging

import pandas as pd

from activitysim.core import inject
from activitysim.core.input import read_input_table

logger = logging.getLogger(__name__)
#
# # households = inject.get_table("households").to_frame()
# # assert not households._is_view
# # chunk.log_df(trace_label, "households", households)
# # del households
# # chunk.log_df(trace_label, "households", None)
# #
# # persons = inject.get_table("persons").to_frame()
# # assert not persons._is_view
# # chunk.log_df(trace_label, "persons", persons)
# # del persons
# # chunk.log_df(trace_label, "persons", None)
#
# persons_merged = inject.get_table("persons_merged").to_frame()
# assert not persons_merged._is_view
# chunk.log_df(trace_label, "persons_merged", persons_merged)
# del persons_merged
# chunk.log_df(trace_label, "persons_merged", None)
#
# model_settings = config.read_model_settings(
#     "disaggregate_accessibility.yaml", mandatory=True
# )
# initialize.annotate_tables(model_settings, trace_label)
#
#
# # Merge

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

# @inject.table()
# def disaggregate_accessibility():
#     """
#     If '*destination model*_accessibilities' is in input_tables list, then read it in.
#     This allows loading of pre-computed accessibility table.
#     """
#
#     def import_table(table_name):
#         df = read_input_table(table_name, required=False)
#
#         # replace table function with dataframe
#         inject.add_table(table_name, df)
#
#         return df
#
#     accessibility_tables = ['workplace_location_accessibilities',
#                             'school_location_accessibilities',
#                             'non_mandatory_tour_destination_accessibilities']
#
#     accessibilities = {k: import_table(k) for k in accessibility_tables}
#
#     return accessibilities
