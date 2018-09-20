# ActivitySim
# See full license in LICENSE.txt.

import logging
import os

import pandas as pd
import numpy as np

from activitysim.core import assign
from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import inject
from activitysim.core import pipeline

from activitysim.core.steps.output import write_data_dictionary
from activitysim.core.steps.output import write_tables
from activitysim.core.steps.output import track_skim_usage

from .util import expressions


logger = logging.getLogger(__name__)


def annotate_tables(model_settings, trace_label):

    annotate_tables = model_settings.get('annotate_tables', [])

    if not annotate_tables:
        logger.warn("annotate_tables setting is empty - nothing to do!")

    t0 = tracing.print_elapsed_time()

    for table_info in annotate_tables:

        tablename = table_info['tablename']
        df = inject.get_table(tablename).to_frame()

        # - rename columns
        column_map = table_info.get('column_map', None)
        if column_map:
            logger.info("renaming %s columns %s" % (tablename, column_map,))
            df.rename(columns=column_map, inplace=True)

        # - annotate
        annotate = table_info.get('annotate', None)
        if annotate:
            logger.info("annotated %s SPEC %s" % (tablename, annotate['SPEC'],))
            expressions.assign_columns(
                df=df,
                model_settings=annotate,
                trace_label=tracing.extend_trace_label(trace_label, 'annotate_%s' % tablename))

        # - write table to pipeline
        pipeline.replace_table(tablename, df)


@inject.step()
def initialize_landuse():

    trace_label = 'initialize_landuse'

    model_settings = config.read_model_settings('initialize_landuse.yaml', mandatory=True)

    annotate_tables(model_settings, trace_label)

    # create accessibility
    land_use = pipeline.get_table('land_use')

    accessibility_df = pd.DataFrame(index=land_use.index)

    # - write table to pipeline
    pipeline.replace_table("accessibility", accessibility_df)


@inject.step()
def initialize_households():

    trace_label = 'initialize_households'

    model_settings = config.read_model_settings('initialize_households.yaml', mandatory=True)

    annotate_tables(model_settings, trace_label)

    t0 = tracing.print_elapsed_time()
    inject.get_table('person_windows').to_frame()
    t0 = tracing.print_elapsed_time("preload person_windows", t0, debug=True)


@inject.injectable(cache=True)
def preload_injectables():
    """
    preload bulky injectables up front - stuff that isn't inserted into the pipeline
    """

    logger.info("preload_injectables")

    inject.add_step('track_skim_usage', track_skim_usage)
    inject.add_step('write_data_dictionary', write_data_dictionary)
    inject.add_step('write_tables', write_tables)

    t0 = tracing.print_elapsed_time()

    if inject.get_injectable('skim_dict', None) is not None:
        t0 = tracing.print_elapsed_time("preload skim_dict", t0, debug=True)

    if inject.get_injectable('skim_stack', None) is not None:
        t0 = tracing.print_elapsed_time("preload skim_stack", t0, debug=True)
