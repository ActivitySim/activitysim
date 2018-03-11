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

from .util import expressions


logger = logging.getLogger(__name__)


@inject.step()
def initialize(store, configs_dir):
    """
    Because random seed is set differently for each step, the sampling of households depends
    on which step they are initially loaded in so we force them to load here and they get
    stored to the pipeline,
    """

    trace_label = 'initialize'

    model_settings = config.read_model_settings(configs_dir, 'initialize.yaml')

    t0 = tracing.print_elapsed_time()

    annotate_tables = model_settings.get('annotate_tables')
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

        t0 = tracing.print_elapsed_time("annotate %s" % tablename, t0, debug=True)

    inject.get_table('person_windows').to_frame()
    t0 = tracing.print_elapsed_time("preload person_windows", t0, debug=True)


@inject.injectable(cache=True)
def preload_injectables():
    """
    preload bulky injectables up front - stuff that isn't inserted into eh pipeline
    """

    logger.info("preload_injectables")

    t0 = tracing.print_elapsed_time()

    if inject.get_injectable('skim_dict', None) is not None:
        t0 = tracing.print_elapsed_time("preload skim_dict", t0, debug=True)

    if inject.get_injectable('skim_stack', None) is not None:
        t0 = tracing.print_elapsed_time("preload skim_stack", t0, debug=True)
