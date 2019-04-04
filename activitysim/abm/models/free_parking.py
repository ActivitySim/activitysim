# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402

from future.utils import iteritems

import logging

import pandas as pd

from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import pipeline
from activitysim.core import simulate
from activitysim.core import inject
from activitysim.core.mem import force_garbage_collect

from activitysim.core.interaction_sample_simulate import interaction_sample_simulate
from activitysim.core.interaction_sample import interaction_sample

from .util import expressions

logger = logging.getLogger(__name__)


@inject.step()
def free_parking(
        persons_merged, persons, households,
        skim_dict, skim_stack,
        chunk_size, trace_hh_id, locutor):
    """

    """

    trace_label = 'free_parking'
    model_settings = config.read_model_settings('free_parking.yaml')

    choosers = persons_merged.to_frame()
    choosers = choosers[choosers.workplace_taz > -1]

    logger.info("Running %s with %d persons", trace_label, len(choosers))

    constants = config.get_model_constants(model_settings)

    # - preprocessor
    preprocessor_settings = model_settings.get('preprocessor', None)
    if preprocessor_settings:

        locals_d = {}
        if constants is not None:
            locals_d.update(constants)

        expressions.assign_columns(
            df=choosers,
            model_settings=preprocessor_settings,
            locals_dict=locals_d,
            trace_label=trace_label)

    model_spec = simulate.read_model_spec(file_name='free_parking.csv')
    nest_spec = config.get_logit_model_settings(model_settings)

    choices = simulate.simple_simulate(
        choosers=choosers,
        spec=model_spec,
        nest_spec=nest_spec,
        locals_d=constants,
        chunk_size=chunk_size,
        trace_label=trace_label,
        trace_choice_name='free_parking_at_work')

    persons = persons.to_frame()

    # no need to reindex as we used all households
    free_parking_alt = model_settings['FREE_PARKING_ALT']
    choices = (choices == free_parking_alt)
    persons['free_parking_at_work'] = choices.reindex(persons.index).fillna(0).astype(bool)

    pipeline.replace_table("persons", persons)

    tracing.print_summary('free_parking', persons.free_parking_at_work, value_counts=True)

    if trace_hh_id:
        tracing.trace_df(persons,
                         label=trace_label,
                         warn_if_empty=True)
