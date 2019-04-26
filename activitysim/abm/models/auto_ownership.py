# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402

import logging

from activitysim.core import simulate
from activitysim.core import tracing
from activitysim.core import pipeline
from activitysim.core import config
from activitysim.core import inject

logger = logging.getLogger(__name__)


@inject.step()
def auto_ownership_simulate(households,
                            households_merged,
                            chunk_size,
                            trace_hh_id):
    """
    Auto ownership is a standard model which predicts how many cars a household
    with given characteristics owns
    """
    trace_label = 'auto_ownership_simulate'
    model_settings = config.read_model_settings('auto_ownership.yaml')

    logger.info("Running %s with %d households", trace_label, len(households_merged))

    model_spec = simulate.read_model_spec(file_name='auto_ownership.csv')

    nest_spec = config.get_logit_model_settings(model_settings)
    constants = config.get_model_constants(model_settings)

    choices = simulate.simple_simulate(
        choosers=households_merged.to_frame(),
        spec=model_spec,
        nest_spec=nest_spec,
        locals_d=constants,
        chunk_size=chunk_size,
        trace_label=trace_label,
        trace_choice_name='auto_ownership')

    households = households.to_frame()

    # no need to reindex as we used all households
    households['auto_ownership'] = choices

    pipeline.replace_table("households", households)

    tracing.print_summary('auto_ownership', households.auto_ownership, value_counts=True)

    if trace_hh_id:
        tracing.trace_df(households,
                         label='auto_ownership',
                         warn_if_empty=True)
