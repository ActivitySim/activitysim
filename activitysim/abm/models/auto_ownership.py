# ActivitySim
# See full license in LICENSE.txt.

import logging

from activitysim.core import simulate
from activitysim.core import tracing
from activitysim.core import pipeline
from activitysim.core import config
from activitysim.core import inject

from .util import expressions

logger = logging.getLogger(__name__)


@inject.injectable()
def auto_ownership_spec(configs_dir):
    return simulate.read_model_spec(configs_dir, 'auto_ownership.csv')


@inject.injectable()
def auto_ownership_settings(configs_dir):
    return config.read_model_settings(configs_dir, 'auto_ownership.yaml')


@inject.step()
def auto_ownership_simulate(households,
                            households_merged,
                            auto_ownership_spec,
                            auto_ownership_settings,
                            configs_dir,
                            chunk_size,
                            trace_hh_id):
    """
    Auto ownership is a standard model which predicts how many cars a household
    with given characteristics owns
    """
    trace_label = 'auto_ownership_simulate'

    logger.info("Running auto_ownership_simulate with %d households" % len(households_merged))

    nest_spec = config.get_logit_model_settings(auto_ownership_settings)
    constants = config.get_model_constants(auto_ownership_settings)

    choices = simulate.simple_simulate(
        choosers=households_merged.to_frame(),
        spec=auto_ownership_spec,
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
