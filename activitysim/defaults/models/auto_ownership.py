# ActivitySim
# See full license in LICENSE.txt.

import os

import orca

from activitysim import activitysim as asim
from activitysim import tracing
from .util.misc import add_dependent_columns
from .util.misc import read_model_settings, get_logit_model_settings, get_model_constants


@orca.injectable()
def auto_ownership_spec(configs_dir):
    f = os.path.join(configs_dir, 'auto_ownership.csv')
    return asim.read_model_spec(f).fillna(0)


@orca.injectable()
def auto_ownership_settings(configs_dir):
    return read_model_settings(configs_dir, 'auto_ownership.yaml')


@orca.step()
def auto_ownership_simulate(set_random_seed, households_merged,
                            auto_ownership_spec,
                            auto_ownership_settings,
                            trace_hh_id):
    """
    Auto ownership is a standard model which predicts how many cars a household
    with given characteristics owns
    """

    tracing.info(__name__,
                 "Running auto_ownership_simulate with %d households" % len(households_merged))

    nest_spec = get_logit_model_settings(auto_ownership_settings)
    constants = get_model_constants(auto_ownership_settings)

    choices = asim.simple_simulate(
        choosers=households_merged.to_frame(),
        spec=auto_ownership_spec,
        nest_spec=nest_spec,
        locals_d=constants,
        trace_label=trace_hh_id and 'auto_ownership',
        trace_choice_name='auto_ownership')

    tracing.print_summary('auto_ownership', choices, value_counts=True)

    orca.add_column('households', 'auto_ownership', choices)

    add_dependent_columns('households', 'households_autoown')

    if trace_hh_id:
        trace_columns = ['auto_ownership'] + orca.get_table('households_autoown').columns
        tracing.trace_df(orca.get_table('households').to_frame(),
                         label='auto_ownership',
                         columns=trace_columns,
                         warn=True)
