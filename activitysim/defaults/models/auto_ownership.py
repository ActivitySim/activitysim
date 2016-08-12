# ActivitySim
# See full license in LICENSE.txt.

import os
import orca

from activitysim import activitysim as asim
from activitysim import tracing
from .util.misc import add_dependent_columns


@orca.injectable()
def auto_ownership_spec(configs_dir):
    f = os.path.join(configs_dir, 'configs', "auto_ownership.csv")
    return asim.read_model_spec(f).fillna(0)


@orca.step()
def auto_ownership_simulate(set_random_seed, households_merged,
                            auto_ownership_spec,
                            trace_hh_id):
    """
    Auto ownership is a standard model which predicts how many cars a household
    with given characteristics owns
    """

    tracing.info(__name__,
                 "Running auto_ownership_simulate with %d households" % len(households_merged))

    choices, _ = asim.simple_simulate(
        choosers=households_merged.to_frame(),
        spec=auto_ownership_spec,
        trace_label=trace_hh_id and 'auto_ownership',
        trace_choice_name='auto_ownership')

    tracing.print_summary('auto_ownership', choices, value_counts=True)

    orca.add_column("households", "auto_ownership", choices)

    add_dependent_columns("households", "households_autoown")

    if trace_hh_id:
        trace_columns = ['auto_ownership'] + orca.get_table('households_autoown').columns
        tracing.trace_df(orca.get_table('households').to_frame(),
                         label="auto_ownership",
                         columns=trace_columns,
                         warn=True)
