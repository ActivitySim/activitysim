# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import numpy as np
import pandas as pd

import orca

from activitysim import activitysim as asim

from activitysim import asim_eval as asim_eval

from activitysim import tracing
from activitysim.defaults.models.util.misc import read_model_settings, get_logit_model_settings, get_model_constants


logger = logging.getLogger(__name__)


@orca.injectable()
def best_transit_path_spec(configs_dir):
    f = os.path.join(configs_dir, 'best_transit_path.csv')
    return asim_eval.read_assignment_spec(f)


@orca.injectable()
def best_transit_path_settings(configs_dir):
    return read_model_settings(configs_dir, 'best_transit_path.yaml')


VECTOR_TEST_SIZE = 10000


@orca.step()
def best_transit_path(set_random_seed,
                      network_los,
                      best_transit_path_spec,
                      best_transit_path_settings):



    omaz = network_los.maz_df.sample(VECTOR_TEST_SIZE, replace=True).index
    dmaz = network_los.maz_df.sample(VECTOR_TEST_SIZE, replace=True).index
    tod = np.random.choice(['AM', 'PM'], VECTOR_TEST_SIZE)
    od_df = pd.DataFrame({'omaz': omaz, 'dmaz': dmaz, 'tod': tod})

    trace_od = ( od_df.omaz[0], od_df.dmaz[0])
    print "\ntrace_od\n", trace_od

    # build exploded atap_btap_df

    # FIXME - pathological knowledge about mode - should be parameterized
    # filter out rows with no drive time omaz-btap or no walk time from dmaz-atap
    atap_btap_df = network_los.get_tappairs_mazpairs(od_df.omaz, od_df.dmaz,
                                                     ofilter='drive_time',
                                                     dfilter='walk_alightingActual')

    # add in tod column
    atap_btap_df = atap_btap_df.merge(
        right=od_df[['tod']],
        left_on='idx',
        right_index=True,
        how='left'
    )

    print "\nlen od_df", len(od_df.index)
    print "\nlen atap_btap_df", len(atap_btap_df.index)
    print "\navg explosion", len(atap_btap_df.index) / (1.0 * len(od_df.index))


    if trace_od:
        trace_orig, trace_dest = trace_od
        trace_oabd_rows = (atap_btap_df.omaz == trace_orig) & (atap_btap_df.dmaz == trace_dest)
    else:
        trace_oabd_rows = None

    constants = get_model_constants(best_transit_path_settings)
    locals_d = {
        'network_los': network_los
    }
    if constants is not None:
        locals_d.update(constants)

    results, trace_results = asim_eval.assign_variables(best_transit_path_spec, atap_btap_df, locals_d, trace_rows=trace_oabd_rows)

    # tracing.trace_df(results,
    #                  label='results',
    #                  slicer='NONE',
    #                  transpose=False)

    #print "\nresults\n", results

    # copy results
    for column in results.columns:
        atap_btap_df[column] = results[column]

    # drop rows if no utility
    n = len(atap_btap_df.index)
    atap_btap_df = atap_btap_df.dropna(subset=['utility'])

    print "\nDropped %s of %s rows with null utility" % (n - len(atap_btap_df.index), n)

    # choose max utility
    atap_btap_df = atap_btap_df.sort_values(by='utility').groupby('idx').tail(1)

    print "\natap_btap_df\n", atap_btap_df

    if trace_od:

        if not trace_oabd_rows.any():
            logger.warn("trace_od not found origin = %s, dest = %s" % (trace_orig, trace_dest))
        else:

            tracing.trace_df(atap_btap_df,
                             label='best_transit_path',
                             slicer='NONE',
                             transpose=False)

            tracing.trace_df(trace_results,
                             label='trace_best_transit_path',
                             slicer='NONE',
                             transpose=False)






