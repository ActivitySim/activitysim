# ActivitySim
# See full license in LICENSE.txt.

import logging

import numpy as np
import pandas as pd

from activitysim.core import assign
from activitysim.core import inject
from activitysim.core import tracing
from activitysim.core import config


logger = logging.getLogger('activitysim')


@inject.injectable()
def best_transit_path_spec():
    return assign.read_assignment_spec(config.config_file_path('best_transit_path.csv'))


VECTOR_TEST_SIZE = 100000
VECTOR_TEST_SIZE = 1014699


@inject.step()
def best_transit_path(set_random_seed,
                      network_los,
                      best_transit_path_spec):

    model_settings = config.read_model_settings('best_transit_path.yaml')

    logger.info("best_transit_path VECTOR_TEST_SIZE %s", VECTOR_TEST_SIZE)

    omaz = network_los.maz_df.sample(VECTOR_TEST_SIZE, replace=True).index
    dmaz = network_los.maz_df.sample(VECTOR_TEST_SIZE, replace=True).index
    tod = np.random.choice(['AM', 'PM'], VECTOR_TEST_SIZE)
    od_df = pd.DataFrame({'omaz': omaz, 'dmaz': dmaz, 'tod': tod})

    trace_od = (od_df.omaz[0], od_df.dmaz[0])
    logger.info("trace_od omaz %s dmaz %s" % trace_od)

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

    logger.info("len od_df %s", len(od_df.index))
    logger.info("len atap_btap_df %s", len(atap_btap_df.index))
    logger.info("avg explosion %s", (len(atap_btap_df.index) / (1.0 * len(od_df.index))))

    if trace_od:
        trace_orig, trace_dest = trace_od
        trace_oabd_rows = (atap_btap_df.omaz == trace_orig) & (atap_btap_df.dmaz == trace_dest)
    else:
        trace_oabd_rows = None

    constants = config.get_model_constants(model_settings)
    locals_d = {
        'np': np,
        'network_los': network_los
    }
    if constants is not None:
        locals_d.update(constants)

    results, trace_results, trace_assigned_locals \
        = assign.assign_variables(best_transit_path_spec, atap_btap_df, locals_d,
                                  trace_rows=trace_oabd_rows)

    # copy results
    for column in results.columns:
        atap_btap_df[column] = results[column]

    # drop rows if no utility
    n = len(atap_btap_df.index)
    atap_btap_df = atap_btap_df.dropna(subset=['utility'])

    logger.info("Dropped %s of %s rows with null utility", n - len(atap_btap_df.index), n)

    # choose max utility
    atap_btap_df = atap_btap_df.sort_values(by='utility').groupby('idx').tail(1)

    if trace_od:

        if not trace_oabd_rows.any():
            logger.warning("trace_od not found origin = %s, dest = %s", trace_orig, trace_dest)
        else:

            tracing.trace_df(atap_btap_df,
                             label='best_transit_path',
                             slicer='NONE',
                             transpose=False)

            tracing.trace_df(trace_results,
                             label='trace_best_transit_path',
                             slicer='NONE',
                             transpose=False)

            if trace_assigned_locals:
                tracing.write_csv(trace_assigned_locals, file_name="trace_best_transit_path_locals")
