# ActivitySim
# See full license in LICENSE.txt.

import logging

import openmatrix as omx
import pandas as pd
import numpy as np

from activitysim.core import config
from activitysim.core import inject
from activitysim.core import pipeline

from .util import expressions
from .util.expressions import skim_time_period_label

logger = logging.getLogger(__name__)


@inject.step()
def write_trip_matrices(trips, skim_dict, skim_stack):
    """
    Write trip matrices step.

    Adds boolean columns to local trips table via annotation expressions,
    then aggregates trip counts and writes OD matrices to OMX.  Save annotated
    trips table to pipeline if desired.
    """

    model_settings = config.read_model_settings('write_trip_matrices.yaml')
    trips_df = annotate_trips(trips, skim_dict, skim_stack, model_settings)

    if bool(model_settings.get('SAVE_TRIPS_TABLE')):
        pipeline.replace_table('trips', trips_df)

    logger.info('Aggregating trips...')
    aggregate_trips = trips_df.groupby(['origin', 'destination'], sort=False).sum()

    # use the average household weight for all trips in the origin destination pair
    hh_weight_col = model_settings.get('HH_EXPANSION_WEIGHT_COL')
    aggregate_weight = trips_df[['origin', 'destination', hh_weight_col]].groupby(['origin', 'destination'],
                                                                                  sort=False).mean()
    aggregate_trips[hh_weight_col] = aggregate_weight[hh_weight_col]

    logger.info('Finished.')

    orig_vals = aggregate_trips.index.get_level_values('origin')
    dest_vals = aggregate_trips.index.get_level_values('destination')

    zone_index = pipeline.get_table('land_use').index
    assert all(zone in zone_index for zone in orig_vals)
    assert all(zone in zone_index for zone in dest_vals)

    _, orig_index = zone_index.reindex(orig_vals)
    _, dest_index = zone_index.reindex(dest_vals)

    write_matrices(aggregate_trips, zone_index, orig_index, dest_index, model_settings)


def annotate_trips(trips, skim_dict, skim_stack, model_settings):
    """
    Add columns to local trips table. The annotator has
    access to the origin/destination skims and everything
    defined in the model settings CONSTANTS.

    Pipeline tables can also be accessed by listing them under
    TABLES in the preprocessor settings.
    """

    trips_df = trips.to_frame()

    trace_label = 'trip_matrices'

    # setup skim keys
    assert ('trip_period' not in trips_df)
    trips_df['trip_period'] = skim_time_period_label(trips_df.depart)
    od_skim_wrapper = skim_dict.wrap('origin', 'destination')
    odt_skim_stack_wrapper = skim_stack.wrap(left_key='origin', right_key='destination',
                                             skim_key='trip_period')
    skims = {
        'od_skims': od_skim_wrapper,
        "odt_skims": odt_skim_stack_wrapper
    }

    locals_dict = {}
    constants = config.get_model_constants(model_settings)
    if constants is not None:
        locals_dict.update(constants)

    expressions.annotate_preprocessors(
        trips_df, locals_dict, skims,
        model_settings, trace_label)

    # Data will be expanded by an expansion weight column from
    # the households pipeline table, if specified in the model settings.
    hh_weight_col = model_settings.get('HH_EXPANSION_WEIGHT_COL')

    if hh_weight_col and hh_weight_col not in trips_df:
        logger.info("adding '%s' from households to trips table" % hh_weight_col)
        household_weights = pipeline.get_table('households')[hh_weight_col]
        trips_df[hh_weight_col] = trips_df.household_id.map(household_weights)

    return trips_df


def write_matrices(aggregate_trips, zone_index, orig_index, dest_index, model_settings):
    """
    Write aggregated trips to OMX format.

    The MATRICES setting lists the new OMX files to write.
    Each file can contain any number of 'tables', each specified by a
    table key ('name') and a trips table column ('data_field') to use
    for aggregated counts.

    Any data type may be used for columns added in the annotation phase,
    but the table 'data_field's must be summable types: ints, floats, bools.
    """

    matrix_settings = model_settings.get('MATRICES')

    if not matrix_settings:
        logger.error('Missing MATRICES setting in write_trip_matrices.yaml')

    for matrix in matrix_settings:
        filename = matrix.get('file_name')
        filepath = config.output_file_path(filename)
        logger.info('opening %s' % filepath)
        file = omx.open_file(filepath, 'w')  # possibly overwrite existing file
        table_settings = matrix.get('tables')

        for table in table_settings:
            table_name = table.get('name')
            col = table.get('data_field')

            if col not in aggregate_trips:
                logger.error(f'missing {col} column in aggregate_trips DataFrame')
                return

            hh_weight_col = model_settings.get('HH_EXPANSION_WEIGHT_COL')
            if hh_weight_col:
                aggregate_trips[col] = aggregate_trips[col] / aggregate_trips[hh_weight_col]

            data = np.zeros((len(zone_index), len(zone_index)))
            data[orig_index, dest_index] = aggregate_trips[col]
            logger.info('writing %s' % table_name)
            file[table_name] = data  # write to file

        # include the index-to-zone map in the file
        logger.info('adding %s mapping for %s zones to %s' %
                    (zone_index.name, zone_index.size, filename))
        file.create_mapping(zone_index.name, zone_index.to_numpy())

        logger.info('closing %s' % filepath)
        file.close()
