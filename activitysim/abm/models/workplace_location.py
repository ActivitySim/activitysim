# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import pandas as pd

from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import pipeline
from activitysim.core import simulate
from activitysim.core import inject

from activitysim.core.interaction_sample_simulate import interaction_sample_simulate
from activitysim.core.interaction_sample import interaction_sample

from activitysim.core.util import reindex

from .util import expressions
from .util.logsums import compute_logsums
from .util.expressions import skim_time_period_label

from .util import logsums as logsum

from .util.tour_destination import tour_destination_size_terms


"""
The workplace location model predicts the zones in which various people will
work.

for now we generate a workplace location for everyone -
presumably it will not get used in downstream models for everyone -
it should depend on CDAP and mandatory tour generation as to whether
it gets used
"""

logger = logging.getLogger(__name__)


@inject.injectable()
def workplace_location_sample_spec(configs_dir):
    return simulate.read_model_spec(configs_dir, 'workplace_location_sample.csv')


@inject.step()
def workplace_location_sample(persons_merged,
                              workplace_location_sample_spec,
                              skim_dict,
                              land_use, size_terms,
                              configs_dir, chunk_size, trace_hh_id):
    """
    build a table of workers * all zones in order to select a sample of alternative work locations.

    PERID,  dest_TAZ, rand,            pick_count
    23750,  14,       0.565502716034,  4
    23750,  16,       0.711135838871,  6
    ...
    23751,  12,       0.408038878552,  1
    23751,  14,       0.972732479292,  2
    """

    trace_label = 'workplace_location_sample'
    model_settings = config.read_model_settings(configs_dir, 'workplace_location.yaml')

    # FIXME - only choose workplace_location of workers? is this the right criteria?
    choosers = persons_merged.to_frame()
    choosers = choosers[choosers.is_worker]

    if choosers.shape[0] == 0:
        logger.info("Skipping %s: no workers" % trace_label)
        inject.add_table('workplace_location_sample', pd.DataFrame())
        return

    # FIXME - MEMORY HACK - only include columns actually used in spec
    chooser_columns = model_settings['SIMULATE_CHOOSER_COLUMNS']
    choosers = choosers[chooser_columns]

    alternatives = tour_destination_size_terms(land_use, size_terms, 'work')

    sample_size = model_settings["SAMPLE_SIZE"]
    alt_dest_col_name = model_settings["ALT_DEST_COL_NAME"]

    logger.info("Running workplace_location_sample with %d persons" % len(choosers))

    # create wrapper with keys for this lookup - in this case there is a TAZ in the choosers
    # and a TAZ in the alternatives which get merged during interaction
    # the skims will be available under the name "skims" for any @ expressions
    skims = skim_dict.wrap("TAZ", "TAZ_r")

    locals_d = {
        'skims': skims
    }
    constants = config.get_model_constants(model_settings)
    if constants is not None:
        locals_d.update(constants)

    choices = interaction_sample(
        choosers,
        alternatives,
        sample_size=sample_size,
        alt_col_name=alt_dest_col_name,
        spec=workplace_location_sample_spec,
        skims=skims,
        locals_d=locals_d,
        chunk_size=chunk_size,
        trace_label=trace_label)

    inject.add_table('workplace_location_sample', choices)


@inject.step()
def workplace_location_logsums(persons_merged,
                               land_use,
                               skim_dict, skim_stack,
                               workplace_location_sample,
                               configs_dir, chunk_size, trace_hh_id):
    """
    add logsum column to existing workplace_location_sample able

    logsum is calculated by running the mode_choice model for each sample (person, dest_taz) pair
    in workplace_location_sample, and computing the logsum of all the utilities

    +-------+--------------+----------------+------------+----------------+
    | PERID | dest_TAZ     | rand           | pick_count | logsum (added) |
    +=======+==============+================+============+================+
    | 23750 |  14          | 0.565502716034 | 4          |  1.85659498857 |
    +-------+--------------+----------------+------------+----------------+
    + 23750 | 16           | 0.711135838871 | 6          | 1.92315598631  |
    +-------+--------------+----------------+------------+----------------+
    + ...   |              |                |            |                |
    +-------+--------------+----------------+------------+----------------+
    | 23751 | 12           | 0.408038878552 | 1          | 2.40612135416  |
    +-------+--------------+----------------+------------+----------------+
    | 23751 | 14           | 0.972732479292 | 2          |  1.44009018355 |
    +-------+--------------+----------------+------------+----------------+
    """

    trace_label = 'workplace_location_logsums'

    location_sample = workplace_location_sample.to_frame()
    if location_sample.shape[0] == 0:
        tracing.no_results(trace_label)
        return

    model_settings = config.read_model_settings(configs_dir, 'workplace_location.yaml')
    logsum_settings = config.read_model_settings(configs_dir, 'logsum.yaml')

    persons_merged = persons_merged.to_frame()
    # FIXME - MEMORY HACK - only include columns actually used in spec
    persons_merged = logsum.filter_chooser_columns(persons_merged, logsum_settings, model_settings)

    logger.info("Running workplace_location_logsums with %s rows" % len(location_sample))

    logsum_spec = logsum.get_logsum_spec(logsum_settings, selector='nontour', segment='work',
                                         configs_dir=configs_dir, want_tracing=trace_hh_id)

    choosers = pd.merge(location_sample,
                        persons_merged,
                        left_index=True,
                        right_index=True,
                        how="left")

    logsums = logsum.compute_logsums(
        choosers, logsum_spec,
        logsum_settings, model_settings,
        skim_dict, skim_stack,
        chunk_size, trace_hh_id,
        trace_label)

    # "add_column series should have an index matching the table to which it is being added"
    # when the index has duplicates, however, in the special case that the series index exactly
    # matches the table index, then the series value order is preserved
    # logsums now does, since workplace_location_sample was on left side of merge de-dup merge
    inject.add_column('workplace_location_sample', 'mode_choice_logsum', logsums)


@inject.injectable()
def workplace_location_spec(configs_dir):
    return simulate.read_model_spec(configs_dir, 'workplace_location.csv')


@inject.step()
def workplace_location_simulate(persons_merged, persons,
                                workplace_location_sample,
                                workplace_location_spec,
                                skim_dict,
                                land_use, size_terms,
                                configs_dir, chunk_size, trace_hh_id):
    """
    Workplace location model on workplace_location_sample annotated with mode_choice logsum
    to select a work_taz from sample alternatives
    """

    trace_label = 'workplace_location_simulate'
    model_settings = config.read_model_settings(configs_dir, 'workplace_location.yaml')
    NO_WORKPLACE_TAZ = -1

    location_sample = workplace_location_sample.to_frame()
    persons = persons.to_frame()

    if location_sample.shape[0] > 0:

        choosers = persons_merged.to_frame()
        choosers = choosers[choosers.is_worker]

        # FIXME - MEMORY HACK - only include columns actually used in spec
        chooser_columns = model_settings['SIMULATE_CHOOSER_COLUMNS']
        choosers = choosers[chooser_columns]

        alt_dest_col_name = model_settings["ALT_DEST_COL_NAME"]

        # alternatives are pre-sampled and annotated with logsums and pick_count
        # but we have to merge additional alt columns into alt sample list
        location_sample = workplace_location_sample.to_frame()
        destination_size_terms = tour_destination_size_terms(land_use, size_terms, 'work')

        alternatives = \
            pd.merge(location_sample, destination_size_terms,
                     left_on=alt_dest_col_name, right_index=True, how="left")

        logger.info("Running workplace_location_simulate with %d persons" % len(choosers))

        # create wrapper with keys for this lookup - in this case there is a TAZ in the choosers
        # and a TAZ in the alternatives which get merged during interaction
        # the skims will be available under the name "skims" for any @ expressions
        skims = skim_dict.wrap("TAZ", alt_dest_col_name)

        locals_d = {
            'skims': skims,
        }
        constants = config.get_model_constants(model_settings)
        if constants is not None:
            locals_d.update(constants)

        choices = interaction_sample_simulate(
            choosers,
            alternatives,
            spec=workplace_location_spec,
            choice_column=alt_dest_col_name,
            skims=skims,
            locals_d=locals_d,
            chunk_size=chunk_size,
            trace_label=trace_label,
            trace_choice_name='workplace_location')

        persons['workplace_taz'] = \
            choices.reindex(persons.index).fillna(NO_WORKPLACE_TAZ).astype(int)

    else:

        # no workers (but we still want to annotate persons)
        persons['workplace_taz'] = NO_WORKPLACE_TAZ

    expressions.assign_columns(
        df=persons,
        model_settings=model_settings.get('annotate_persons'),
        trace_label=tracing.extend_trace_label(trace_label, 'annotate_persons'))

    pipeline.replace_table("persons", persons)

    pipeline.drop_table('workplace_location_sample')

    tracing.print_summary('workplace_taz', persons.workplace_taz, describe=True)

    if trace_hh_id:
        tracing.trace_df(persons,
                         label="workplace_location",
                         warn_if_empty=True)
