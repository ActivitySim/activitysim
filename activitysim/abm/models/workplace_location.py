# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402

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
from .util import logsums as logsum

from activitysim.abm.tables import shadow_pricing

"""
The workplace location model predicts the zones in which various people will
work.

for now we generate a workplace location for everyone -
presumably it will not get used in downstream models for everyone -
it should depend on CDAP and mandatory tour generation as to whether
it gets used
"""

logger = logging.getLogger(__name__)


def run_workplace_location_sample(
        persons_merged,
        skim_dict,
        dest_size_terms,
        chunk_size, trace_hh_id):
    """
    build a table of workers * all zones in order to select a sample of alternative work locations.

    person_id,  dest_TAZ, rand,            pick_count
    23750,      14,       0.565502716034,  4
    23750,      16,       0.711135838871,  6
    ...
    23751,      12,       0.408038878552,  1
    23751,      14,       0.972732479292,  2
    """

    trace_label = 'workplace_location_sample'
    model_settings = config.read_model_settings('workplace_location.yaml')
    model_spec = simulate.read_model_spec(file_name='workplace_location_sample.csv')

    choosers = persons_merged

    if choosers.empty:
        logger.info("Skipping %s: no workers" % trace_label)
        choices = pd.DataFrame()
        return choices

    # FIXME - MEMORY HACK - only include columns actually used in spec
    chooser_columns = model_settings['SIMULATE_CHOOSER_COLUMNS']
    choosers = choosers[chooser_columns]

    alternatives = dest_size_terms

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
        spec=model_spec,
        skims=skims,
        locals_d=locals_d,
        chunk_size=chunk_size,
        trace_label=trace_label)

    return choices


def run_workplace_location_logsums(
        persons_merged_df,
        skim_dict, skim_stack,
        location_sample_df,
        chunk_size, trace_hh_id):
    """
    add logsum column to existing workplace_location_sample able

    logsum is calculated by running the mode_choice model for each sample (person, dest_taz) pair
    in workplace_location_sample, and computing the logsum of all the utilities

    +-----------+--------------+----------------+------------+----------------+
    | PERID     | dest_TAZ     | rand           | pick_count | logsum (added) |
    +===========+==============+================+============+================+
    | 23750     |  14          | 0.565502716034 | 4          |  1.85659498857 |
    +-----------+--------------+----------------+------------+----------------+
    + 23750     | 16           | 0.711135838871 | 6          | 1.92315598631  |
    +-----------+--------------+----------------+------------+----------------+
    + ...       |              |                |            |                |
    +-----------+--------------+----------------+------------+----------------+
    | 23751     | 12           | 0.408038878552 | 1          | 2.40612135416  |
    +-----------+--------------+----------------+------------+----------------+
    | 23751     | 14           | 0.972732479292 | 2          |  1.44009018355 |
    +-----------+--------------+----------------+------------+----------------+
    """

    trace_label = 'workplace_location_logsums'

    if location_sample_df.empty:
        tracing.no_results(trace_label)
        return location_sample_df

    model_settings = config.read_model_settings('workplace_location.yaml')
    logsum_settings = config.read_model_settings(model_settings['LOGSUM_SETTINGS'])

    # FIXME - MEMORY HACK - only include columns actually used in spec
    persons_merged_df = \
        logsum.filter_chooser_columns(persons_merged_df, logsum_settings, model_settings)

    logger.info("Running workplace_location_logsums with %s rows" % len(location_sample_df))

    choosers = pd.merge(location_sample_df,
                        persons_merged_df,
                        left_index=True,
                        right_index=True,
                        how="left")

    tour_purpose = 'work'
    logsums = logsum.compute_logsums(
        choosers,
        tour_purpose,
        logsum_settings, model_settings,
        skim_dict, skim_stack,
        chunk_size, trace_hh_id,
        trace_label)

    # "add_column series should have an index matching the table to which it is being added"
    # when the index has duplicates, however, in the special case that the series index exactly
    # matches the table index, then the series value order is preserved
    # logsums now does, since workplace_location_sample was on left side of merge de-dup merge
    location_sample_df['mode_choice_logsum'] = logsums

    return location_sample_df


def run_workplace_location_simulate(
        persons_merged,
        location_sample_df,
        skim_dict,
        dest_size_terms,
        chunk_size, trace_hh_id):
    """
    Workplace location model on workplace_location_sample annotated with mode_choice logsum
    to select a work_taz from sample alternatives
    """

    trace_label = 'workplace_location_simulate'
    model_settings = config.read_model_settings('workplace_location.yaml')
    model_spec = simulate.read_model_spec(file_name='workplace_location.csv')

    if location_sample_df.empty:
        logger.info("%s no workers" % trace_label)
        choices = pd.Series()
    else:

        choosers = persons_merged

        # FIXME - MEMORY HACK - only include columns actually used in spec
        chooser_columns = model_settings['SIMULATE_CHOOSER_COLUMNS']
        choosers = choosers[chooser_columns]

        alt_dest_col_name = model_settings["ALT_DEST_COL_NAME"]

        # alternatives are pre-sampled and annotated with logsums and pick_count
        # but we have to merge additional alt columns into alt sample list

        alternatives = \
            pd.merge(location_sample_df, dest_size_terms,
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
            spec=model_spec,
            choice_column=alt_dest_col_name,
            skims=skims,
            locals_d=locals_d,
            chunk_size=chunk_size,
            trace_label=trace_label,
            trace_choice_name='workplace_location')

    return choices


def run_workplace_location(
        persons_merged_df,
        skim_dict, skim_stack,
        dest_size_terms,
        model_settings,
        chunk_size, trace_hh_id, trace_label
        ):

    # - workplace_location_sample
    location_sample_df = \
        run_workplace_location_sample(
            persons_merged_df,
            skim_dict,
            dest_size_terms,
            chunk_size,
            trace_hh_id)

    # - workplace_location_logsums
    location_sample_df = \
        run_workplace_location_logsums(
            persons_merged_df,
            skim_dict, skim_stack,
            location_sample_df,
            chunk_size,
            trace_hh_id)

    # - school_location_simulate
    choices = \
        run_workplace_location_simulate(
            persons_merged_df,
            location_sample_df,
            skim_dict,
            dest_size_terms,
            chunk_size,
            trace_hh_id)

    return choices


@inject.step()
def workplace_location(
        persons_merged, persons,
        skim_dict, skim_stack,
        chunk_size, trace_hh_id, locutor):

    trace_label = 'workplace_location'
    model_settings = config.read_model_settings('workplace_location.yaml')

    chooser_segment_column = model_settings['CHOOSER_SEGMENT_COLUMN']

    persons_merged_df = persons_merged.to_frame()

    # presumably is_worker or something similar
    persons_merged_df = persons_merged_df[persons_merged[model_settings['CHOOSER_FILTER_COLUMN']]]

    spc = shadow_pricing.load_shadow_price_calculator(model_settings)
    max_iterations = spc.max_iterations

    logging.debug("%s max_iterations: %s" % (trace_label, max_iterations))

    choices = None
    for iteration in range(max_iterations):

        if iteration > 0:
            spc.update_shadow_prices()

        choices = run_workplace_location(
            persons_merged_df,
            skim_dict, skim_stack,
            spc.shadow_price_adjusted_predicted_size(),
            model_settings,
            chunk_size, trace_hh_id,
            trace_label=tracing.extend_trace_label(trace_label, 'i%s' % iteration))

        force_garbage_collect()

        choices_df = choices.to_frame('dest_choice')
        choices_df['segment_id'] = \
            persons_merged_df[chooser_segment_column].reindex(choices_df.index)

        spc.set_choices(choices_df)

        fit = spc.check_fit(iteration)

        if locutor:
            spc.write_trace_files(iteration)

        if fit:
            break

    if fit:
        logging.info("%s converged after iteration %s" % (trace_label, iteration,))
    else:
        logging.info("%s did not converge after iteration %s" % (trace_label, iteration,))

    # - convergence stats
    logging.info("\nshadow_pricing max_abs_diff\n%s" % spc.max_abs_diff)
    logging.info("\nshadow_pricing max_rel_diff\n%s" % spc.max_rel_diff)
    logging.info("\nshadow_pricing num_fail\n%s" % spc.num_fail)

    # - shadow price table
    if locutor:
        if 'SHADOW_PRICE_TABLE' in model_settings:
            inject.add_table(model_settings['SHADOW_PRICE_TABLE'], spc.shadow_prices)
        if 'MODELED_SIZE_TABLE' in model_settings:
            inject.add_table(model_settings['MODELED_SIZE_TABLE'], spc.modeled_size)

    tracing.print_summary('workplace_taz', choices, describe=True)

    persons_df = persons.to_frame()

    # We only chose school locations for the subset of persons who go to school
    # so we backfill the empty choices with -1 to code as no school location
    NO_WORKPLACE_TAZ = -1
    persons_df['workplace_taz'] = \
        choices.reindex(persons_df.index).fillna(NO_WORKPLACE_TAZ).astype(int)

    # - annotate persons
    model_name = 'workplace_location'
    model_settings = config.read_model_settings('workplace_location.yaml')

    expressions.assign_columns(
        df=persons_df,
        model_settings=model_settings.get('annotate_persons'),
        trace_label=tracing.extend_trace_label(model_name, 'annotate_persons'))

    pipeline.replace_table("persons", persons_df)

    if trace_hh_id:
        tracing.trace_df(persons_df,
                         label="workplace_location",
                         warn_if_empty=True)
