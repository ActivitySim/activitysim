# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402

from future.utils import iteritems

import logging
from collections import OrderedDict

import numpy as np
import pandas as pd

from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import pipeline
from activitysim.core import simulate
from activitysim.core import inject

from activitysim.core.interaction_sample_simulate import interaction_sample_simulate
from activitysim.core.interaction_sample import interaction_sample

from activitysim.core.util import reindex

from activitysim.abm.tables import shadow_pricing

from .util import logsums as logsum


from .util import expressions

"""
The school location model predicts the zones in which various people will
go to school.
"""

logger = logging.getLogger(__name__)


NO_SCHOOL_TAZ = -1


# we want to iterate over segment_ids in the same order every time
def order_dict_by_keys(segment_ids):
    return OrderedDict([(k, segment_ids[k]) for k in sorted(segment_ids.keys())])


def run_school_location_sample(
        persons_merged,
        skim_dict,
        dest_size_terms,
        model_settings,
        chunk_size,
        trace_hh_id,
        trace_label):

    """
    build a table of persons * all zones to select a sample of alternative school locations.

    | PERID     | dest_TAZ     | rand           | pick_count |
    +===========+==============+================+============+
    | 23750     |  14          | 0.565502716034 | 4          |
    +-----------+--------------+----------------+------------+
    + 23750     | 16           | 0.711135838871 | 6          |
    +-----------+--------------+----------------+------------+
    + ...       |              |                |            |
    +-----------+--------------+----------------+------------+
    | 23751     | 12           | 0.408038878552 | 1          |
    +-----------+--------------+----------------+------------+
    | 23751     | 14           | 0.972732479292 | 2          |
    +-----------+--------------+----------------+------------+
    """

    model_spec = simulate.read_model_spec(file_name=model_settings['SAMPLE_SPEC'])

    # FIXME - MEMORY HACK - only include columns actually used in spec
    chooser_columns = model_settings['SIMULATE_CHOOSER_COLUMNS']
    choosers = persons_merged[chooser_columns]

    chooser_segment_column = model_settings['CHOOSER_SEGMENT_COLUMN']

    sample_size = model_settings["SAMPLE_SIZE"]
    alt_dest_col_name = model_settings["ALT_DEST_COL_NAME"]

    logger.info("Running school_location_simulate with %d persons", len(choosers))

    # create wrapper with keys for this lookup - in this case there is a TAZ in the choosers
    # and a TAZ in the alternatives which get merged during interaction
    # the skims will be available under the name "skims" for any @ expressions
    skims = skim_dict.wrap("TAZ", "TAZ_r")

    locals_d = {
        'skims': skims
    }
    constants_dict = config.get_model_constants(model_settings)
    if constants_dict is not None:
        locals_d.update(constants_dict)

    # we want to iterate over segment_ids in the same order in sample, logsums, and simulate
    segment_ids = order_dict_by_keys(model_settings['SEGMENT_IDS'])

    choices_list = []
    for segment_name, segment_id in iteritems(segment_ids):

        locals_d['segment'] = segment_name

        choosers_segment = choosers[choosers[chooser_segment_column] == segment_id]

        if choosers_segment.shape[0] == 0:
            logger.info("%s skipping school_type %s: no choosers", trace_label, segment_name)
            continue

        # alts indexed by taz with one column containing size_term for  this tour_type
        alternatives_segment = dest_size_terms[[segment_name]]

        # no point in considering impossible alternatives (where dest size term is zero)
        alternatives_segment = alternatives_segment[alternatives_segment[segment_name] > 0]

        logger.info("%s segment %s:  %s persons %s alternatives" %
                    (trace_label, segment_name, len(choosers_segment), len(alternatives_segment)))

        choices = interaction_sample(
            choosers_segment,
            alternatives_segment,
            sample_size=sample_size,
            alt_col_name=alt_dest_col_name,
            spec=model_spec[[segment_name]],
            skims=skims,
            locals_d=locals_d,
            chunk_size=chunk_size,
            trace_label=tracing.extend_trace_label(trace_label, segment_name))

        choices['segment_id'] = segment_id
        choices_list.append(choices)

    if len(choices_list) > 0:
        choices = pd.concat(choices_list)
        # - NARROW
        choices['segment_id'] = choices['segment_id'].astype(np.uint8)
    else:
        logger.info("Skipping %s: add_null_results" % trace_label)
        choices = pd.DataFrame()

    return choices


def run_school_location_logsums(
        persons_merged,
        skim_dict, skim_stack,
        location_sample_df,
        model_settings,
        chunk_size,
        trace_hh_id,
        trace_label):

    """
    compute and add mode_choice_logsum column to location_sample_df

    logsum is calculated by running the mode_choice model for each sample (person, dest_taz) pair
    in location_sample_df, and computing the logsum of all the utilities

    +-----------+--------------+----------------+------------+----------------+
    | person_id | dest_TAZ     | rand           | pick_count | logsum (added) |
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

    logsum_settings = config.read_model_settings(model_settings['LOGSUM_SETTINGS'])

    if location_sample_df.empty:
        tracing.no_results(trace_label)
        return location_sample_df

    logger.info("Running school_location_logsums with %s rows" % location_sample_df.shape[0])

    # - only include columns actually used in spec
    persons_merged = logsum.filter_chooser_columns(persons_merged, logsum_settings, model_settings)

    # we want to iterate over segment_ids in the same order in sample, logsums, and simulate
    segment_ids = order_dict_by_keys(model_settings['SEGMENT_IDS'])

    logsums_list = []
    for segment_name, segment_id in iteritems(segment_ids):

        # FIXME - pathological knowledge of relation between segment name and tour_purpose
        tour_purpose = 'univ' if segment_name == 'university' else 'school'

        choosers = location_sample_df[location_sample_df['segment_id'] == segment_id]

        if choosers.shape[0] == 0:
            logger.info("%s skipping school_type %s: no choosers" % (trace_label, segment_name))
            continue

        choosers = pd.merge(
            choosers,
            persons_merged,
            left_index=True,
            right_index=True,
            how="left")

        logsums = logsum.compute_logsums(
            choosers,
            tour_purpose,
            logsum_settings, model_settings,
            skim_dict, skim_stack,
            chunk_size, trace_hh_id,
            tracing.extend_trace_label(trace_label, segment_name))

        logsums_list.append(logsums)

    logsums = pd.concat(logsums_list)

    # add_column series should have an index matching the table to which it is being added
    # logsums does, since location_sample_df was on left side of merge creating choosers

    # "add_column series should have an index matching the table to which it is being added"
    # when the index has duplicates, however, in the special case that the series index exactly
    # matches the table index, then the series value order is preserved.
    # logsums does align with location_sample_df as we loop through it in exactly the same
    # order as we did when we created it
    location_sample_df['mode_choice_logsum'] = logsums

    return location_sample_df


def run_school_location_simulate(
        persons_merged,
        location_sample_df,
        skim_dict,
        dest_size_terms,
        model_settings,
        chunk_size,
        trace_hh_id,
        trace_label):
    """
    School location model on school_location_sample annotated with mode_choice logsum
    to select a school_taz from sample alternatives
    """

    model_spec = simulate.read_model_spec(file_name=model_settings['SPEC'])

    if location_sample_df.empty:

        logger.info("%s no school-goers" % trace_label)
        choices = pd.Series()

    else:

        alt_dest_col_name = model_settings["ALT_DEST_COL_NAME"]

        # create wrapper with keys for this lookup - in this case there is a TAZ in the choosers
        # and a TAZ in the alternatives which get merged during interaction
        # the skims will be available under the name "skims" for any @ expressions
        skims = skim_dict.wrap("TAZ", alt_dest_col_name)

        locals_d = {
            'skims': skims,
        }
        constants_dict = config.get_model_constants(model_settings)
        if constants_dict is not None:
            locals_d.update(constants_dict)

        # FIXME - MEMORY HACK - only include columns actually used in spec
        chooser_columns = model_settings['SIMULATE_CHOOSER_COLUMNS']
        choosers = persons_merged[chooser_columns]

        # we want to iterate over segment_ids in the same order in sample, logsums, and simulate
        segment_ids = order_dict_by_keys(model_settings['SEGMENT_IDS'])

        choices_list = []
        for segment_name, segment_id in iteritems(segment_ids):

            locals_d['segment'] = segment_name

            choosers_segment = choosers[choosers['school_segment'] == segment_id]

            if choosers_segment.shape[0] == 0:
                logger.info("%s skipping school_type %s: no choosers" % (trace_label, segment_name))
                continue

            alts_segment = \
                location_sample_df[location_sample_df['segment_id'] == segment_id]

            # alternatives are pre-sampled and annotated with logsums and pick_count
            # but we have to merge dest_choice_size column into alt sample list
            alts_segment[segment_name] = \
                reindex(dest_size_terms[segment_name], alts_segment[alt_dest_col_name])

            choices = interaction_sample_simulate(
                choosers_segment,
                alts_segment,
                spec=model_spec[[segment_name]],
                choice_column=alt_dest_col_name,
                skims=skims,
                locals_d=locals_d,
                chunk_size=chunk_size,
                trace_label=tracing.extend_trace_label(trace_label, segment_name),
                trace_choice_name=trace_label)

            choices_list.append(choices)

        choices = pd.concat(choices_list)

    return choices


def run_school_location(
        persons_merged_df,
        skim_dict, skim_stack,
        dest_size_terms,
        model_settings,
        chunk_size, trace_hh_id, trace_label
        ):

    # - school_location_sample
    location_sample_df = \
        run_school_location_sample(
            persons_merged_df,
            skim_dict,
            dest_size_terms,
            model_settings,
            chunk_size,
            trace_hh_id,
            tracing.extend_trace_label(trace_label, 'sample'))

    # - school_location_logsums
    location_sample_df = \
        run_school_location_logsums(
            persons_merged_df,
            skim_dict, skim_stack,
            location_sample_df,
            model_settings,
            chunk_size,
            trace_hh_id,
            tracing.extend_trace_label(trace_label, 'logsums'))

    # - school_location_simulate
    choices = \
        run_school_location_simulate(
            persons_merged_df,
            location_sample_df,
            skim_dict,
            dest_size_terms,
            model_settings,
            chunk_size,
            trace_hh_id,
            tracing.extend_trace_label(trace_label, 'simulate'))

    return choices


@inject.step()
def school_location(
        persons_merged, persons,
        skim_dict, skim_stack,
        chunk_size,
        trace_hh_id):

    trace_label = 'school_location'
    model_settings = config.read_model_settings('school_location.yaml')

    chooser_segment_column = model_settings['CHOOSER_SEGMENT_COLUMN']

    persons_merged_df = persons_merged.to_frame()

    spc = shadow_pricing.load_shadow_price_calculator(model_settings)

    # - max_iterations
    if spc.saved_shadow_prices:
        max_iterations = model_settings.get('MAX_SHADOW_PRICE_ITERATIONS_WITH_SAVED', 1)
    else:
        max_iterations = model_settings.get('MAX_SHADOW_PRICE_ITERATIONS', 5)
    logging.debug("%s max_iterations: %s" % (trace_label, max_iterations))

    choices = None
    for iteration in range(max_iterations):

        if iteration > 0:
            spc.update_shadow_prices()

        # - shadow_price adjusted predicted_size
        shadow_price_adjusted_predicted_size = spc.predicted_size * spc.shadow_prices

        choices = run_school_location(
            persons_merged_df,
            skim_dict, skim_stack,
            shadow_price_adjusted_predicted_size,
            model_settings,
            chunk_size, trace_hh_id,
            trace_label=tracing.extend_trace_label(trace_label, 'i%s' % iteration))

        choices_df = choices.to_frame('dest_choice')
        choices_df['segment_id'] = \
            persons_merged_df[chooser_segment_column].reindex(choices_df.index)

        spc.set_choices(choices_df)

        number_of_failed_zones = spc.check_fit(iteration)

        logging.info("%s iteration: %s number_of_failed_zones: %s" %
                     (trace_label, iteration, number_of_failed_zones))

        if number_of_failed_zones == 0:
            break

    # - print convergence stats
    # print("\nshadow_pricing rms_error\n", spc.rms_error)
    print("\nshadow_pricing num_fail\n", spc.num_fail)

    persons_df = persons.to_frame()

    # We only chose school locations for the subset of persons who go to school
    # so we backfill the empty choices with -1 to code as no school location
    persons_df['school_taz'] = choices.reindex(persons_df.index).fillna(NO_SCHOOL_TAZ).astype(int)
    # tracing.print_summary('school_taz', choices, value_counts=True)

    # - shadow price table
    if 'SHADOW_PRICE_TABLE' in model_settings:
        inject.add_table(model_settings['SHADOW_PRICE_TABLE'], spc.shadow_prices)
    if 'MODELED_SIZE_TABLE' in model_settings:
        inject.add_table(model_settings['MODELED_SIZE_TABLE'], spc.modeled_size)

    # - annotate persons
    model_name = 'school_location'
    model_settings = config.read_model_settings('school_location.yaml')
    expressions.assign_columns(
        df=persons_df,
        model_settings=model_settings.get('annotate_persons'),
        trace_label=tracing.extend_trace_label(model_name, 'annotate_persons'))

    pipeline.replace_table("persons", persons_df)

    if trace_hh_id:
        tracing.trace_df(persons_df,
                         label="school_location",
                         warn_if_empty=True)
