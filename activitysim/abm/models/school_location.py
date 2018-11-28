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
from activitysim.core.util import left_merge_on_index_and_col

from .util import logsums as logsum
from .util.tour_destination import tour_destination_size_terms

from .util import expressions

"""
The school location model predicts the zones in which various people will
go to school.
"""

logger = logging.getLogger(__name__)

# use int not str to identify school type in sample df
SCHOOL_TYPE_ID = OrderedDict([('university', 1), ('highschool', 2), ('gradeschool', 3)])


def run_school_location_sample(
        persons_merged,
        skim_dict,
        land_use, size_terms,
        chunk_size,
        trace_hh_id):

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

    model_name = 'school_location_sample'
    model_settings = config.read_model_settings('school_location.yaml')
    model_spec = simulate.read_model_spec(file_name=model_settings['SAMPLE_SPEC'])

    # FIXME - MEMORY HACK - only include columns actually used in spec
    chooser_columns = model_settings['SIMULATE_CHOOSER_COLUMNS']
    choosers = persons_merged[chooser_columns]

    size_terms = tour_destination_size_terms(land_use, size_terms, 'school')

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
    constants = config.get_model_constants(model_settings)
    if constants is not None:
        locals_d.update(constants)

    choices_list = []
    for school_type, school_type_id in iteritems(SCHOOL_TYPE_ID):

        locals_d['segment'] = school_type

        choosers_segment = choosers[choosers["is_" + school_type]]

        if choosers_segment.shape[0] == 0:
            logger.info("%s skipping school_type %s: no choosers", model_name, school_type)
            continue

        # alts indexed by taz with one column containing size_term for  this tour_type
        alternatives_segment = size_terms[[school_type]]

        # no point in considering impossible alternatives (where dest size term is zero)
        alternatives_segment = alternatives_segment[alternatives_segment[school_type] > 0]

        logger.info("school_type %s:  %s persons %s alternatives" %
                    (school_type, len(choosers_segment), len(alternatives_segment)))

        choices = interaction_sample(
            choosers_segment,
            alternatives_segment,
            sample_size=sample_size,
            alt_col_name=alt_dest_col_name,
            spec=model_spec[[school_type]],
            skims=skims,
            locals_d=locals_d,
            chunk_size=chunk_size,
            trace_label=tracing.extend_trace_label(model_name, school_type))

        choices['school_type'] = school_type_id
        choices_list.append(choices)

    if len(choices_list) > 0:
        choices = pd.concat(choices_list)
        # - NARROW
        choices['school_type'] = choices['school_type'].astype(np.uint8)
    else:
        logger.info("Skipping %s: add_null_results" % model_name)
        choices = pd.DataFrame()

    return choices


def run_school_location_logsums(
        persons_merged,
        skim_dict, skim_stack,
        location_sample_df,
        chunk_size,
        trace_hh_id):

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
    model_name = 'school_location_logsums'
    model_settings = config.read_model_settings('school_location.yaml')

    logsum_settings = config.read_model_settings(model_settings['LOGSUM_SETTINGS'])

    if location_sample_df.empty:
        tracing.no_results(model_name)
        return location_sample_df

    logger.info("Running school_location_logsums with %s rows" % location_sample_df.shape[0])

    # - only include columns actually used in spec
    persons_merged = logsum.filter_chooser_columns(persons_merged, logsum_settings, model_settings)

    logsums_list = []
    for school_type, school_type_id in iteritems(SCHOOL_TYPE_ID):

        tour_purpose = 'univ' if school_type == 'university' else 'school'

        choosers = location_sample_df[location_sample_df['school_type'] == school_type_id]

        if choosers.shape[0] == 0:
            logger.info("%s skipping school_type %s: no choosers" % (model_name, school_type))
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
            tracing.extend_trace_label(model_name, school_type))

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
        land_use, size_terms,
        chunk_size,
        trace_hh_id):
    """
    School location model on school_location_sample annotated with mode_choice logsum
    to select a school_taz from sample alternatives
    """
    model_name = 'school_location'
    model_settings = config.read_model_settings('school_location.yaml')

    model_spec = simulate.read_model_spec(file_name=model_settings['SPEC'])

    if location_sample_df.empty:

        logger.info("%s no school-goers" % model_name)
        choices = pd.Series()

    else:

        destination_size_terms = tour_destination_size_terms(land_use, size_terms, 'school')

        alt_dest_col_name = model_settings["ALT_DEST_COL_NAME"]

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

        # FIXME - MEMORY HACK - only include columns actually used in spec
        chooser_columns = model_settings['SIMULATE_CHOOSER_COLUMNS']
        choosers = persons_merged[chooser_columns]

        choices_list = []
        for school_type, school_type_id in iteritems(SCHOOL_TYPE_ID):

            locals_d['segment'] = school_type

            choosers_segment = choosers[choosers["is_" + school_type]]

            if choosers_segment.shape[0] == 0:
                logger.info("%s skipping school_type %s: no choosers" % (model_name, school_type))
                continue

            alts_segment = \
                location_sample_df[location_sample_df['school_type'] == school_type_id]

            # alternatives are pre-sampled and annotated with logsums and pick_count
            # but we have to merge size_terms column into alt sample list
            alts_segment[school_type] = \
                reindex(destination_size_terms[school_type], alts_segment[alt_dest_col_name])

            choices = interaction_sample_simulate(
                choosers_segment,
                alts_segment,
                spec=model_spec[[school_type]],
                choice_column=alt_dest_col_name,
                skims=skims,
                locals_d=locals_d,
                chunk_size=chunk_size,
                trace_label=tracing.extend_trace_label(model_name, school_type),
                trace_choice_name=model_name)

            choices_list.append(choices)

        choices = pd.concat(choices_list)

    return choices


@inject.step()
def school_location(
        persons_merged, persons,
        skim_dict, skim_stack,
        land_use, size_terms,
        chunk_size,
        trace_hh_id):

    persons_merged_df = persons_merged.to_frame()

    # - school_location_sample
    location_sample_df = \
        run_school_location_sample(
            persons_merged_df,
            skim_dict,
            land_use, size_terms,
            chunk_size,
            trace_hh_id)

    # - school_location_logsums
    location_sample_df = \
        run_school_location_logsums(
            persons_merged_df,
            skim_dict, skim_stack,
            location_sample_df,
            chunk_size,
            trace_hh_id)

    # - school_location_simulate
    choices = \
        run_school_location_simulate(
            persons_merged_df,
            location_sample_df,
            skim_dict,
            land_use, size_terms,
            chunk_size,
            trace_hh_id)

    tracing.print_summary('school_taz', choices, describe=True)

    persons_df = persons.to_frame()

    # We only chose school locations for the subset of persons who go to school
    # so we backfill the empty choices with -1 to code as no school location
    NO_SCHOOL_TAZ = -1
    persons_df['school_taz'] = choices.reindex(persons_df.index).fillna(NO_SCHOOL_TAZ).astype(int)
    tracing.print_summary('school_taz', choices, describe=True)

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
