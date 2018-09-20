# ActivitySim
# See full license in LICENSE.txt.

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


@inject.step()
def school_location_sample(
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

    trace_label = 'school_location_sample'
    model_settings = config.read_model_settings('school_location.yaml')

    model_spec = simulate.read_model_spec(file_name='school_location_sample.csv')

    choosers = persons_merged.to_frame()
    # FIXME - MEMORY HACK - only include columns actually used in spec
    chooser_columns = model_settings['SIMULATE_CHOOSER_COLUMNS']
    choosers = choosers[chooser_columns]

    size_terms = tour_destination_size_terms(land_use, size_terms, 'school')

    sample_size = model_settings["SAMPLE_SIZE"]
    alt_dest_col_name = model_settings["ALT_DEST_COL_NAME"]

    logger.info("Running school_location_simulate with %d persons" % len(choosers))

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
    for school_type, school_type_id in SCHOOL_TYPE_ID.iteritems():

        locals_d['segment'] = school_type

        choosers_segment = choosers[choosers["is_" + school_type]]

        if choosers_segment.shape[0] == 0:
            logger.info("%s skipping school_type %s: no choosers" % (trace_label, school_type))
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
            trace_label=tracing.extend_trace_label(trace_label, school_type))

        choices['school_type'] = school_type_id
        choices_list.append(choices)

    if len(choices_list) > 0:
        choices = pd.concat(choices_list)
        # - # NARROW
        choices['school_type'] = choices['school_type'].astype(np.uint8)
    else:
        logger.info("Skipping %s: add_null_results" % trace_label)
        choices = pd.DataFrame()

    inject.add_table('school_location_sample', choices)


@inject.step()
def school_location_logsums(
        persons_merged,
        skim_dict, skim_stack,
        school_location_sample,
        chunk_size,
        trace_hh_id):

    """
    add logsum column to existing school_location_sample able

    logsum is calculated by running the mode_choice model for each sample (person, dest_taz) pair
    in school_location_sample, and computing the logsum of all the utilities

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

    trace_label = 'school_location_logsums'

    model_settings = config.read_model_settings('school_location.yaml')
    logsum_settings = config.read_model_settings(model_settings['LOGSUM_SETTINGS'])

    location_sample = school_location_sample.to_frame()

    if location_sample.shape[0] == 0:
        tracing.no_results(trace_label)
        return

    logger.info("Running school_location_logsums with %s rows" % location_sample.shape[0])

    persons_merged = persons_merged.to_frame()
    # - only include columns actually used in spec
    persons_merged = logsum.filter_chooser_columns(persons_merged, logsum_settings, model_settings)

    logsums_list = []
    for school_type, school_type_id in SCHOOL_TYPE_ID.iteritems():

        tour_purpose = 'univ' if school_type == 'university' else 'school'

        choosers = location_sample[location_sample['school_type'] == school_type_id]

        if choosers.shape[0] == 0:
            logger.info("%s skipping school_type %s: no choosers" % (trace_label, school_type))
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
            tracing.extend_trace_label(trace_label, school_type))

        logsums_list.append(logsums)

    logsums = pd.concat(logsums_list)

    # add_column series should have an index matching the table to which it is being added
    # logsums does, since school_location_sample was on left side of merge creating choosers

    # "add_column series should have an index matching the table to which it is being added"
    # when the index has duplicates, however, in the special case that the series index exactly
    # matches the table index, then the series value order is preserved.
    # logsums does align with school_location_sample as we loop through it in exactly the same
    # order as we did when we created it
    inject.add_column('school_location_sample', 'mode_choice_logsum', logsums)


@inject.step()
def school_location_simulate(persons_merged, persons,
                             school_location_sample,
                             skim_dict,
                             land_use, size_terms,
                             chunk_size,
                             trace_hh_id):
    """
    School location model on school_location_sample annotated with mode_choice logsum
    to select a school_taz from sample alternatives
    """
    trace_label = 'school_location_simulate'
    model_settings = config.read_model_settings('school_location.yaml')
    model_spec = simulate.read_model_spec(file_name='school_location.csv')

    NO_SCHOOL_TAZ = -1

    location_sample = school_location_sample.to_frame()
    persons = persons.to_frame()

    # if there are any school-goers
    if location_sample.shape[0] > 0:

        choosers = persons_merged.to_frame()
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
        choosers = choosers[chooser_columns]

        choices_list = []
        for school_type, school_type_id in SCHOOL_TYPE_ID.iteritems():

            locals_d['segment'] = school_type

            choosers_segment = choosers[choosers["is_" + school_type]]

            if choosers_segment.shape[0] == 0:
                logger.info("%s skipping school_type %s: no choosers" % (trace_label, school_type))
                continue

            alts_segment = \
                location_sample[location_sample['school_type'] == school_type_id]

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
                trace_label=tracing.extend_trace_label(trace_label, school_type),
                trace_choice_name='school_location')

            choices_list.append(choices)

        choices = pd.concat(choices_list)

        # We only chose school locations for the subset of persons who go to school
        # so we backfill the empty choices with -1 to code as no school location
        persons['school_taz'] = choices.reindex(persons.index).fillna(NO_SCHOOL_TAZ).astype(int)

        tracing.print_summary('school_taz', choices, describe=True)

    else:

        # no school-goers (but we still want to annotate persons)
        persons['school_taz'] = NO_SCHOOL_TAZ

        logger.info("%s no school-goers" % trace_label)

    expressions.assign_columns(
        df=persons,
        model_settings=model_settings.get('annotate_persons'),
        trace_label=tracing.extend_trace_label(trace_label, 'annotate_persons'))

    pipeline.replace_table("persons", persons)

    pipeline.drop_table('school_location_sample')

    if trace_hh_id:
        tracing.trace_df(persons,
                         label="school_location",
                         warn_if_empty=True)
