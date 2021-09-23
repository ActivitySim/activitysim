# ActivitySim
# See full license in LICENSE.txt.
import logging

import numpy as np
import pandas as pd

from activitysim.core import (
    config,
    expressions,
    inject,
    logit,
    pipeline,
    simulate,
    tracing,
)
from activitysim.core.interaction_simulate import interaction_simulate

from .util import estimation
from .util.overlap import person_max_window
from .util.tour_frequency import process_non_mandatory_tours

logger = logging.getLogger(__name__)


def extension_probs():
    f = config.config_file_path("non_mandatory_tour_frequency_extension_probs.csv")
    df = pd.read_csv(f, comment="#")

    # convert cum probs to individual probs
    df["2_tours"] = df["2_tours"] - df["1_tours"]
    df["1_tours"] = df["1_tours"] - df["0_tours"]

    return df


def extend_tour_counts(persons, tour_counts, alternatives, trace_hh_id, trace_label):
    """
    extend tour counts based on a probability table

    counts can only be extended if original count is between 1 and 4
    and tours can only be extended if their count is at the max possible
    (e.g. 2 for escort, 1 otherwise) so escort might be increased to 3 or 4
    and other tour types might be increased to 2 or 3

    Parameters
    ----------
    persons: pandas dataframe
        (need this for join columns)
    tour_counts: pandas dataframe
        one row per person, once column per tour_type
    alternatives
        alternatives from nmtv interaction_simulate
        only need this to know max possible frequency for a tour type
    trace_hh_id
    trace_label

    Returns
    -------
    extended tour_counts


    tour_counts looks like this:
               escort  shopping  othmaint  othdiscr    eatout    social
    parent_id
    2588676         2         0         0         1         1         0
    2588677         0         1         0         1         0         0

    """

    assert tour_counts.index.name == persons.index.name

    PROBABILITY_COLUMNS = ["0_tours", "1_tours", "2_tours"]
    JOIN_COLUMNS = ["ptype", "has_mandatory_tour", "has_joint_tour"]
    TOUR_TYPE_COL = "nonmandatory_tour_type"

    probs_spec = extension_probs()
    persons = persons[JOIN_COLUMNS]

    # only extend if there are 1 - 4 non_mandatory tours to start with
    extend_tour_counts = tour_counts.sum(axis=1).between(1, 4)
    if not extend_tour_counts.any():
        logger.info("extend_tour_counts - no persons eligible for tour_count extension")
        return tour_counts

    have_trace_targets = trace_hh_id and tracing.has_trace_targets(extend_tour_counts)

    for i, tour_type in enumerate(alternatives.columns):

        i_tour_type = i + 1  # (probs_spec nonmandatory_tour_type column is 1-based)
        tour_type_trace_label = tracing.extend_trace_label(trace_label, tour_type)

        # - only extend tour if frequency is max possible frequency for this tour type
        tour_type_is_maxed = extend_tour_counts & (
            tour_counts[tour_type] == alternatives[tour_type].max()
        )
        maxed_tour_count_idx = tour_counts.index[tour_type_is_maxed]

        if len(maxed_tour_count_idx) == 0:
            continue

        # - get extension probs for tour_type
        choosers = pd.merge(
            persons.loc[maxed_tour_count_idx],
            probs_spec[probs_spec[TOUR_TYPE_COL] == i_tour_type],
            on=JOIN_COLUMNS,
            how="left",
        ).set_index(maxed_tour_count_idx)
        assert choosers.index.name == tour_counts.index.name

        # - random choice of extension magnitude based on relative probs
        choices, rands = logit.make_choices(
            choosers[PROBABILITY_COLUMNS],
            trace_label=tour_type_trace_label,
            trace_choosers=choosers,
        )

        # - extend tour_count (0-based prob alternative choice equals magnitude of extension)
        if choices.any():
            tour_counts.loc[choices.index, tour_type] += choices

        if have_trace_targets:
            tracing.trace_df(
                choices,
                tracing.extend_trace_label(tour_type_trace_label, "choices"),
                columns=[None, "choice"],
            )
            tracing.trace_df(
                rands,
                tracing.extend_trace_label(tour_type_trace_label, "rands"),
                columns=[None, "rand"],
            )

    return tour_counts


@inject.step()
def non_mandatory_tour_frequency(persons, persons_merged, chunk_size, trace_hh_id):
    """
    This model predicts the frequency of making non-mandatory trips
    (alternatives for this model come from a separate csv file which is
    configured by the user) - these trips include escort, shopping, othmaint,
    othdiscr, eatout, and social trips in various combination.
    """

    trace_label = "non_mandatory_tour_frequency"
    model_settings_file_name = "non_mandatory_tour_frequency.yaml"

    model_settings = config.read_model_settings(model_settings_file_name)

    # FIXME kind of tacky both that we know to add this here and del it below
    # 'tot_tours' is used in model_spec expressions
    alternatives = simulate.read_model_alts(
        "non_mandatory_tour_frequency_alternatives.csv", set_index=None
    )
    alternatives["tot_tours"] = alternatives.sum(axis=1)

    # filter based on results of CDAP
    choosers = persons_merged.to_frame()
    choosers = choosers[choosers.cdap_activity.isin(["M", "N"])]

    # - preprocessor
    preprocessor_settings = model_settings.get("preprocessor", None)
    if preprocessor_settings:

        locals_dict = {"person_max_window": person_max_window}

        expressions.assign_columns(
            df=choosers,
            model_settings=preprocessor_settings,
            locals_dict=locals_dict,
            trace_label=trace_label,
        )

    logger.info("Running non_mandatory_tour_frequency with %d persons", len(choosers))

    constants = config.get_model_constants(model_settings)

    model_spec = simulate.read_model_spec(file_name=model_settings["SPEC"])
    spec_segments = model_settings.get("SPEC_SEGMENTS", {})

    # segment by person type and pick the right spec for each person type
    choices_list = []
    for segment_settings in spec_segments:

        segment_name = segment_settings["NAME"]
        ptype = segment_settings["PTYPE"]

        # pick the spec column for the segment
        segment_spec = model_spec[[segment_name]]

        chooser_segment = choosers[choosers.ptype == ptype]

        logger.info(
            "Running segment '%s' of size %d", segment_name, len(chooser_segment)
        )

        if len(chooser_segment) == 0:
            # skip empty segments
            continue

        estimator = estimation.manager.begin_estimation(
            model_name=segment_name, bundle_name="non_mandatory_tour_frequency"
        )

        coefficients_df = simulate.read_model_coefficients(segment_settings)
        segment_spec = simulate.eval_coefficients(
            segment_spec, coefficients_df, estimator
        )

        if estimator:
            estimator.write_spec(model_settings, bundle_directory=True)
            estimator.write_model_settings(
                model_settings, model_settings_file_name, bundle_directory=True
            )
            # preserving coefficients file name makes bringing back updated coefficients more straightforward
            estimator.write_coefficients(coefficients_df, segment_settings)
            estimator.write_choosers(chooser_segment)
            estimator.write_alternatives(alternatives, bundle_directory=True)

            # FIXME #interaction_simulate_estimation_requires_chooser_id_in_df_column
            #  shuold we do it here or have interaction_simulate do it?
            # chooser index must be duplicated in column or it will be omitted from interaction_dataset
            # estimation requires that chooser_id is either in index or a column of interaction_dataset
            # so it can be reformatted (melted) and indexed by chooser_id and alt_id
            assert chooser_segment.index.name == "person_id"
            assert "person_id" not in chooser_segment.columns
            chooser_segment["person_id"] = chooser_segment.index

            # FIXME set_alt_id - do we need this for interaction_simulate estimation bundle tables?
            estimator.set_alt_id("alt_id")

            estimator.set_chooser_id(chooser_segment.index.name)

        log_alt_losers = config.setting("log_alt_losers", False)

        choices = interaction_simulate(
            chooser_segment,
            alternatives,
            spec=segment_spec,
            log_alt_losers=log_alt_losers,
            locals_d=constants,
            chunk_size=chunk_size,
            trace_label="non_mandatory_tour_frequency.%s" % segment_name,
            trace_choice_name="non_mandatory_tour_frequency",
            estimator=estimator,
        )

        if estimator:
            estimator.write_choices(choices)
            choices = estimator.get_survey_values(
                choices, "persons", "non_mandatory_tour_frequency"
            )
            estimator.write_override_choices(choices)
            estimator.end_estimation()

        choices_list.append(choices)

    del alternatives["tot_tours"]  # del tot_tours column we added above

    # The choice value 'non_mandatory_tour_frequency' assigned by interaction_simulate
    # is the index value of the chosen alternative in the alternatives table.
    choices = pd.concat(choices_list).sort_index()

    # add non_mandatory_tour_frequency column to persons
    persons = persons.to_frame()
    # we expect there to be an alt with no tours - which we can use to backfill non-travelers
    no_tours_alt = (alternatives.sum(axis=1) == 0).index[0]
    # need to reindex as we only handled persons with cdap_activity in ['M', 'N']
    persons["non_mandatory_tour_frequency"] = (
        choices.reindex(persons.index).fillna(no_tours_alt).astype(np.int8)
    )

    """
    We have now generated non-mandatory tour frequencies, but they are attributes of the person table
    Now we create a "tours" table which has one row per tour that has been generated
    (and the person id it is associated with)

    But before we do that, we run an additional probablilistic step to extend/increase tour counts
    beyond the strict limits of the tour_frequency alternatives chosen above (which are currently limited
    to at most 2 escort tours and 1 each of shopping, othmaint, othdiscr, eatout, and social tours)

    The choice value 'non_mandatory_tour_frequency' assigned by interaction_simulate is simply the
    index value of the chosen alternative in the alternatives table.

    get counts of each of the tour type alternatives (so we can extend)
               escort  shopping  othmaint  othdiscr    eatout    social
    parent_id
    2588676         2         0         0         1         1         0
    2588677         0         1         0         1         0         0
    """

    # counts of each of the tour type alternatives (so we can extend)
    modeled_tour_counts = alternatives.loc[choices]
    modeled_tour_counts.index = choices.index  # assign person ids to the index

    # - extend_tour_counts - probabalistic
    extended_tour_counts = extend_tour_counts(
        choosers,
        modeled_tour_counts.copy(),
        alternatives,
        trace_hh_id,
        tracing.extend_trace_label(trace_label, "extend_tour_counts"),
    )

    num_modeled_tours = modeled_tour_counts.sum().sum()
    num_extended_tours = extended_tour_counts.sum().sum()
    logger.info(
        "extend_tour_counts increased tour count by %s from %s to %s"
        % (
            num_extended_tours - num_modeled_tours,
            num_modeled_tours,
            num_extended_tours,
        )
    )

    """
    create the non_mandatory tours based on extended_tour_counts
    """
    if estimator:
        override_tour_counts = estimation.manager.get_survey_values(
            extended_tour_counts,
            table_name="persons",
            column_names=["_%s" % c for c in extended_tour_counts.columns],
        )
        override_tour_counts = override_tour_counts.rename(
            columns={("_%s" % c): c for c in extended_tour_counts.columns}
        )
        logger.info(
            "estimation get_survey_values override_tour_counts %s changed cells"
            % (override_tour_counts != extended_tour_counts).sum().sum()
        )
        extended_tour_counts = override_tour_counts

    """
    create the non_mandatory tours based on extended_tour_counts
    """
    non_mandatory_tours = process_non_mandatory_tours(persons, extended_tour_counts)
    assert len(non_mandatory_tours) == extended_tour_counts.sum().sum()

    if estimator:

        # make sure they created the right tours
        survey_tours = estimation.manager.get_survey_table("tours").sort_index()
        non_mandatory_survey_tours = survey_tours[
            survey_tours.tour_category == "non_mandatory"
        ]
        assert len(non_mandatory_survey_tours) == len(non_mandatory_tours)
        assert non_mandatory_survey_tours.index.equals(
            non_mandatory_tours.sort_index().index
        )

        # make sure they created tours with the expected tour_ids
        columns = ["person_id", "household_id", "tour_type", "tour_category"]
        survey_tours = estimation.manager.get_survey_values(
            non_mandatory_tours, table_name="tours", column_names=columns
        )

        tours_differ = (non_mandatory_tours[columns] != survey_tours[columns]).any(
            axis=1
        )

        if tours_differ.any():
            print("tours_differ\n%s" % tours_differ)
            print("%s of %s tours differ" % (tours_differ.sum(), len(tours_differ)))
            print("differing survey_tours\n%s" % survey_tours[tours_differ])
            print(
                "differing modeled_tours\n%s"
                % non_mandatory_tours[columns][tours_differ]
            )

        assert not tours_differ.any()

    pipeline.extend_table("tours", non_mandatory_tours)

    tracing.register_traceable_table("tours", non_mandatory_tours)
    pipeline.get_rn_generator().add_channel("tours", non_mandatory_tours)

    expressions.assign_columns(
        df=persons,
        model_settings=model_settings.get("annotate_persons"),
        trace_label=trace_label,
    )

    pipeline.replace_table("persons", persons)

    tracing.print_summary(
        "non_mandatory_tour_frequency",
        persons.non_mandatory_tour_frequency,
        value_counts=True,
    )

    if trace_hh_id:
        tracing.trace_df(
            non_mandatory_tours,
            label="non_mandatory_tour_frequency.non_mandatory_tours",
            warn_if_empty=True,
        )

        tracing.trace_df(
            choosers, label="non_mandatory_tour_frequency.choosers", warn_if_empty=True
        )

        tracing.trace_df(
            persons,
            label="non_mandatory_tour_frequency.annotated_persons",
            warn_if_empty=True,
        )
