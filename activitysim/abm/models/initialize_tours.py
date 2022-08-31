# ActivitySim
# See full license in LICENSE.txt.
import logging
import os
import warnings

import pandas as pd

from activitysim.abm.models.util import tour_frequency as tf
from activitysim.core import config, expressions, inject, pipeline, tracing
from activitysim.core.input import read_input_table

logger = logging.getLogger(__name__)

SURVEY_TOUR_ID = "external_tour_id"
SURVEY_PARENT_TOUR_ID = "external_parent_tour_id"
SURVEY_PARTICIPANT_ID = "external_participant_id"
ASIM_TOUR_ID = "tour_id"
ASIM_PARENT_TOUR_ID = "parent_tour_id"
REQUIRED_TOUR_COLUMNS = set(["person_id", "tour_category", "tour_type"])


def patch_tour_ids(tours):
    def set_tour_index(tours, parent_tour_num_col, is_joint):
        group_cols = ["person_id", "tour_category", "tour_type"]

        if "parent_tour_num" in tours:
            group_cols += ["parent_tour_num"]

        tours["tour_type_num"] = (
            tours.sort_values(by=group_cols).groupby(group_cols).cumcount() + 1
        )

        return tf.set_tour_index(
            tours, parent_tour_num_col=parent_tour_num_col, is_joint=is_joint
        )

    assert REQUIRED_TOUR_COLUMNS.issubset(
        set(tours.columns)
    ), f"Required columns missing from tours table: {REQUIRED_TOUR_COLUMNS.difference(set(tours.columns))}"

    # replace tour index with asim standard tour_ids (which are based on person_id and tour_type)
    if tours.index.name is not None:
        tours.insert(loc=0, column="legacy_index", value=tours.index)

    # FIXME - for now, only grok simple tours
    assert set(tours.tour_category.unique()).issubset({"mandatory", "non_mandatory"})

    # mandatory tours
    mandatory_tours = set_tour_index(
        tours[tours.tour_category == "mandatory"],
        parent_tour_num_col=None,
        is_joint=False,
    )

    assert mandatory_tours.index.name == "tour_id"

    # FIXME joint tours not implemented
    assert not (tours.tour_category == "joint").any()

    # non_mandatory tours
    non_mandatory_tours = set_tour_index(
        tours[tours.tour_category == "non_mandatory"],
        parent_tour_num_col=None,
        is_joint=False,
    )

    # FIXME atwork tours ot implemented
    assert not (tours.tour_category == "atwork").any()

    patched_tours = pd.concat([mandatory_tours, non_mandatory_tours])
    del patched_tours["tour_type_num"]

    return patched_tours


@inject.step()
def initialize_tours(network_los, households, persons, trace_hh_id):

    trace_label = "initialize_tours"

    tours = read_input_table("tours")

    # FIXME can't use households_sliced injectable as flag like persons table does in case of resume_after.
    # FIXME could just always slice...
    slice_happened = (
        inject.get_injectable("households_sample_size", 0) > 0
        or inject.get_injectable("households_sample_size", 0) > 0
    )
    if slice_happened:
        logger.info("slicing tours %s" % (tours.shape,))
        # keep all persons in the sampled households
        tours = tours[tours.person_id.isin(persons.index)]

    # annotate before patching tour_id to allow addition of REQUIRED_TOUR_COLUMNS defined above
    model_settings = config.read_model_settings("initialize_tours.yaml", mandatory=True)
    expressions.assign_columns(
        df=tours,
        model_settings=model_settings.get("annotate_tours"),
        trace_label=tracing.extend_trace_label(trace_label, "annotate_tours"),
    )

    skip_patch_tour_ids = model_settings.get("skip_patch_tour_ids", False)
    if skip_patch_tour_ids:
        pass
    else:
        tours = patch_tour_ids(tours)
    assert tours.index.name == "tour_id"

    # replace table function with dataframe
    inject.add_table("tours", tours)

    pipeline.get_rn_generator().add_channel("tours", tours)

    tracing.register_traceable_table("tours", tours)

    logger.debug(f"{len(tours.household_id.unique())} unique household_ids in tours")
    logger.debug(f"{len(households.index.unique())} unique household_ids in households")
    assert not tours.index.duplicated().any()

    tours_without_persons = ~tours.person_id.isin(persons.index)
    if tours_without_persons.any():
        logger.error(
            f"{tours_without_persons.sum()} tours out of {len(persons)} without persons\n"
            f"{pd.Series({'person_id': tours_without_persons.index.values})}"
        )
        raise RuntimeError(f"{tours_without_persons.sum()} tours with bad person_id")

    if trace_hh_id:
        tracing.trace_df(tours, label="initialize_tours", warn_if_empty=True)
