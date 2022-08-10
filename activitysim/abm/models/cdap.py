# ActivitySim
# See full license in LICENSE.txt.
import logging

import pandas as pd

from activitysim.core import config, expressions, inject, pipeline, simulate, tracing
from activitysim.core.util import reindex

from .util import cdap, estimation

logger = logging.getLogger(__name__)


@inject.step()
def cdap_simulate(persons_merged, persons, households, chunk_size, trace_hh_id):
    """
    CDAP stands for Coordinated Daily Activity Pattern, which is a choice of
    high-level activity pattern for each person, in a coordinated way with other
    members of a person's household.

    Because Python requires vectorization of computation, there are some specialized
    routines in the cdap directory of activitysim for this purpose.  This module
    simply applies those utilities using the simulation framework.
    """

    trace_label = "cdap"
    model_settings = config.read_model_settings("cdap.yaml")
    person_type_map = model_settings.get("PERSON_TYPE_MAP", None)
    assert (
        person_type_map is not None
    ), f"Expected to find PERSON_TYPE_MAP setting in cdap.yaml"
    estimator = estimation.manager.begin_estimation("cdap")

    cdap_indiv_spec = simulate.read_model_spec(
        file_name=model_settings["INDIV_AND_HHSIZE1_SPEC"]
    )

    coefficients_df = simulate.read_model_coefficients(model_settings)
    cdap_indiv_spec = simulate.eval_coefficients(
        cdap_indiv_spec, coefficients_df, estimator
    )

    # Rules and coefficients for generating interaction specs for different household sizes
    interaction_coefficients_file_name = model_settings.get(
        "INTERACTION_COEFFICIENTS", "cdap_interaction_coefficients.csv"
    )
    cdap_interaction_coefficients = pd.read_csv(
        config.config_file_path(interaction_coefficients_file_name), comment="#"
    )

    # replace cdap_interaction_coefficients coefficient labels with numeric values
    # for backward compatibility, use where() to allow hard-coded coefficients and dummy (empty) coefficients_file
    coefficients = cdap_interaction_coefficients.coefficient.map(
        coefficients_df.value.to_dict()
    )
    coefficients = cdap_interaction_coefficients.coefficient.where(
        coefficients.isnull(), coefficients
    )
    coefficients = pd.to_numeric(coefficients, errors="coerce").astype(float)
    if coefficients.isnull().any():
        # show them the offending lines from interaction_coefficients_file
        logger.warning(
            f"bad coefficients in INTERACTION_COEFFICIENTS {interaction_coefficients_file_name}\n"
            f"{cdap_interaction_coefficients[coefficients.isnull()]}"
        )
        assert not coefficients.isnull().any()
    cdap_interaction_coefficients.coefficient = coefficients

    """
    spec to compute/specify the relative proportions of each activity (M, N, H)
    that should be used to choose activities for additional household members not handled by CDAP
    This spec is handled much like an activitysim logit utility spec,
    EXCEPT that the values computed are relative proportions, not utilities
    (i.e. values are not exponentiated before being normalized to probabilities summing to 1.0)
    """
    cdap_fixed_relative_proportions = simulate.read_model_spec(
        file_name=model_settings["FIXED_RELATIVE_PROPORTIONS_SPEC"]
    )

    persons_merged = persons_merged.to_frame()

    # add tour-based chunk_id so we can chunk all trips in tour together
    assert "chunk_id" not in persons_merged.columns
    unique_household_ids = persons_merged.household_id.unique()
    household_chunk_ids = pd.Series(
        range(len(unique_household_ids)), index=unique_household_ids
    )
    persons_merged["chunk_id"] = reindex(
        household_chunk_ids, persons_merged.household_id
    )

    constants = config.get_model_constants(model_settings)

    cdap_interaction_coefficients = cdap.preprocess_interaction_coefficients(
        cdap_interaction_coefficients
    )

    # specs are built just-in-time on demand and cached as injectables
    # prebuilding here allows us to write them to the output directory
    # (also when multiprocessing locutor might not see all household sizes)
    logger.info("Pre-building cdap specs")
    for hhsize in range(2, cdap.MAX_HHSIZE + 1):
        spec = cdap.build_cdap_spec(cdap_interaction_coefficients, hhsize, cache=True)
        if inject.get_injectable("locutor", False):
            spec.to_csv(
                config.output_file_path("cdap_spec_%s.csv" % hhsize), index=True
            )

    if estimator:
        estimator.write_model_settings(model_settings, "cdap.yaml")
        estimator.write_spec(model_settings, tag="INDIV_AND_HHSIZE1_SPEC")
        estimator.write_spec(
            model_settings=model_settings, tag="FIXED_RELATIVE_PROPORTIONS_SPEC"
        )
        estimator.write_coefficients(coefficients_df, model_settings)
        estimator.write_table(
            cdap_interaction_coefficients,
            "interaction_coefficients",
            index=False,
            append=False,
        )
        estimator.write_choosers(persons_merged)
        for hhsize in range(2, cdap.MAX_HHSIZE + 1):
            spec = cdap.get_cached_spec(hhsize)
            estimator.write_table(spec, "spec_%s" % hhsize, append=False)

    logger.info("Running cdap_simulate with %d persons", len(persons_merged.index))

    choices = cdap.run_cdap(
        persons=persons_merged,
        person_type_map=person_type_map,
        cdap_indiv_spec=cdap_indiv_spec,
        cdap_interaction_coefficients=cdap_interaction_coefficients,
        cdap_fixed_relative_proportions=cdap_fixed_relative_proportions,
        locals_d=constants,
        chunk_size=chunk_size,
        trace_hh_id=trace_hh_id,
        trace_label=trace_label,
    )

    if estimator:
        estimator.write_choices(choices)
        choices = estimator.get_survey_values(choices, "persons", "cdap_activity")
        estimator.write_override_choices(choices)
        estimator.end_estimation()

    # - assign results to persons table and annotate
    persons = persons.to_frame()

    choices = choices.reindex(persons.index)
    persons["cdap_activity"] = choices

    expressions.assign_columns(
        df=persons,
        model_settings=model_settings.get("annotate_persons"),
        trace_label=tracing.extend_trace_label(trace_label, "annotate_persons"),
    )

    pipeline.replace_table("persons", persons)

    # - annotate households table
    households = households.to_frame()
    expressions.assign_columns(
        df=households,
        model_settings=model_settings.get("annotate_households"),
        trace_label=tracing.extend_trace_label(trace_label, "annotate_households"),
    )
    pipeline.replace_table("households", households)

    tracing.print_summary("cdap_activity", persons.cdap_activity, value_counts=True)
    logger.info(
        "cdap crosstabs:\n%s"
        % pd.crosstab(persons.ptype, persons.cdap_activity, margins=True)
    )
