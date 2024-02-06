# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging

import pandas as pd

from activitysim.abm.models.util import tour_od
from activitysim.core import estimation, los, workflow

logger = logging.getLogger(__name__)


@workflow.step
def tour_od_choice(
    state: workflow.State,
    tours: pd.DataFrame,
    persons: pd.DataFrame,
    households: pd.DataFrame,
    land_use: pd.DataFrame,
    network_los: los.Network_LOS,
    model_settings: tour_od.TourODSettings | None = None,
    model_settings_file_name: str = "tour_od_choice.yaml",
    trace_label: str = "tour_od_choice",
) -> None:
    """Simulates joint origin/destination choice for all tours.

    Given a set of previously generated tours, each tour needs to have an
    origin and a destination. In this case tours are the choosers, but
    the associated person that's making the tour does not necessarily have
    a home location assigned already. So we choose a tour origin at the same
    time as we choose a tour destination, and assign the tour origin as that
    person's home location.

    Parameters
    ----------
    tours : pd.DataFrame
        lazy-loaded tours table
    persons : pd.DataFrame
        lazy-loaded persons table
    households : pd.DataFrame
        lazy-loaded households table
    land_use : pd.DataFrame
        lazy-loaded land use data table
    network_los : los.Network_LOS
        lazy-loaded activitysim.los.Network_LOS object
    """
    if model_settings is None:
        model_settings = tour_od.TourODSettings.read_settings_file(
            state.filesystem,
            model_settings_file_name,
        )
    origin_col_name = model_settings.ORIG_COL_NAME
    dest_col_name = model_settings.DEST_COL_NAME
    alt_id_col = tour_od.get_od_id_col(origin_col_name, dest_col_name)
    trace_hh_id = state.settings.trace_hh_id
    chunk_size = state.settings.chunk_size

    sample_table_name = model_settings.OD_CHOICE_SAMPLE_TABLE_NAME
    want_sample_table = (
        state.settings.want_dest_choice_sample_tables and sample_table_name is not None
    )

    logsum_column_name = model_settings.OD_CHOICE_LOGSUM_COLUMN_NAME
    want_logsums = logsum_column_name is not None

    # interaction_sample_simulate insists choosers appear in same order as alts
    tours = tours.sort_index()

    estimator = estimation.manager.begin_estimation(state, "tour_od_choice")
    if estimator:
        estimator.write_coefficients(model_settings=model_settings)
        estimator.write_spec(file_name=model_settings.SAMPLE_SPEC, tag="SAMPLE_SPEC")
        estimator.write_spec(file_name=model_settings.SPEC, tag="SPEC")
        estimator.set_alt_id(alt_id_col)
        estimator.write_table(
            state.get_injectable("size_terms"), "size_terms", append=False
        )
        estimator.write_table(state.get_dataframe("land_use"), "landuse", append=False)
        estimator.write_model_settings(model_settings, model_settings_file_name)

    choices_df, save_sample_df = tour_od.run_tour_od(
        state,
        tours,
        persons,
        want_logsums,
        want_sample_table,
        model_settings,
        network_los,
        estimator,
        chunk_size,
        trace_hh_id,
        trace_label,
    )

    if estimator:
        assert estimator.want_unsampled_alternatives
        estimator.write_choices(choices_df.choice)
        survey_od = estimator.get_survey_values(
            choices_df.choice, "tours", [origin_col_name, dest_col_name]
        )
        choices_df[origin_col_name] = survey_od[origin_col_name]
        choices_df[dest_col_name] = survey_od[dest_col_name]
        survey_od[alt_id_col] = tour_od.create_od_id_col(
            survey_od, origin_col_name, dest_col_name
        )
        choices_df.choice = survey_od[alt_id_col]
        estimator.write_override_choices(choices_df.choice)
        estimator.end_estimation()

    tours[origin_col_name] = choices_df[origin_col_name].reindex(tours.index)
    tours[dest_col_name] = choices_df[dest_col_name].reindex(tours.index)
    if want_logsums:
        tours[logsum_column_name] = (
            choices_df["logsum"].reindex(tours.index).astype("float")
        )
    tours["poe_id"] = tours[origin_col_name].map(land_use.poe_id)

    households[origin_col_name] = tours.set_index("household_id")[
        origin_col_name
    ].reindex(households.index)
    persons[origin_col_name] = (
        households[origin_col_name].reindex(persons.household_id).values
    )

    # Downstream steps require that persons and households have a 'home_zone_id'
    # column. We assume that if the tour_od_choice model is used, this field is
    # missing from the population data, so it gets inherited from the tour origin
    households["home_zone_id"] = households[origin_col_name]
    persons["home_zone_id"] = persons[origin_col_name]

    state.add_table("tours", tours)
    state.add_table("persons", persons)
    state.add_table("households", households)

    if want_sample_table:
        assert len(save_sample_df.index.get_level_values(0).unique()) == len(choices_df)
        state.extend_table(sample_table_name, save_sample_df)

    if trace_hh_id:
        state.tracing.trace_df(
            tours,
            label="tours_od_choice",
            slicer="person_id",
            index_label="tour",
            columns=None,
            warn_if_empty=True,
        )
