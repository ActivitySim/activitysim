# ActivitySim
# See full license in LICENSE.txt.
import logging

import numpy as np
import pandas as pd

from activitysim.core import (
    config,
    expressions,
    los,
    estimation,
    simulate,
    tracing,
    workflow,
)
from activitysim.core.configuration.logit import (
    LogitComponentSettings,
    PreprocessorSettings,
)
from activitysim.core.interaction_simulate import interaction_simulate
from activitysim.abm.models.util import logsums


logger = logging.getLogger(__name__)


class ParkAndRideLotChoiceSettings(LogitComponentSettings, extra="forbid"):
    """
    Settings for the `external_identification` component.
    """

    LANDUSE_PNR_SPACES_COLUMN: str
    """lists the column name in the land use table that contains the number of park-and-ride spaces available in the zone"""

    TRANSIT_SKIMS_FOR_ELIGIBILITY: list[str] | None = None
    """A list of skim names to use for filtering choosers to only those with destinations that have transit access.
    If None, all tours are considered eligible for park-and-ride lot choice."""

    explicit_chunk: float = 0
    """
    If > 0, use this chunk size instead of adaptive chunking.
    If less than 1, use this fraction of the total number of rows.
    """

    preprocessor: PreprocessorSettings | None = None
    """FIXME preprocessor can be removed once preprocessor / annotator work is pulled in."""

    # FIXME need to add alts preprocessor as well


def filter_chooser_to_transit_accessible_destinations(
    state: workflow.State,
    choosers: pd.DataFrame,
    pnr_alts: pd.DataFrame,
    network_los: los.Network_LOS,
    model_settings: ParkAndRideLotChoiceSettings,
    choosers_dest_col_name: str,
) -> pd.DataFrame:
    """
    Filter choosers to only those with destinations that have transit access.
    We look at the skims and check the destination has any non-zero terms for transit access.
    We get the skims to check from the model settings.
    """
    # all choosers are eligible if transit skims are not provided
    if model_settings.TRANSIT_SKIMS_FOR_ELIGIBILITY is None:
        logger.info(
            "No transit skims provided for park-and-ride lot choice model. All tours are eligible."
        )
        return choosers

    skim_dict = network_los.get_default_skim_dict()
    unique_destinations = choosers[choosers_dest_col_name].unique()
    unique_lot_locations = pnr_alts.index.values

    for skim_name in model_settings.TRANSIT_SKIMS_FOR_ELIGIBILITY:
        if "__" in skim_name:
            # If the skim name contains '__', it is a 3D skim
            # we need to pass the skim name as a tuple to the lookup method, e.g. ('WALK_TRANSIT_IVTT', 'MD')
            skim_name = tuple(skim_name.split("__"))
        if skim_name not in skim_dict.skim_info.omx_keys.keys():
            raise ValueError(
                f"Skim '{skim_name}' not found in the skim dictionary."
                "Please update the model setting TRANSIT_SKIMS_FOR_ELIGIBILITY with valid skim names."
            )
        # Filter choosers to only those with destinations that have transit access
        # want to check whether ANY of the lot locations have transit access to EVERY destination
        transit_accessible = [
            (
                skim_dict.lookup(
                    unique_lot_locations,
                    np.full(shape=len(unique_lot_locations), fill_value=dest),
                    skim_name,
                )
                > 0
            ).any()
            for dest in unique_destinations
        ]

    eligible_destinations = unique_destinations[transit_accessible]
    filtered_choosers = choosers[
        choosers[choosers_dest_col_name].isin(eligible_destinations)
    ]

    logger.info(
        f"Preparing choosers for park-and-ride lot choice model:\n"
        f" Filtered tours to {len(filtered_choosers)} with transit access to their destination.\n"
        f" Total number of tours: {len(choosers)}.\n"
        f" Percentage of tours with transit access at destination: "
        f"{len(filtered_choosers) / len(choosers) * 100:.2f}%"
    )

    return filtered_choosers


def run_park_and_ride_lot_choice(
    state: workflow.State,
    choosers: pd.DataFrame,
    land_use: pd.DataFrame,
    network_los: los.Network_LOS,
    model_settings: ParkAndRideLotChoiceSettings | None = None,
    choosers_dest_col_name: str = "destination",
    estimator=None,
    model_settings_file_name: str = "park_and_ride_lot_choice.yaml",
    trace_label: str = "park_and_ride_lot_choice",
) -> None:
    """
    Run the park-and-ride lot choice model.

    Provides another entry point for the model which is useful for getting pnr locations while creating logsums.
    """

    if model_settings is None:
        model_settings = ParkAndRideLotChoiceSettings.read_settings_file(
            state.filesystem,
            "park_and_ride_lot_choice.yaml",
        )

    spec = state.filesystem.read_model_spec(file_name=model_settings.SPEC)
    coefficients = state.filesystem.read_model_coefficients(model_settings)
    model_spec = simulate.eval_coefficients(state, spec, coefficients, estimator)
    locals_dict = model_settings.CONSTANTS

    pnr_alts = land_use[land_use[model_settings.LANDUSE_PNR_SPACES_COLUMN] > 0]
    pnr_alts["pnr_zone_id"] = pnr_alts.index.values

    trn_accessible_choosers = filter_chooser_to_transit_accessible_destinations(
        state,
        choosers,
        pnr_alts,
        network_los,
        model_settings,
        choosers_dest_col_name,
    )

    add_periods = False if "in_period" in trn_accessible_choosers.columns else True
    skims = logsums.setup_skims(
        network_los,
        trn_accessible_choosers,
        add_periods=add_periods,
        include_pnr_skims=True,
        dest_col_name=choosers_dest_col_name,
    )
    locals_dict.update(skims)

    if trn_accessible_choosers.index.name == "proto_person_id":
        # if the choosers are indexed by proto_person_id, we need to reset the index
        # so that interaction_simulate has a unique index to work with
        trn_accessible_choosers = trn_accessible_choosers.reset_index()

    # FIXME: add alts preprocessors
    expressions.annotate_preprocessors(
        state,
        df=trn_accessible_choosers,
        locals_dict=locals_dict,
        # not including skims because lot alt destination not in chooser table
        # they are included through the locals_dict instead
        skims={},
        model_settings=model_settings,
        trace_label=trace_label,
    )

    if estimator:
        estimator.write_coefficients(model_settings=model_settings)
        estimator.write_coefficients_template(model_settings=model_settings)
        estimator.write_spec(model_settings)
        estimator.write_model_settings(model_settings, model_settings_file_name)
        # in production, all choosers with a transit accessible destination are selected as choosers
        # but in estimation, it would only be those who actually reported a pnr lot
        # unclear exactly how to handle this, but for now, we will write all choosers
        estimator.write_choosers(trn_accessible_choosers)
        estimator.write_alternatives(pnr_alts)

    choices = interaction_simulate(
        state,
        choosers=trn_accessible_choosers,
        alternatives=pnr_alts,
        spec=model_spec,
        skims=skims,
        log_alt_losers=state.settings.log_alt_losers,
        locals_d=locals_dict,
        trace_label=trace_label,
        trace_choice_name=trace_label,
        estimator=estimator,
        explicit_chunk_size=model_settings.explicit_chunk,
        compute_settings=model_settings.compute_settings,
    )

    choices = choices.reindex(choosers.index, fill_value=-1)

    if "proto_person_id" in trn_accessible_choosers.columns:
        # if the choosers are indexed by proto_person_id, we need to set the index to the original index
        choices.index = trn_accessible_choosers["proto_person_id"]

    if estimator:
        # careful -- there could be some tours in estimation data that are not transit accessible
        # but still reported a pnr location
        # warning! untested code!
        estimator.write_choices(choices)
        choices = estimator.get_survey_values(choices, "tours", "pnr_zone_id")
        estimator.write_override_choices(choices)
        estimator.end_estimation()

    return choices


@workflow.step
def park_and_ride_lot_choice(
    state: workflow.State,
    tours: pd.DataFrame,
    tours_merged: pd.DataFrame,
    land_use: pd.DataFrame,
    network_los: los.Network_LOS,
    model_settings: ParkAndRideLotChoiceSettings | None = None,
    model_settings_file_name: str = "park_and_ride_lot_choice.yaml",
    trace_label: str = "park_and_ride_lot_choice",
    trace_hh_id: bool = False,
) -> None:
    """
    This model predicts which lot location would be used for a park-and-ride tour.
    """
    if model_settings is None:
        model_settings = ParkAndRideLotChoiceSettings.read_settings_file(
            state.filesystem,
            model_settings_file_name,
        )

    estimator = estimation.manager.begin_estimation(state, "park_and_ride_lot_choice")

    choices = run_park_and_ride_lot_choice(
        state,
        choosers=tours_merged,
        land_use=land_use,
        network_los=network_los,
        model_settings=model_settings,
        choosers_dest_col_name="destination",
        estimator=estimator,
        trace_label=trace_label,
    )

    choices = choices.reindex(tours.index, fill_value=-1)

    tours["pnr_zone_id"] = choices

    state.add_table("tours", tours)

    if trace_hh_id:
        state.tracing.trace_df(tours, label=trace_label, warn_if_empty=True)
