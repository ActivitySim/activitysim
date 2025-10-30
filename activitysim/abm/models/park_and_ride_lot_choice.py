# ActivitySim
# See full license in LICENSE.txt.
import logging

import numpy as np
import pandas as pd
from typing import Literal

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
from activitysim.abm.models.util.park_and_ride_capacity import ParkAndRideCapacity

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

    LANDUSE_COL_FOR_PNR_ELIGIBLE_DEST: str | None = None
    """The column name in the land use table that indicates whether a destination is eligible for park-and-ride.
    If supplied, then TRANSIT_SKIMS_FOR_ELIGIBILITY is not used.
    """

    explicit_chunk: float = 0
    """
    If > 0, use this chunk size instead of adaptive chunking.
    If less than 1, use this fraction of the total number of rows.
    """

    alts_preprocessor: PreprocessorSettings | None = None
    """Preprocessor settings for park-and-ride lot alternatives."""

    ITERATE_WITH_TOUR_MODE_CHOICE: bool = False
    """If True, iterate with tour mode choice to find un-capacitated pnr lots. """

    MAX_ITERATIONS: int = 5
    """Maximum number of iterations to iterate park-and-ride choice with tour mode choice."""

    ACCEPTED_TOLERANCE: float = 0.95
    """Lot is considered full if the lot is at least this percentage full."""

    RESAMPLE_STRATEGY: Literal["latest", "random"] = "latest"
    """Strategy to use when selecting tours for resampling park-and-ride lot choices.
    - latest: tours arriving the latest are selected for resampling.
    - random: randomly resample from all tours at the over-capacitated lot.
    """

    PARK_AND_RIDE_MODES: list[str] | None = None
    """List of modes that are considered park-and-ride modes.
    Needed for filtering choices to calculate park-and-ride lot capacities.
    Should correspond to the columns in the tour mode choice specification file.
    """

    TRACE_PNR_CAPACITIES_PER_ITERATION: bool = True
    """If True, output park-and-ride lot occupancy at each iteration to the trace folder."""


def filter_chooser_to_transit_accessible_destinations(
    state: workflow.State,
    choosers: pd.DataFrame,
    land_use: pd.DataFrame,
    pnr_alts: pd.DataFrame,
    network_los: los.Network_LOS,
    model_settings: ParkAndRideLotChoiceSettings,
    choosers_dest_col_name: str,
) -> pd.DataFrame:
    """
    Filter choosers to only those with destinations that have transit access.
    If precomputed landuse column is supplied, use that.
    Otherwise look at the skims and check the destination has any non-zero terms for transit access.
    If no landuse column or skim cores supplied, return all choosers.
    """

    if model_settings.LANDUSE_COL_FOR_PNR_ELIGIBLE_DEST is not None:
        col = model_settings.LANDUSE_COL_FOR_PNR_ELIGIBLE_DEST
        assert (
            col in land_use.columns
        ), f"{col} not in landuse table, check LANDUSE_COL_FOR_PNR_ELIGIBLE_DEST setting in park_and_ride_lot_choice.yaml"
        available_dests = land_use[land_use[col]].index
        filtered_choosers = choosers[
            choosers[choosers_dest_col_name].isin(available_dests)
        ]

    elif model_settings.TRANSIT_SKIMS_FOR_ELIGIBILITY is not None:

        skim_dict = network_los.get_default_skim_dict()
        unique_destinations = choosers[choosers_dest_col_name].unique()
        unique_lot_locations = pnr_alts.index.values
        transit_accessible = np.full(
            shape=len(unique_destinations), fill_value=False, dtype=bool
        )

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
            transit_accessible_i = [
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
            transit_accessible = np.logical_or(transit_accessible, transit_accessible_i)

        eligible_destinations = unique_destinations[transit_accessible]
        filtered_choosers = choosers[
            choosers[choosers_dest_col_name].isin(eligible_destinations)
        ]
    else:
        filtered_choosers = choosers

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
    choosers_origin_col_name: str = "home_zone_id",
    estimator=None,
    model_settings_file_name: str = "park_and_ride_lot_choice.yaml",
    pnr_capacity_cls: ParkAndRideCapacity | None = None,
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

    # if we are running with capacitated pnr lots, we need to flag the lots that are over-capacitated
    if pnr_capacity_cls is not None:
        pnr_alts["pnr_lot_full"] = pnr_capacity_cls.flag_capacitated_pnr_zones(pnr_alts)
        # if there are no available pnr lots left, we return a series of -1
        if (pnr_alts["pnr_lot_full"] == 1).all():
            logger.info(
                "All park-and-ride lots are full. Returning -1 as park-and-ride lot choice."
            )
            return pd.Series(data=-1, index=choosers.index)
    else:
        pnr_alts["pnr_lot_full"] = 0

    original_index = None
    if not choosers.index.is_unique:
        # non-unique index will crash interaction_simulate
        # so we need to reset the index and add it to ActivitySim's rng
        # this happens while the disaggregate accessibility model is running pnr lot choice
        original_index = choosers.index
        oi_name = original_index.name
        oi_name = oi_name if oi_name else "index"
        choosers = choosers.reset_index(drop=False)
        idx_multiplier = choosers.groupby(oi_name).size().max()
        # round to the nearest 10's place
        idx_multiplier = int(np.ceil(idx_multiplier / 10.0) * 10)
        choosers.index = (
            original_index * idx_multiplier + choosers.groupby(oi_name).cumcount()
        )
        choosers.index.name = oi_name + "_pnr_lot_choice"
        state.get_rn_generator().add_channel("pnr_lot_choice", choosers)
        assert (
            choosers.index.is_unique
        ), "The index of the choosers DataFrame must be unique after resetting the index."
        assert len(choosers.index) == len(original_index), (
            f"The length of the choosers DataFrame must be equal to the original index length:"
            f" {len(choosers.index)} != {len(original_index)}"
        )

    trn_accessible_choosers = filter_chooser_to_transit_accessible_destinations(
        state,
        choosers,
        land_use,
        pnr_alts,
        network_los,
        model_settings,
        choosers_dest_col_name,
    )

    if trn_accessible_choosers.empty:
        logger.debug(
            "No choosers with transit accessible destinations found. Returning -1 as park-and-ride lot choice."
        )
        # need to drop rng channel that we created before trn_accessible_choosers
        state.get_rn_generator().drop_channel("pnr_lot_choice")
        index = choosers.index if original_index is None else original_index
        return pd.Series(data=-1, index=index)

    add_periods = False if "in_period" in trn_accessible_choosers.columns else True
    skims = logsums.setup_skims(
        network_los,
        trn_accessible_choosers,
        add_periods=add_periods,
        include_pnr_skims=True,
        orig_col_name=choosers_origin_col_name,
        dest_col_name=choosers_dest_col_name,
    )
    locals_dict.update(skims)

    if model_settings.preprocessor:
        # Need to check whether the table exists in the state.
        # This can happen if you have preprocessor settings that reference tours
        # but the tours table doesn't exist yet because you are calculating logsums.
        # Expressions using these tables need to be written with short-circuiting conditionals
        # to avoid errors when the table is not present.
        tables = model_settings.preprocessor.TABLES.copy()
        for table_name in model_settings.preprocessor.TABLES:
            if table_name not in state.existing_table_names:
                logger.debug(
                    f"Table '{table_name}' not found in state. "
                    "Removing table from preprocessor list."
                )
                tables.remove(table_name)
        model_settings.preprocessor.TABLES = tables

    # preprocess choosers
    expressions.annotate_preprocessors(
        state,
        df=trn_accessible_choosers,
        locals_dict=locals_dict,
        # not including skims because lot alt destination not in chooser table
        # they are included through the locals_dict instead
        skims=None,
        model_settings=model_settings,
        trace_label=trace_label,
    )

    # preprocess alternatives
    expressions.annotate_preprocessors(
        state,
        df=pnr_alts,
        locals_dict=locals_dict,
        skims=None,
        model_settings=model_settings,
        trace_label=trace_label,
        preprocessor_setting_name="alts_preprocessor",
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

    if original_index is not None:
        # set the choices index back to the original index
        choices.index = original_index
        state.get_rn_generator().drop_channel("pnr_lot_choice")

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
        choosers_origin_col_name="home_zone_id",
        estimator=estimator,
        pnr_capacity_cls=None,
        trace_label=trace_label,
    )

    choices = choices.reindex(tours.index, fill_value=-1)

    tours["pnr_zone_id"] = choices

    state.add_table("tours", tours)

    if trace_hh_id:
        state.tracing.trace_df(tours, label=trace_label, warn_if_empty=True)
