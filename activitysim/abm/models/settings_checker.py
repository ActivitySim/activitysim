import logging
from pandas import DataFrame
from pydantic import BaseModel as PydanticBase
from typing import Type

from activitysim.core.workflow import State
from activitysim.core.simulate import eval_coefficients, eval_nest_coefficients

# import model settings
from activitysim.abm.models.accessibility import AccessibilitySettings
from activitysim.abm.models.atwork_subtour_frequency import AtworkSubtourFrequencySettings
from activitysim.abm.models.auto_ownership import AutoOwnershipSettings
from activitysim.abm.models.cdap import CdapSettings
from activitysim.abm.models.disaggregate_accessibility import (
    DisaggregateAccessibilitySettings, 
    read_disaggregate_accessibility_yaml,
)
from activitysim.abm.models.free_parking import FreeParkingSettings
from activitysim.abm.models.initialize import InitializeTableSettings
from activitysim.abm.models.joint_tour_composition import JointTourCompositionSettings
from activitysim.abm.models.joint_tour_frequency_composition import JointTourFreqCompSettings
from activitysim.abm.models.joint_tour_frequency import JointTourFrequencySettings
from activitysim.abm.models.joint_tour_participation import JointTourParticipationSettings
from activitysim.abm.models.mandatory_tour_frequency import MandatoryTourFrequencySettings
from activitysim.abm.models.parking_location_choice import ParkingLocationSettings
from activitysim.abm.models.school_escorting import SchoolEscortSettings
from activitysim.abm.models.stop_frequency import StopFrequencySettings
from activitysim.abm.models.summarize import SummarizeSettings
from activitysim.abm.models.telecommute_frequency import TelecommuteFrequencySettings
from activitysim.abm.models.tour_scheduling_probabilistic import TourSchedulingProbabilisticSettings
from activitysim.abm.models.transit_pass_ownership import TransitPassOwnershipSettings
from activitysim.abm.models.transit_pass_subsidy import TransitPassSubsidySettings
from activitysim.abm.models.trip_departure_choice import TripDepartureChoiceSettings
from activitysim.abm.models.trip_destination import TripDestinationSettings
from activitysim.abm.models.trip_matrices import WriteTripMatricesSettings

# import util settings
from activitysim.abm.models.util.vectorize_tour_scheduling import (
    TourSchedulingSettings, 
)
from activitysim.abm.models.util.tour_od import (
    TourODSettings
)

# import logit model settings
from activitysim.core.configuration.logit import (
    TourLocationComponentSettings,
    TourModeComponentSettings
)

logger = logging.getLogger(__name__)
file_logger = logger.getChild("logfile")

COMPONENTS_TO_SETTINGS = {
    "compute_accessibility": {
        "settings_cls": AccessibilitySettings,
        "settings_file": "accessibility.yaml",
    },
    "atwork_subtour_destination": {
        "settings_cls": TourLocationComponentSettings,
        "settings_file": "atwork_subtour_destination.yaml",
    },
    "atwork_subtour_frequency": {
        "settings_cls": AtworkSubtourFrequencySettings,
        "settings_file": "atwork_subtour_frequency.yaml",
    },
    "atwork_subtour_mode_choice": {
        "settings_cls": TourModeComponentSettings,
        "settings_file": "tour_mode_choice.yaml",
    },
    "atwork_subtour_scheduling": {
        "settings_cls": TourSchedulingSettings,
        "settings_file": "tour_scheduling_atwork.yaml" 
    },
    "auto_ownership_simulate": {
        "settings_cls": AutoOwnershipSettings,
        "settings_file": "auto_ownership.yaml" 
    },
    "cdap_simulate": {
        "settings_cls": CdapSettings,
        "settings_file": "cdap.yaml"
    },
    "compute_disaggregate_accessibility": {
        "settings_cls": DisaggregateAccessibilitySettings,
        "settings_file": "disaggregate_accessibility.yaml"
    }, # TODO: needs testing with further models
    "free_parking": {
        "settings_cls": FreeParkingSettings,
        "settings_file": "free_parking.yaml"
    },
    "initialize_landuse": {
        "settings_cls": InitializeTableSettings,
        "settings_file": "initialize_landuse.yaml"
    },
    "joint_tour_composition": {
        "settings_cls": JointTourCompositionSettings,
        "settings_file": "joint_tour_composition.yaml"
    },
    "joint_tour_destination": {
        "settings_cls": TourLocationComponentSettings,
        "settings_file": "joint_tour_destination.yaml"
    },
    "joint_tour_frequency_composition": {
        "settings_cls": JointTourFreqCompSettings,
        "settings_file": "joint_tour_frequency_composition.yaml"
    },
    "joint_tour_frequency": {
        "settings_cls": JointTourFrequencySettings,
        "settings_file": "joint_tour_frequency.yaml"
    },
     "joint_tour_participation": {
        "settings_cls": JointTourParticipationSettings,
        "settings_file": "joint_tour_participation.yaml"
    },
    "joint_tour_scheduling": {
        "settings_cls": TourSchedulingSettings,
        "settings_file": "joint_tour_scheduling.yaml"
    },
    "workplace_location": {
        "settings_cls": TourLocationComponentSettings,
        "settings_file": "workplace_location.yaml"
    },
    "school_location": {
        "settings_cls": TourLocationComponentSettings,
        "settings_file": "school_location.yaml"
    },
    "mandatory_tour_frequency": {
        "settings_cls": MandatoryTourFrequencySettings,
        "settings_file": "mandatory_tour_frequency.yaml"
    },
    "non_mandatory_tour_destination": {
        "settings_cls": TourLocationComponentSettings,
        "settings_file": "non_mandatory_tour_destination.yaml"
    },
    "parking_location": {
        "settings_cls": ParkingLocationSettings,
        "settings_file": "parking_location_choice.yaml"
    },
    "school_escorting": {
        "settings_cls": SchoolEscortSettings,
        "settings_file": "school_escorting.yaml"
    },
    "stop_frequency": {
        "settings_cls": StopFrequencySettings,
        "settings_file": "stop_frequency.yaml"
    }, # NOTE: Stop Frequency requires a separate check (Not Implemented) because of NESTED_SPEC
    "summarize": {
        "settings_cls": SummarizeSettings,
        "settings_file": "summarize.yaml"
    },
    "telecommute_frequency": {
        "settings_cls": TelecommuteFrequencySettings,
        "settings_file": "telecommute_frequency.yaml"
    },
    "tour_mode_choice_simulate": {
        "settings_cls": TourModeComponentSettings,
        "settings_file": "tour_mode_choice.yaml"
    },
    "tour_od_choice": {
        "settings_cls": TourODSettings,
        "settings_file": "tour_od_choice.yaml"
    },
    "tour_scheduling_probabilistic": {
        "settings_cls": TourSchedulingProbabilisticSettings,
        "settings_file": "tour_scheduling_probabilistic.yaml"
    },
    "transit_pass_ownership": {
        "settings_cls": TransitPassOwnershipSettings,
        "settings_file": "transit_pass_ownership.yaml"
    },
    "transit_pass_subsidy": {
        "settings_cls": TransitPassSubsidySettings,
        "settings_file": "transit_pass_subsidy.yaml"
    },
    "trip_departure_choice": {
        "settings_cls": TripDepartureChoiceSettings,
        "settings_file": "trip_departure_choice.yaml"
    },
    "trip_destination": {
        "settings_cls": TripDestinationSettings,
        "settings_file": "trip_destination.yaml"
    },
    "write_trip_matrices": {
        "settings_cls": WriteTripMatricesSettings,
        "settings_file": "write_trip_matrices.yaml"
    },
}


def try_load_model_settings(
    model_name: str,
    model_settings_class: Type[PydanticBase],
    model_settings_file: str,
    state: State,
) -> PydanticBase:
    logger.info(
        f"Attempting to load model settings for {model_name} via {model_settings_class.__name__} and {model_settings_file}"
    )
    if isinstance(model_settings_class, DisaggregateAccessibilitySettings):
        settings = read_disaggregate_accessibility_yaml(state, model_settings_file)
    else:
        settings = model_settings_class.read_settings_file(
            state.filesystem, model_settings_file
        )
    logger.info(f"Successfully loaded model settings from {model_settings_file}")
    return settings


def try_load_spec(
    model_name: str, model_settings: PydanticBase, state: State
) -> DataFrame:
    
    if isinstance(model_settings, CdapSettings):
        # debug
        pass

    logger.info(
        f"Attempting to load SPEC for {model_name} via {model_settings.__class__.__name__}"
    )
    spec_file = model_settings.model_dump().get("SPEC")
    if spec_file is None:
        logger.info(
            f"No SPEC file is associated with {model_settings.__class__.__name__}"
        )
        # Need to return object with .copy() interface for eval_coefficients
        return DataFrame()
    spec = state.filesystem.read_model_spec(spec_file)
    logger.info(f"Successfully loaded model SPEC from {spec_file}")
    return spec


def try_load_and_eval_coefs(
    model_name: str, model_settings: PydanticBase, spec: DataFrame, state: State
) -> tuple[DataFrame | None, DataFrame | None]:
    logger.info(
        f"Attempting to load coefficients for {model_name} via {model_settings.__class__.__name__}"
    )
    # Assumes that if COEFFICIENTS is required, will be validated by Pydantic
    if model_settings.model_dump().get("COEFFICIENTS"):
        coefs_file = model_settings.COEFFICIENTS
        coefs = state.filesystem.read_model_coefficients(model_settings)
        # check whether coefficients should be evaluated as NESTS or not
        if model_settings.model_dump().get("NESTS"):
            # Proper Trace label is probably unneeded here
            # NOTE: is this the correct way to invoke this?
            eval_coefs = eval_nest_coefficients(
                model_settings.NESTS, coefs, trace_label=None
            )
        else:
            eval_coefs = eval_coefficients(state, spec, coefs, estimator=None)
        logger.info(f"Successfully read and evaluated coefficients from {coefs_file}")
        return coefs, eval_coefs
    else:
        logger.info(
            f"No coefficients file is associated with {model_settings.__class__.__name__}"
        )
    return None, None


def try_load_and_check(
    model_name: str,
    model_settings_class: Type[PydanticBase],
    model_settings_file: str,
    state: State,
) -> None:

    # first, attempt to load settings
    settings = try_load_model_settings(
        model_name=model_name,
        model_settings_class=model_settings_class,
        model_settings_file=model_settings_file,
        state=state,
    )

    # then, attempt to read SPEC file
    spec = try_load_spec(model_name=model_name, model_settings=settings, state=state)

    # then, attempt to read and evaluate coefficients
    coefs, eval_coefs = try_load_and_eval_coefs(
        model_name=model_name, model_settings=settings, spec=spec, state=state
    )

    # then, check preprocessors if any
    # for now, check is limited to check that the SPEC file is loadable
    if settings.model_dump().get("preprocessor"):
        preprocessor_settings = settings.preprocessor
        spec = try_load_spec(
            model_name=model_name + ": preprocessor",
            model_settings=preprocessor_settings,
            state=state,
        )


def check_model_settings(state: State) -> None:

    components = state.settings.models  # _RUNNABLE_STEPS.keys() may be better?

    for c in components:

        # TODO: this check allows incremental development, but should be deleted.
        if not c in COMPONENTS_TO_SETTINGS:
            logger.info(
                f"Cannot pre-check settings for model component {c}: mapping to a Pydantic data model is undefined in the checker."
            )
            continue

        # first, attempt to load the model settings
        settings_cls = COMPONENTS_TO_SETTINGS[c]["settings_cls"]
        settings_file = COMPONENTS_TO_SETTINGS[c]["settings_file"]
        try_load_and_check(
            model_name=c,
            model_settings_class=settings_cls,
            model_settings_file=settings_file,
            state=state,
        )

    print("Settings Checker Complete")
