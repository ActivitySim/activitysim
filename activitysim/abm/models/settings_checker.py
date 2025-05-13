import logging
import os
from pandas import DataFrame
from pydantic import BaseModel as PydanticBase
from typing import Type, Optional

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
from activitysim.abm.models.non_mandatory_tour_frequency import NonMandatoryTourFrequencySettings
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
from activitysim.abm.models.trip_mode_choice import TripModeChoiceSettings
from activitysim.abm.models.trip_purpose_and_destination import TripPurposeAndDestinationSettings
from activitysim.abm.models.trip_purpose import TripPurposeSettings
from activitysim.abm.models.vehicle_allocation import VehicleAllocationSettings
from activitysim.abm.models.vehicle_type_choice import VehicleTypeChoiceSettings
from activitysim.abm.models.work_from_home import WorkFromHomeSettings

# import util settings
from activitysim.abm.models.util.vectorize_tour_scheduling import (
    TourSchedulingSettings, 
)
from activitysim.abm.models.util.tour_od import (
    TourODSettings
)

# import logit model settings
from activitysim.core.configuration.logit import (
    LogitNestSpec,
    TourLocationComponentSettings,
    TourModeComponentSettings,
)

# setup logging
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
    "initialize_households": {
        "settings_cls": InitializeTableSettings,
        "settings_file": "initialize_households.yaml"
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
    }, # NOTE: Mandatory Frequency requires a separate check (Not Implemented) because of NESTED_SPEC
    "mandatory_tour_scheduling": {
        "settings_cls": TourSchedulingSettings,
        "settings_file": "mandatory_tour_scheduling.yaml"
    },
    "non_mandatory_tour_destination": {
        "settings_cls": TourLocationComponentSettings,
        "settings_file": "non_mandatory_tour_destination.yaml"
    },
    "non_mandatory_tour_frequency": {
        "settings_cls": NonMandatoryTourFrequencySettings,
        "settings_file": "non_mandatory_tour_frequency.yaml"
    }, # NOTE: Non-mandatory Frequency requires a separate check (Not Implemented) because of NESTED_SPEC
    "non_mandatory_tour_scheduling": {
        "settings_cls": TourSchedulingSettings,
        "settings_file": "non_mandatory_tour_scheduling.yaml"
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
    "trip_mode_choice": {
        "settings_cls": TripModeChoiceSettings,
        "settings_file": "trip_mode_choice.yaml"
    },
    "trip_purpose_and_destination": {
        "settings_cls": TripPurposeAndDestinationSettings,
        "settings_file": "trip_purpose_and_destination.yaml"
    },
    "trip_purpose": {
        "settings_cls": TripPurposeSettings,
        "settings_file": "trip_purpose.yaml"
    },
    "vehicle_allocation": {
        "settings_cls": VehicleAllocationSettings,
        "settings_file": "vehicle_allocation.yaml"
    },
    "vehicle_type_choice": {
        "settings_cls": VehicleAllocationSettings,
        "settings_file": "vehicle_allocation.yaml"
    },
    "vehicle_type_choice": {
        "settings_cls": VehicleTypeChoiceSettings,
        "settings_file": "vehicle_type_choice.yaml"
    },
    "work_from_home": {
        "settings_cls": WorkFromHomeSettings,
        "settings_file": "work_from_home.yaml"
    },
}


def try_load_model_settings(
    model_name: str,
    model_settings_class: Type[PydanticBase],
    model_settings_file: str,
    state: State,
) -> tuple[Optional[PydanticBase], Optional[Exception]]:
    logger.info(
        f"Attempting to load model settings for {model_name} via {model_settings_class.__name__} and {model_settings_file}"
    )
    try:
        if isinstance(model_settings_class, DisaggregateAccessibilitySettings):
            model_settings = read_disaggregate_accessibility_yaml(state, model_settings_file)
        else:
            model_settings = model_settings_class.read_settings_file(
                state.filesystem, model_settings_file
            )
        result = model_settings, None
        logger.info(f"Successfully loaded model settings from {model_settings_file}")
    except Exception as e:
        result = None, e
    return result


def try_load_spec(
    model_name: str, 
    model_settings: PydanticBase, 
    spec_file: str, 
    state: State
) -> tuple[DataFrame, Optional[Exception]]:
    logger.info(
        f"Attempting to load SPEC for {model_name} via {model_settings.__class__.__name__}"
    )
    try:
        result = state.filesystem.read_model_spec(spec_file), None
        logger.info(f"Successfully loaded model SPEC from {spec_file}")
    except Exception as e:
        # always return a dataframe
        result = DataFrame(), None
        # raise e
    return result

def try_load_coefs(
    model_name: str, 
    model_settings: PydanticBase, 
    coefs_file: str, 
    state: State
) -> tuple[DataFrame, Optional[Exception]]:
    msg = f"Attempting to load COEFFICIENTS for {model_name} via {model_settings.__class__.__name__}"
    logger.info(msg)
    file_logger.info(msg)

    try:
        result = state.filesystem.read_model_coefficients(model_settings), None
        msg = f"Successfully loaded model Coefficients from {coefs_file}"
        logger.info(msg)
        file_logger.info(msg)
    except Exception as e:
        result = DataFrame(), e
        # raise e
    return result
    
def try_eval_spec_coefs(
    model_name: str, 
    model_settings: PydanticBase, 
    spec: DataFrame,
    coefs: DataFrame,
    state: State
) -> tuple[DataFrame, Optional[Exception]]:
        if model_name == "non_mandatory_tour_frequency":
            breakpoint
        
        try:
            # check whether coefficients should be evaluated as NESTS or not
            if model_settings.model_dump().get("NESTS"):
                if isinstance(model_settings.NESTS, LogitNestSpec):
                    nests = model_settings.NESTS
            else:
                nests = None
            if nests is not None:
                # Proper Trace label is probably unneeded here
                # TODO: Verify that this method is correct
                result = eval_nest_coefficients(
                    model_settings.NESTS, coefs, trace_label=None
                ), None
            else:
                if spec.empty:
                    msg = f"No SPEC available for {model_name}. " \
                          "Attempting to resolve coefficients against empty DataFrame, " \
                          "but errors may not be fully caught"
                    logger.warning(msg)
                    file_logger.warning(msg)
                result = eval_coefficients(state, spec, coefs, estimator=None), None
            msg = (f"Successfully evaluated coefficients for {model_name}")
            logger.info(msg)
            file_logger.info(msg)
        except Exception as e:
            result = DataFrame(), e
            # raise e
        return result

def try_load_and_check_spec_coefs(
    model_name: str,
    model_settings: Type[PydanticBase],
    state: State
) -> list[Exception]:
    # collect all errors
    errors = []

    # then, attempt to read SPEC file
    # only checks against the SPEC attr at top level of model.
    # checking specification files keyed against arbtrary attrs
    # is not currently supported - e.g. SchoolEscortingSettings: INBOUND_SPEC
    spec_file = model_settings.model_dump().get("SPEC")
    if spec_file:
        spec, spec_error = try_load_spec(
            model_name=model_name, 
            model_settings=model_settings,
            spec_file=spec_file,
            state=state
        )
    else:
        spec, spec_error = DataFrame(), None
        msg = f"No SPEC file is associated with {model_settings.__class__.__name__}"
        logger.info(msg)
        file_logger.info(msg)
    
    if spec_error is not None:
        errors.append(spec_error)

    # then attempt to read coefficients
    coefs_file = model_settings.model_dump().get("COEFFICIENTS")
    if coefs_file:
        coefs, coefs_error = try_load_coefs(
            model_name=model_name,
            model_settings=model_settings,
            coefs_file=coefs_file,
            state=state
        )
    else:
        coefs, coefs_error = DataFrame(), None
        msg = f"No coefficients file is associated with {model_settings.__class__.__name__}"
        logger.info(msg)
        file_logger.info(msg)

    if coefs_error is not None:
        errors.append(coefs_error)

    # then attempt to evaluate coefficients against spec
    if not coefs.empty:
        eval_coefs, eval_coefs_error = try_eval_spec_coefs(
            model_name=model_name,
            model_settings=model_settings,
            spec=spec,
            coefs=coefs,
            state=state
        )
    else:
        eval_coefs, eval_coefs_error = DataFrame(), None

    if eval_coefs_error is not None:
        errors.append(eval_coefs_error)

    # then, check preprocessors if any
    # for now, check is limited to check that the SPEC file is loadable
    if model_settings.model_dump().get("preprocessor"):
        preprocessor_settings = model_settings.preprocessor
        spec_file = preprocessor_settings.SPEC
        spec = try_load_spec(
            model_name=model_name + ": preprocessor",
            model_settings=preprocessor_settings,
            spec_file = spec_file,
            state=state
        )
    
    return errors


def check_model_settings(state: State) -> None:

    # Collect all errors
    all_errors = []

    # additional logging set up
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", 
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    out_log_file = state.get_log_file_path("settings_checker.log")
    if os.path.exists(out_log_file):
        os.remove(out_log_file)
    module_handler = logging.FileHandler(out_log_file)
    module_handler.setFormatter(formatter)
    file_logger.addHandler(module_handler)
    file_logger.propagate = False

    # extract all model components
    all_models = state.settings.models

    for model_name in all_models:

        if not model_name in COMPONENTS_TO_SETTINGS:
            msg = f"Cannot pre-check settings for model component {model_name}: " \
                  "mapping to a Pydantic data model is undefined in the checker."
            logger.info(msg)
            file_logger.info(msg)
            continue

        model_settings_class = COMPONENTS_TO_SETTINGS[model_name]["settings_cls"]
        model_settings_file = COMPONENTS_TO_SETTINGS[model_name]["settings_file"]

        # first, attempt to load settings
        # continue if any errorr
        model_settings, model_settings_error = try_load_model_settings(
            model_name=model_name,
            model_settings_class=model_settings_class,
            model_settings_file=model_settings_file,
            state=state,
        )

        if model_settings_error is not None:
            all_errors.append(model_settings_error)
            continue
        
        # then attempt to load and resolve spec/coef files
        errors = try_load_and_check_spec_coefs(
            model_name=model_name,
            model_settings=model_settings,
            state=state,
        )
        all_errors.extend(errors)

        # finally, if model has nested SPEC_SEGMENTS, check each of these.
        # TODO: There are two main methods of using SPEC_SEGMENTS:
        #  - Individual SPEC files keyed to the SPEC field of the segment
        #  - Via PTYPE columns in the top level SPEC, which can be 
        #    evaluated against individual coefficient files
        # Only the the first is fully supported - 
        seg_num = 1
        if model_settings.model_dump().get("SPEC_SEGMENTS"):
            spec_segments = model_settings.SPEC_SEGMENTS
            # SPEC_SEGMENTS can be of type list or dict. If dict, unpack into list
            if isinstance(spec_segments, dict):
                spec_segments = [v for _, v in spec_segments.items()]
            
            # Skip empty lists in case SPEC_SEGMENT field is optional
            if len(spec_segments) > 0:
                for segment_settings in spec_segments:
                    errors = try_load_and_check_spec_coefs(
                        model_name=model_name,
                        model_settings=segment_settings,
                        state=state
                    )
                all_errors.extend(errors)

    if len(all_errors) > 0:
        msg = "Settings Checker Failed with the following errors:"
        logger.error(msg)
        file_logger.error(msg)
        for e in all_errors:
            logger.error(f"\t{e}")
            file_logger.error(f"\t{e}")
        raise RuntimeError("Encountered error in settings checker. See settings_checker.log for details.")
    msg = "Setting Checker Complete! No Errors Found"
    logger.info(msg)
    file_logger.info(msg)
