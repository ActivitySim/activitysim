import logging
import os
from pandas import DataFrame
from pydantic import BaseModel as PydanticBase
from typing import Type, Optional

from activitysim.core.configuration.base import PydanticReadable

# import core settings
from activitysim.core.configuration.logit import (
    LogitNestSpec,
    TourLocationComponentSettings,
    TourModeComponentSettings,
    TemplatedLogitComponentSettings,
)
from activitysim.core import config
from activitysim.core.configuration.network import NetworkSettings
from activitysim.core.workflow import State
from activitysim.core.simulate import (
    eval_coefficients,
    eval_nest_coefficients,
    read_model_coefficient_template,
)
from activitysim.core.exceptions import ModelConfigurationError

# import model settings
from activitysim.abm.models.accessibility import AccessibilitySettings
from activitysim.abm.models.atwork_subtour_frequency import (
    AtworkSubtourFrequencySettings,
)
from activitysim.abm.models.auto_ownership import AutoOwnershipSettings
from activitysim.abm.models.cdap import CdapSettings
from activitysim.abm.models.disaggregate_accessibility import (
    DisaggregateAccessibilitySettings,
    read_disaggregate_accessibility_yaml,
)
from activitysim.abm.models.free_parking import FreeParkingSettings
from activitysim.abm.models.initialize import InitializeTableSettings
from activitysim.abm.models.joint_tour_composition import JointTourCompositionSettings
from activitysim.abm.models.joint_tour_frequency_composition import (
    JointTourFreqCompSettings,
)
from activitysim.abm.models.joint_tour_frequency import JointTourFrequencySettings
from activitysim.abm.models.joint_tour_participation import (
    JointTourParticipationSettings,
)
from activitysim.abm.models.mandatory_tour_frequency import (
    MandatoryTourFrequencySettings,
)
from activitysim.abm.models.non_mandatory_tour_frequency import (
    NonMandatoryTourFrequencySettings,
)
from activitysim.abm.models.parking_location_choice import ParkingLocationSettings
from activitysim.abm.models.school_escorting import SchoolEscortSettings
from activitysim.abm.models.stop_frequency import StopFrequencySettings
from activitysim.abm.models.summarize import SummarizeSettings
from activitysim.abm.models.telecommute_frequency import TelecommuteFrequencySettings
from activitysim.abm.models.tour_scheduling_probabilistic import (
    TourSchedulingProbabilisticSettings,
)
from activitysim.abm.models.transit_pass_ownership import TransitPassOwnershipSettings
from activitysim.abm.models.transit_pass_subsidy import TransitPassSubsidySettings
from activitysim.abm.models.trip_departure_choice import TripDepartureChoiceSettings
from activitysim.abm.models.trip_destination import TripDestinationSettings
from activitysim.abm.models.trip_matrices import WriteTripMatricesSettings
from activitysim.abm.models.trip_mode_choice import TripModeChoiceSettings
from activitysim.abm.models.trip_purpose_and_destination import (
    TripPurposeAndDestinationSettings,
)
from activitysim.abm.models.trip_purpose import TripPurposeSettings
from activitysim.abm.models.vehicle_allocation import VehicleAllocationSettings
from activitysim.abm.models.vehicle_type_choice import VehicleTypeChoiceSettings
from activitysim.abm.models.work_from_home import WorkFromHomeSettings

# import util settings
from activitysim.abm.models.util.vectorize_tour_scheduling import (
    TourSchedulingSettings,
)
from activitysim.abm.models.util.tour_od import TourODSettings

# import table settings
from activitysim.abm.tables.shadow_pricing import ShadowPriceSettings


class SettingsCheckerError(Exception):
    """Custom exception for settings checker errors."""

    def __init__(
        self,
        model_name: str,
        exception: Exception,
        error_files: None = None,
        additional_info: str = None,
    ):
        self.model_name = model_name
        self.exception = exception
        self.error_files = error_files
        self.additional_info = additional_info
        self.message = self._construct_message()

        super().__init__(self.message)

    def _construct_message(self) -> str:
        message = f"Error checking settings for {self.model_name}"
        if self.error_files is not None:
            # cast all files from path to strings if required and wrap into list
            if not isinstance(self.error_files, list):
                message_files = [self.error_files]
            else:
                message_files = self.error_files
            message += f" using files {', '.join([str(f) for f in message_files])}"
        message += f": {str(self.exception)}"
        if self.additional_info is not None:
            message += f". {self.additional_info}"
        return message


# setup logging
logger = logging.getLogger(__name__)
file_logger = logger.getChild("logfile")

CHECKER_SETTINGS = {
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
        "settings_file": "tour_scheduling_atwork.yaml",
    },
    "auto_ownership_simulate": {
        "settings_cls": AutoOwnershipSettings,
        "settings_file": "auto_ownership.yaml",
    },
    "cdap_simulate": {"settings_cls": CdapSettings, "settings_file": "cdap.yaml"},
    "compute_disaggregate_accessibility": {
        "settings_cls": DisaggregateAccessibilitySettings,
        "settings_file": "disaggregate_accessibility.yaml",
    },
    "free_parking": {
        "settings_cls": FreeParkingSettings,
        "settings_file": "free_parking.yaml",
    },
    "initialize_households": {
        "settings_cls": InitializeTableSettings,
        "settings_file": "initialize_households.yaml",
    },
    "initialize_landuse": {
        "settings_cls": InitializeTableSettings,
        "settings_file": "initialize_landuse.yaml",
    },
    "initialize_los": {
        "settings_cls": NetworkSettings,
        "settings_file": "network_los.yaml",
    },
    "input_checker": {
        "settings_cls": PydanticReadable,  # input checker uses state.filesystem.read_model_settings directly
        "settings_file": "input_checker.yaml",
    },
    "joint_tour_composition": {
        "settings_cls": JointTourCompositionSettings,
        "settings_file": "joint_tour_composition.yaml",
    },
    "joint_tour_destination": {
        "settings_cls": TourLocationComponentSettings,
        "settings_file": "joint_tour_destination.yaml",
    },
    "joint_tour_frequency_composition": {
        "settings_cls": JointTourFreqCompSettings,
        "settings_file": "joint_tour_frequency_composition.yaml",
    },
    "joint_tour_frequency": {
        "settings_cls": JointTourFrequencySettings,
        "settings_file": "joint_tour_frequency.yaml",
    },
    "joint_tour_participation": {
        "settings_cls": JointTourParticipationSettings,
        "settings_file": "joint_tour_participation.yaml",
    },
    "joint_tour_scheduling": {
        "settings_cls": TourSchedulingSettings,
        "settings_file": "joint_tour_scheduling.yaml",
    },
    "mandatory_tour_frequency": {
        "settings_cls": MandatoryTourFrequencySettings,
        "settings_file": "mandatory_tour_frequency.yaml",
    },
    "mandatory_tour_scheduling": {
        "settings_cls": TourSchedulingSettings,
        "settings_file": "mandatory_tour_scheduling.yaml",
    },
    "non_mandatory_tour_destination": {
        "settings_cls": TourLocationComponentSettings,
        "settings_file": "non_mandatory_tour_destination.yaml",
    },
    "non_mandatory_tour_frequency": {
        "settings_cls": NonMandatoryTourFrequencySettings,
        "settings_file": "non_mandatory_tour_frequency.yaml",
    },
    "non_mandatory_tour_scheduling": {
        "settings_cls": TourSchedulingSettings,
        "settings_file": "non_mandatory_tour_scheduling.yaml",
    },
    "parking_location": {
        "settings_cls": ParkingLocationSettings,
        "settings_file": "parking_location_choice.yaml",
    },
    "school_escorting": {
        "settings_cls": SchoolEscortSettings,
        "settings_file": "school_escorting.yaml",
        "spec_coefficient_keys": [
            {"spec": "OUTBOUND_SPEC", "coefs": "OUTBOUND_COEFFICIENTS"},
            {"spec": "INBOUND_SPEC", "coefs": "INBOUND_COEFFICIENTS"},
            {"spec": "OUTBOUND_COND_SPEC", "coefs": "OUTBOUND_COND_COEFFICIENTS"},
        ],
    },
    "school_location": {
        "settings_cls": TourLocationComponentSettings,
        "settings_file": "school_location.yaml",
    },
    "shadow_pricing": {
        "settings_cls": ShadowPriceSettings,
        "settings_file": "shadow_pricing.yaml",
    },
    "stop_frequency": {
        "settings_cls": StopFrequencySettings,
        "settings_file": "stop_frequency.yaml",
    },
    "summarize": {"settings_cls": SummarizeSettings, "settings_file": "summarize.yaml"},
    "telecommute_frequency": {
        "settings_cls": TelecommuteFrequencySettings,
        "settings_file": "telecommute_frequency.yaml",
    },
    "tour_mode_choice_simulate": {
        "settings_cls": TourModeComponentSettings,
        "settings_file": "tour_mode_choice.yaml",
    },
    "tour_od_choice": {
        "settings_cls": TourODSettings,
        "settings_file": "tour_od_choice.yaml",
    },
    "tour_scheduling_probabilistic": {
        "settings_cls": TourSchedulingProbabilisticSettings,
        "settings_file": "tour_scheduling_probabilistic.yaml",
    },
    "transit_pass_ownership": {
        "settings_cls": TransitPassOwnershipSettings,
        "settings_file": "transit_pass_ownership.yaml",
    },
    "transit_pass_subsidy": {
        "settings_cls": TransitPassSubsidySettings,
        "settings_file": "transit_pass_subsidy.yaml",
    },
    "trip_departure_choice": {
        "settings_cls": TripDepartureChoiceSettings,
        "settings_file": "trip_departure_choice.yaml",
    },
    "trip_destination": {
        "settings_cls": TripDestinationSettings,
        "settings_file": "trip_destination.yaml",
    },
    "trip_mode_choice": {
        "settings_cls": TripModeChoiceSettings,
        "settings_file": "trip_mode_choice.yaml",
    },
    "trip_purpose": {
        "settings_cls": TripPurposeSettings,
        "settings_file": "trip_purpose.yaml",
    },
    "trip_purpose_and_destination": {
        "settings_cls": TripPurposeAndDestinationSettings,
        "settings_file": "trip_purpose_and_destination.yaml",
    },
    "vehicle_allocation": {
        "settings_cls": VehicleAllocationSettings,
        "settings_file": "vehicle_allocation.yaml",
    },
    "vehicle_type_choice": {
        "settings_cls": VehicleAllocationSettings,
        "settings_file": "vehicle_allocation.yaml",
    },
    "vehicle_type_choice": {
        "settings_cls": VehicleTypeChoiceSettings,
        "settings_file": "vehicle_type_choice.yaml",
    },
    "work_from_home": {
        "settings_cls": WorkFromHomeSettings,
        "settings_file": "work_from_home.yaml",
    },
    "workplace_location": {
        "settings_cls": TourLocationComponentSettings,
        "settings_file": "workplace_location.yaml",
    },
    "write_data_dictionary": {
        "settings_cls": PydanticReadable,  # write data dictionary uses state.filesystem.read_model_settings directly
        "settings_file": "write_data_dictionary.yaml",
        "warn_only": True,
    },
    "write_trip_matrices": {
        "settings_cls": WriteTripMatricesSettings,
        "settings_file": "write_trip_matrices.yaml",
    },
}


def try_load_model_settings(
    model_name: str,
    model_settings_class: Type[PydanticBase],
    model_settings_file: str,
    state: State,
) -> tuple[PydanticBase | None, Exception | None]:

    msg = f"Attempting to load model settings for {model_name} via {model_settings_class.__name__} and {model_settings_file}"
    logger.debug(msg)
    file_logger.info(msg)

    try:
        if isinstance(model_settings_class, DisaggregateAccessibilitySettings):
            model_settings = read_disaggregate_accessibility_yaml(
                state, model_settings_file
            )
        elif model_name == "input_checker":
            # HACK: input checker does not define a pydantic data model, but reads directly to dictionary. Wrapping in BaseModel
            # provides the required model_dump interface downstream without adding additional branching logic.
            class InputCheckerSettings(PydanticBase):
                input_check_settings: dict

            input_check_settings = state.filesystem.read_model_settings(
                model_settings_file, mandatory=True
            )
            model_settings = InputCheckerSettings(
                input_check_settings=input_check_settings
            )
        else:
            model_settings = model_settings_class.read_settings_file(
                state.filesystem, model_settings_file
            )
        result = model_settings, None
        msg = f"Successfully loaded model settings from {model_settings_file}"
        logger.debug(msg)
    except Exception as e:
        result = None, e
    return result


def try_load_spec(
    model_name: str, model_settings: PydanticBase, spec_file: str, state: State
) -> tuple[DataFrame | None, Exception | None]:
    msg = f"Attempting to load SPEC for {model_name} via {model_settings.__class__.__name__}"
    logger.debug(msg)
    file_logger.info(msg)
    try:
        result = state.filesystem.read_model_spec(spec_file), None
        msg = f"Successfully loaded model SPEC from {spec_file}"
        logger.debug(msg)
        file_logger.info(msg)
    except Exception as e:
        # always return a dataframe
        result = None, e
    return result


def try_load_coefs(
    model_name: str, model_settings: PydanticBase, coefs_file: str, state: State
) -> tuple[DataFrame, Optional[Exception]]:
    msg = f"Attempting to load COEFFICIENTS for {model_name} via {model_settings.__class__.__name__}"
    logger.debug(msg)
    file_logger.info(msg)

    try:
        result = state.filesystem.read_model_coefficients(file_name=coefs_file), None
        msg = f"Successfully loaded model Coefficients from {coefs_file}"
        logger.debug(msg)
        file_logger.info(msg)
    except Exception as e:
        result = None, e
    return result


def try_eval_spec_coefs(
    model_name: str,
    model_settings: PydanticBase,
    spec: DataFrame | None,
    coefs: DataFrame | None,
    state: State,
) -> tuple[DataFrame | None, Exception | None]:

    if spec is None or coefs is None:
        msg_prefix = (
            f"Skipping Evaluation Check for {model_settings.__class__.__name__}"
        )
        spec_msg = "No SPEC available" if spec is None else ""
        coefs_msg = "No COEFFICENTS available" if coefs is None else ""
        msg = ". ".join([msg_prefix, spec_msg, coefs_msg])
        logger.debug(msg)
        file_logger.debug(msg)
        return None, None

    try:
        # check whether coefficients should be evaluated as NESTS or not
        if model_settings.model_dump().get("NESTS"):
            if isinstance(model_settings.NESTS, LogitNestSpec):
                nests = model_settings.NESTS
        else:
            nests = None
        if nests is not None:
            # Proper Trace label is probably unneeded here
            result = (
                eval_nest_coefficients(model_settings.NESTS, coefs, trace_label=None),
                None,
            )
        else:
            result = eval_coefficients(state, spec, coefs, estimator=None), None
        msg = f"Successfully evaluated coefficients for {model_name}"
        logger.debug(msg)
        file_logger.info(msg)
    except Exception as e:
        result = None, e
    return result


def try_check_spec_coefs_templated(
    model_name: str, model_settings: TemplatedLogitComponentSettings, state: State
) -> list[Exception]:
    """Alternative function for checking mode choice settings using a templated coefficients files"""

    errors = []
    inner_errors = []

    try:
        coefs_template = read_model_coefficient_template(
            state.filesystem, model_settings
        )
        coefs_segments = list(coefs_template.columns)

        for segment_name in coefs_segments:
            try:
                nest_spec = config.get_logit_model_settings(model_settings)
                coefs = state.filesystem.get_segment_coefficients(
                    model_settings, segment_name
                )
                # Proper trace label probably unneeded here
                nest_spec = eval_nest_coefficients(nest_spec, coefs, trace_label=None)
            except Exception as e:
                additional_info = f"Could not evaluate templated coefficients for segment {segment_name}. Check that SPEC, Coefficients, and Template files exist and have compatible labels."
                inner_errors.append(
                    SettingsCheckerError(
                        model_name,
                        e,
                        [
                            model_settings.SPEC,
                            model_settings.COEFFICIENTS,
                            model_settings.COEFFICIENT_TEMPLATE,
                        ],
                        additional_info,
                    )
                )
                continue
    except Exception as e:
        msg = f"{model_name}: Could not evaluate templated coefficients. Check that SPEC, Coefficients, and Template files exist and have compatible labels."
        logger.warning(msg)
        file_logger.warning(msg)

        additional_info = "Could not evaluated templated coefficients. Check that SPEC, Coefficients, and Template files exist and have compatible labels."

        errors.append(
            SettingsCheckerError(
                model_name,
                e,
                [
                    model_settings.SPEC,
                    model_settings.COEFFICIENTS,
                    model_settings.COEFFICIENT_TEMPLATE,
                ],
                additional_info,
            )
        )

    errors.extend(inner_errors)

    return errors


def try_check_spec_coefs_ptype_spec_segments(
    model_name: str, model_settings: PydanticBase, state: State
) -> list[Exception]:
    """Alternative function for checking settings that are segmented by PTYPE within the main model spec"""
    errors = []

    try:
        spec_segments = model_settings.SPEC_SEGMENTS
        model_spec = state.filesystem.read_model_spec(file_name=model_settings.SPEC)

        # pick the spec column for the segment
        for segment_settings in spec_segments:
            segment_name = segment_settings.NAME
            segment_spec = model_spec[[segment_name]]

            coefficients_df = state.filesystem.read_model_coefficients(segment_settings)
            segment_spec = eval_coefficients(
                state, segment_spec, coefficients_df, estimator=None
            )
    except Exception as e:
        errors.append(
            SettingsCheckerError(
                model_name, e, [model_settings.SPEC, model_settings.COEFFICIENTS]
            )
        )
    return errors


def try_load_and_check_spec_coefs(
    model_name: str,
    model_settings: Type[PydanticBase],
    state: State,
    spec_coefficient_keys: list[dict] = None,
) -> list[Exception]:
    """Attempt to load and evaluate SPEC and COEFFICIENTS.
    By default, will look for SPEC and COEFFICIENTS at the top level of the settings.
    This can be overriden by providing an alternative set of spec/coefs keys
    in the settings checker register.
    """
    # collect all errors
    errors = []

    if spec_coefficient_keys is None:
        spec_coefficient_keys = [{"spec": "SPEC", "coefs": "COEFFICIENTS"}]

    for key_pair in spec_coefficient_keys:

        # attempt to read SPEC file
        if hasattr(model_settings, key_pair["spec"]):
            spec_file = model_settings.model_dump().get(key_pair["spec"])

            # HACK: some models may use older "SPECIFICATION" field name instead of "SPEC"
            if spec_file is None and hasattr(model_settings, "SPECIFICATION"):
                spec_file = model_settings.SPECIFICATION

            if spec_file is not None:
                spec, spec_error = try_load_spec(
                    model_name=model_name,
                    model_settings=model_settings,
                    spec_file=spec_file,
                    state=state,
                )
            else:
                spec, spec_error = None, None
                msg = f"{model_name}: Field {key_pair['spec']} is None in {model_settings.__class__.__name__}. Ensure that a filepath is defined YAML settings if required"
                logger.warning(msg)
                file_logger.warning(msg)
        else:
            spec, spec_error = None, None
            # msg = f"No SPEC file is associated with {model_settings.__class__.__name__}"
            # logger.info(msg)
            # file_logger.info(msg)

        if spec_error is not None:
            errors.append(spec_error)

        # then attempt to read coefficients
        if hasattr(model_settings, key_pair["coefs"]):
            coefs_file = model_settings.model_dump().get(key_pair["coefs"])
            if coefs_file is not None:
                coefs, coefs_error = try_load_coefs(
                    model_name=model_name,
                    model_settings=model_settings,
                    coefs_file=coefs_file,
                    state=state,
                )
            else:
                coefs, coefs_error = None, None
                msg = f"{model_name}: Field {key_pair['coefs']} is None in {model_settings.__class__.__name__}. Ensure that a filepath is defined YAML settings if required"
                logger.warning(msg)
                file_logger.warning(msg)
        else:
            coefs, coefs_error = None, None
            # msg = f"No coefficients file is associated with {model_settings.__class__.__name__}"
            # logger.info(msg)
            # file_logger.info(msg)

        if coefs_error is not None:
            errors.append(SettingsCheckerError(model_name, coefs_error, coefs_file))

        # then attempt to evaluate coefficients against spec
        eval_coefs, eval_coefs_error = try_eval_spec_coefs(
            model_name=model_name,
            model_settings=model_settings,
            spec=spec,
            coefs=coefs,
            state=state,
        )

        if eval_coefs_error is not None:
            errors.append(
                SettingsCheckerError(
                    model_name, eval_coefs_error, [spec_file, coefs_file]
                )
            )

    # then, check any other subsettings that may have a SPEC
    # this includes preprocessors and annotators, etc.
    # for now, check is limited to check that the SPEC file is loadable
    for _, setting in model_settings:
        if (
            isinstance(setting, PydanticBase)
            and setting.model_dump().get("SPEC") is not None
        ):
            addl_spec_file = setting.SPEC
            addl_spec, addl_spec_error = try_load_spec(
                model_name=model_name + f": {setting.__class__.__name__}",
                model_settings=setting,
                spec_file=addl_spec_file,
                state=state,
            )
            if addl_spec_error:
                errors.append(
                    SettingsCheckerError(model_name, addl_spec_error, addl_spec_file)
                )
    return errors


def check_model_settings(
    state: State,
    checker_settings: dict = CHECKER_SETTINGS,
    extension_settings: dict = {},
    log_file: str = "settings_checker.log",
) -> None:

    # Collect all errors
    all_errors = []

    # additional logging set up
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    out_log_file = state.get_log_file_path(log_file)
    if os.path.exists(out_log_file):
        os.remove(out_log_file)
    module_handler = logging.FileHandler(out_log_file)
    module_handler.setFormatter(formatter)
    file_logger.addHandler(module_handler)
    file_logger.propagate = False

    # add extension settings to checker settings
    if extension_settings is not None:
        checker_settings.update(extension_settings)

    # extract all model components
    all_models = state.settings.models.copy()

    # add shadow pricing and initalize los (not in state.settings.models)
    if state.settings.use_shadow_pricing == True:
        all_models.append("shadow_pricing")
    if "initialize_los" in state._RUNNABLE_STEPS:
        all_models.append("initialize_los")

    for model_name in all_models:

        if not model_name in checker_settings:
            msg = (
                f"Cannot pre-check settings for model component {model_name}: "
                "mapping to a Pydantic data model is undefined in the checker."
            )
            logger.info(msg)
            file_logger.info(msg)
            continue

        model_settings_class = checker_settings[model_name]["settings_cls"]
        model_settings_file = checker_settings[model_name]["settings_file"]
        spec_coefficient_keys = checker_settings[model_name].get(
            "spec_coefficient_keys"
        )
        # do not raise errors if YAML file cannot be loaded
        # this is used for write_data_dictionary
        warn_only = checker_settings[model_name].get("warn_only", False)

        # first, attempt to load settings
        # continue if any error
        model_settings, model_settings_error = try_load_model_settings(
            model_name=model_name,
            model_settings_class=model_settings_class,
            model_settings_file=model_settings_file,
            state=state,
        )

        if model_settings_error is not None:
            if warn_only:
                msg = f"{model_name} settings file {model_settings_file} could not be loaded. Ensure inclusion of this configuration file is optional."
                logger.warning(msg)
                file_logger.warning(msg)
                continue
            else:
                all_errors.append(
                    SettingsCheckerError(
                        model_name, model_settings_error, model_settings_file
                    )
                )
                continue

        # then attempt to load and resolve spec/coef files
        if isinstance(model_settings, TemplatedLogitComponentSettings):
            errors = try_check_spec_coefs_templated(
                model_name=model_name, model_settings=model_settings, state=state
            )
        else:
            errors = try_load_and_check_spec_coefs(
                model_name=model_name,
                model_settings=model_settings,
                state=state,
                spec_coefficient_keys=spec_coefficient_keys,
            )
        all_errors.extend(errors)

        # if model has nested SPEC_SEGMENTS, check each of these.
        # there are two ways of segmenting specs, which are handled differently:
        #   1) Settings using define separate pairs of spec/coefficient files.
        #   2) Others define segments within the main model spec file, keyed by PTYPE.
        if model_settings.model_dump().get("SPEC_SEGMENTS"):

            spec_segments = model_settings.SPEC_SEGMENTS

            if isinstance(spec_segments, dict):
                spec_segments = [
                    segment for segment_name, segment in spec_segments.items()
                ]

            # check the first segment to see if PTYPE should be defined
            # this avoids needing to hardcode branching logic to determine evaluation method
            if "PTYPE" in spec_segments[0].model_fields:
                errors = try_check_spec_coefs_ptype_spec_segments(
                    model_name=model_name,
                    model_settings=model_settings,
                    state=state,
                )
                all_errors.extend(errors)
            else:
                for segment_settings in spec_segments:
                    errors = try_load_and_check_spec_coefs(
                        model_name=model_name,
                        model_settings=segment_settings,
                        state=state,
                    )
                all_errors.extend(errors)

    if len(all_errors) > 0:
        msg = "Settings Checker Failed with the following errors:"
        logger.error(msg)
        file_logger.error(msg)
        for e in all_errors:
            logger.error(f"\t{str(e)}")
            file_logger.error(f"\t{str(e)}")
        raise ModelConfigurationError(
            f"Encountered one or more errors in settings checker. See f{log_file} for details."
        )
    msg = f"Setting Checker Complete. No runtime errors were raised. Check f{log_file} for warnings. These *may* prevent model from successfully running."
    logger.info(msg)
    file_logger.info(msg)
