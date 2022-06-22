from activitysim.benchmarking.componentwise import (
    template_component_timings,
    template_setup_cache,
)

EXAMPLE_NAME = "prototype_mtc_full"
CONFIGS_DIRS = ("configs",)
DYNAMIC_CONFIG_DIR = "bench_configs"
DATA_DIR = "data"
OUTPUT_DIR = "output"
COMPONENT_NAMES = [
    # "compute_accessibility",
    "school_location",
    "workplace_location",
    "auto_ownership_simulate",
    "free_parking",
    "cdap_simulate",
    "mandatory_tour_frequency",
    "mandatory_tour_scheduling",
    "joint_tour_frequency",
    "joint_tour_composition",
    "joint_tour_participation",
    "joint_tour_destination",
    "joint_tour_scheduling",
    "non_mandatory_tour_frequency",
    "non_mandatory_tour_destination",
    "non_mandatory_tour_scheduling",
    "tour_mode_choice_simulate",
    "atwork_subtour_frequency",
    "atwork_subtour_destination",
    "atwork_subtour_scheduling",
    "atwork_subtour_mode_choice",
    "stop_frequency",
    "trip_purpose",
    "trip_destination",
    "trip_purpose_and_destination",
    "trip_scheduling",
    "trip_mode_choice",
    # "write_data_dictionary",
    # "track_skim_usage",
    "write_trip_matrices",
    # "write_tables",
]
BENCHMARK_SETTINGS = {
    "households_sample_size": 48_769,
}
SKIM_CACHE = False
PRELOAD_INJECTABLES = ("skim_dict",)
REPEAT = 1
NUMBER = 1
TIMEOUT = 36000.0  # ten hours
VERSION = "1"


def setup_cache():
    template_setup_cache(
        EXAMPLE_NAME,
        COMPONENT_NAMES,
        BENCHMARK_SETTINGS,
        dict(
            read_skim_cache=SKIM_CACHE,
            write_skim_cache=SKIM_CACHE,
        ),
        CONFIGS_DIRS,
        DATA_DIR,
        OUTPUT_DIR,
        config_overload_dir=DYNAMIC_CONFIG_DIR,
    )


template_component_timings(
    globals(),
    COMPONENT_NAMES,
    EXAMPLE_NAME,
    (DYNAMIC_CONFIG_DIR, *CONFIGS_DIRS),
    DATA_DIR,
    OUTPUT_DIR,
    PRELOAD_INJECTABLES,
    REPEAT,
    NUMBER,
    TIMEOUT,
    VERSION,
)
