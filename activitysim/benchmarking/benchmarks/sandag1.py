
from .componentwise_template import f_setup_cache, generate_component_timings

# name of example to load from activitysim_resources
EXAMPLE_NAME = "example_sandag_1_zone_full"

# any settings to override in the example's usual settings file
BENCHMARK_SETTINGS = {
    'households_sample_size': 100_000,
}

# the component names to be benchmarked
COMPONENT_NAMES = [
    # "compute_accessibility",
    "school_location",
    # "workplace_location",
    # "auto_ownership_simulate",
    # "free_parking",
    # "cdap_simulate",
    # "mandatory_tour_frequency",
    # "mandatory_tour_scheduling",
    # "joint_tour_frequency",
    # "joint_tour_composition",
    # "joint_tour_participation",
    # "joint_tour_destination",
    # "joint_tour_scheduling",
    # "non_mandatory_tour_frequency",
    # "non_mandatory_tour_destination",
    # "non_mandatory_tour_scheduling",
    # "tour_mode_choice_simulate",
    # "atwork_subtour_frequency",
    # "atwork_subtour_destination",
    # "atwork_subtour_scheduling",
    # "atwork_subtour_mode_choice",
    # "stop_frequency",
    # "trip_purpose",
    # "trip_destination",
    # "trip_purpose_and_destination",
    # "trip_scheduling",
    # "trip_mode_choice",
]

# benchmarking configuration
TIMEOUT = 36000.0 # ten hours
REPEAT = (
    2,    # min_repeat
    10,   # max_repeat
    20.0, # max_time in seconds
)
NUMBER = 1

# any injectables to preload in setup (so loading isn't counted in time)
PRELOAD_INJECTABLES = (
    'skim_dict',
)


# benchmarking implementation

def setup_cache():
    f_setup_cache(EXAMPLE_NAME, COMPONENT_NAMES, BENCHMARK_SETTINGS, CONFIGS_DIRS=("configs_1_zone", "example_mtc/configs"))


for cname in COMPONENT_NAMES:
    globals()[f"time_{cname}"] = generate_component_timings(
        cname,
        EXAMPLE_NAME,
        PRELOAD_INJECTABLES,
        REPEAT,
        NUMBER,
        TIMEOUT,
    )
