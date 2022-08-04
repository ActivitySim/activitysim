import multiprocessing

import numpy as np

from activitysim.benchmarking.componentwise import (
    template_component_timings_mp,
    template_setup_cache,
)

PRETTY_NAME = "MTC1_MP4"
EXAMPLE_NAME = "prototype_mtc_full"
NUM_PROCESSORS = int(np.clip(multiprocessing.cpu_count() - 2, 2, 4))
CONFIGS_DIRS = ("configs_mp", "configs")
DYNAMIC_CONFIG_DIR = "bench_configs_mp"
DATA_DIR = "data"
OUTPUT_DIR = "output_mp"
COMPONENT_NAMES = [
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
]
BENCHMARK_SETTINGS = {
    # TODO: This multiprocess benchmarking is minimally functional,
    # but has a bad habit of crashing due to memory allocation errors on
    # all but the tiniest of examples. It would be great to fix the MP
    # benchmarks so they use chunking, automatically configure for available
    # RAM, and run a training-production cycle to get useful timing results.
    "households_sample_size": 400,
    "num_processes": NUM_PROCESSORS,
}
SKIM_CACHE = False
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


template_component_timings_mp(
    globals(),
    COMPONENT_NAMES,
    EXAMPLE_NAME,
    OUTPUT_DIR,
    PRETTY_NAME,
    VERSION,
)
