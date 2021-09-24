from activitysim.benchmarking.componentwise import template_setup_cache, template_component_timings
from .componentwise_template import apply_template, f_setup_cache, generate_component_timings, generate_complete
from .sandag_example import *

EXAMPLE_NAME = "example_sandag_1_zone"
CONFIGS_DIRS = ("configs_1_zone", "example_mtc/configs")
DATA_DIR = "data_1"
OUTPUT_DIR = "output_1"
MP_SAMPLE_SIZE = 0
MAX_PROCESSES = 10


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
    )


template_component_timings(
    globals(),
    COMPONENT_NAMES,
    EXAMPLE_NAME,
    CONFIGS_DIRS,
    DATA_DIR,
    OUTPUT_DIR,
    PRELOAD_INJECTABLES,
    REPEAT,
    NUMBER,
    TIMEOUT,
)

# BENCHMARK_SETTINGS_COMPLETE = BENCHMARK_SETTINGS.copy()
# BENCHMARK_SETTINGS_COMPLETE['households_sample_size'] = MP_SAMPLE_SIZE
#
# mp_complete = generate_complete(
#     EXAMPLE_NAME,
#     CONFIGS_DIRS,
#     DATA_DIR,
#     OUTPUT_DIR,
#     TIMEOUT,
#     COMPONENT_NAMES,
#     BENCHMARK_SETTINGS_COMPLETE,
#     SKIM_CACHE=SKIM_CACHE,
#     MAX_PROCESSES=MAX_PROCESSES,
# )
