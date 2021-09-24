from .componentwise_template import apply_template, f_setup_cache
from .sandag_example import *

EXAMPLE_NAME = "example_sandag_3_zone"
CONFIGS_DIRS = ("configs_3_zone", "example_mtc/configs")
DATA_DIR = "data_3"
OUTPUT_DIR = "output_3"


def setup_cache():
    f_setup_cache(
        EXAMPLE_NAME, COMPONENT_NAMES, BENCHMARK_SETTINGS,
        CONFIGS_DIRS, DATA_DIR, OUTPUT_DIR,
        SKIM_CACHE=SKIM_CACHE,
    )


apply_template(
    globals(),
    EXAMPLE_NAME=EXAMPLE_NAME,
    CONFIGS_DIRS=CONFIGS_DIRS,
    DATA_DIR=DATA_DIR,
    OUTPUT_DIR=OUTPUT_DIR,
    **common_benchmark_settings,
)