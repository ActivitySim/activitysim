from .componentwise_template import apply_template
from .sandag_f import common_benchmark_settings

apply_template(
    globals(),
    EXAMPLE_NAME="example_sandag_2_zone_full",
    CONFIGS_DIRS=("configs_benchmarking", "configs_3_zone", "example_psrc/configs"),
    DATA_DIR="data_2",
    OUTPUT_DIR="output_2",
    **common_benchmark_settings,
)