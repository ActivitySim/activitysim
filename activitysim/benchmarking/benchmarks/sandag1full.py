from .componentwise_template import apply_template
from .sandag_full import common_benchmark_settings

apply_template(
    globals(),
    EXAMPLE_NAME="example_sandag_1_zone_full",
    CONFIGS_DIRS=("configs_benchmarking", "configs_1_zone", "example_mtc/configs"),
    DATA_DIR="data_1",
    OUTPUT_DIR="output_1",
    **common_benchmark_settings,
)