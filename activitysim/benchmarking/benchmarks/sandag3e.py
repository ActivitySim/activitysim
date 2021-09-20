from .componentwise_template import apply_template
from .sandag_e import common_benchmark_settings

apply_template(
    globals(),
    EXAMPLE_NAME="example_sandag_3_zone",
    CONFIGS_DIRS=("configs_3_zone", "example_mtc/configs"),
    DATA_DIR="data_3",
    OUTPUT_DIR="output_3",
    **common_benchmark_settings,
)