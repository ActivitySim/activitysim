from .componentwise_template import apply_template
from .sandag_example import common_benchmark_settings

apply_template(
    globals(),
    EXAMPLE_NAME="example_sandag_2_zone",
    CONFIGS_DIRS=("configs_2_zone", "example_psrc/configs"),
    DATA_DIR="data_2",
    OUTPUT_DIR="output_2",
    **common_benchmark_settings,
)