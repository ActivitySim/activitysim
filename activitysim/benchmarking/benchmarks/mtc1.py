import argparse
import sys
import os
import logging
import time
import yaml
from datetime import timedelta
from functools import partial
from activitysim.benchmarking import componentwise, modify_yaml, workspace
from activitysim.cli.create import get_example

logger = logging.getLogger("activitysim.benchmarking")
benchmarking_directory = workspace.get_dir()

# name of example to load from activitysim_resources
EXAMPLE_NAME = "example_mtc_full"

# any settings to override in the example's usual settings file
BENCHMARK_SETTINGS = {
    'households_sample_size': 1_000,
}

# the component names to be benchmarked
COMPONENT_NAMES = [
    # "compute_accessibility",
    "school_location",
    "workplace_location",
    "auto_ownership_simulate",
    "free_parking",
    "cdap_simulate",
    "mandatory_tour_frequency",
    "mandatory_tour_scheduling",
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


def setup_cache():

    if workspace.get_dir() is None:
        from asv.console import log
        for k,v in os.environ.items():
            log.error(f" env {k}: {v}")
        raise RuntimeError("workspace unavailable")
    os.makedirs(os.path.join(local_dir(), "models"), exist_ok=True)
    get_example(
        example_name=EXAMPLE_NAME,
        destination=os.path.join(local_dir(), "models"),
    )
    settings_filename = os.path.join(model_dir(), "configs", "settings.yaml")
    with open(settings_filename, 'rt') as f:
        models = yaml.load(f, Loader=yaml.loader.SafeLoader).get('models')

    last_component_to_benchmark = 0
    for component_name in COMPONENT_NAMES:
        last_component_to_benchmark = max(
            models.index(component_name),
            last_component_to_benchmark
        )
    pre_run_model_list = models[:last_component_to_benchmark]
    modify_yaml(
        os.path.join(model_dir(), "configs", "settings.yaml"),
        **BENCHMARK_SETTINGS,
        models=pre_run_model_list,
        checkpoints=True,
        trace_hh_id=None,
        chunk_training_mode='off',
    )
    modify_yaml(
        os.path.join(model_dir(), "configs", "network_los.yaml"),
        read_skim_cache=True,
    )
    componentwise.pre_run(model_dir())


def local_dir():
    if benchmarking_directory is not None:
        return benchmarking_directory
    return os.getcwd()


def model_dir():
    return os.path.join(local_dir(), "models", EXAMPLE_NAME)


def generate_component_timings(component_name):

    class ComponentTiming:
        component_name = component_name
        repeat = REPEAT
        number = NUMBER
        timeout = TIMEOUT
        def setup(self):
            componentwise.setup_component(self.component_name, model_dir(), PRELOAD_INJECTABLES)
        def teardown(self):
            componentwise.teardown_component(self.component_name)
        def time_component(self):
            componentwise.run_component(self.component_name)

    ComponentTiming.__name__ = f"time_{component_name}"

    return ComponentTiming


for component_name in COMPONENT_NAMES:
    globals()[f"time_{component_name}"] = generate_component_timings(component_name)


if __name__ == '__main__':

    benchmarking_data_directory = workspace.get_dir()
    os.chdir(benchmarking_data_directory)

    t0a = time.time()
    setup_cache()
    t0b = time.time()

    timings = {}
    for component_name in COMPONENT_NAMES:

        logger.warning(f"$$$$$$$$ {component_name} #1 $$$$$$$$")
        f = globals()[f"time_{component_name}"]
        f.setup(component_name)
        t1a = time.time()
        f(component_name)
        t1b = time.time()
        f.teardown(component_name)

        logger.warning(f"$$$$$$$$ {component_name} #2 $$$$$$$$")
        f.setup(component_name)
        t2a = time.time()
        f(component_name)
        t2b = time.time()
        f.teardown(component_name)

        timings[component_name] = (
            str(timedelta(seconds=t1b-t1a)),
            str(timedelta(seconds=t2b-t2a)),
        )

    logger.warning(f"Time Base Setup: {timedelta(seconds=t0b-t0a)}")
    for component_name in COMPONENT_NAMES:
        logger.warning(f"Time {component_name}: {timings[component_name]}")
