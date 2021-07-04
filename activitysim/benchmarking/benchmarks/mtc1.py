import argparse
import os
import logging
import time
import yaml
from datetime import timedelta
from activitysim.benchmarking import componentwise, modify_yaml, workspace
from activitysim.cli.create import get_example

logger = logging.getLogger("activitysim.benchmarking")


class BenchSuite_MTC:

    benchmarking_directory = workspace.get_dir()

    # name of example to load from activitysim_resources
    example_name = "example_mtc_full"

    # any settings to override in the example's usual settings file
    benchmark_settings = {
        'households_sample_size': 100_000,
    }

    # the component names to be benchmarked
    params = [
        "compute_accessibility",
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

    param_names = ['component_name']

    timeout = 36000.0 # ten hours
    repeat = (
        2,    # min_repeat
        10,   # max_repeat
        20.0, # max_time in seconds
    )
    number = 1

    preload_injectables = (
        'skim_dict',
    )

    def setup_cache(self):
        if workspace.get_dir() is None:
            from asv.console import log
            for k,v in os.environ.items():
                log.error(f" env {k}: {v}")
            raise RuntimeError("workspace unavailable")
        get_example(
            example_name=self.example_name,
            destination=os.path.join(self.local_dir, "models"),
        )
        settings_filename = os.path.join(self.model_dir, "configs", "settings.yaml")
        with open(settings_filename, 'rt') as f:
            self.models = yaml.load(f, Loader=yaml.loader.SafeLoader).get('models')

        last_component_to_benchmark = 0
        for component_name in self.params:
            last_component_to_benchmark = max(
                self.models.index(component_name),
                last_component_to_benchmark
            )
        pre_run_model_list = self.models[:last_component_to_benchmark]
        modify_yaml(
            os.path.join(self.model_dir, "configs", "settings.yaml"),
            **self.benchmark_settings,
            models=pre_run_model_list,
            checkpoints=True,
            trace_hh_id=None,
            chunk_training_mode='off',
        )
        modify_yaml(
            os.path.join(self.model_dir, "configs", "network_los.yaml"),
            read_skim_cache=True,
        )
        componentwise.pre_run(self.model_dir)

    def setup(self, component_name):
        componentwise.setup_component(
            component_name,
            self.model_dir,
            self.preload_injectables,
        )

    def time_component(self, component_name):
        return componentwise.run_component(component_name)

    def teardown(self, component_name):
        componentwise.teardown_component(component_name)

    @property
    def local_dir(self):
        if self.benchmarking_directory is not None:
            return self.benchmarking_directory
        return os.getcwd()

    @property
    def model_dir(self):
        return os.path.join(self.local_dir, "models", self.example_name)

if __name__ == '__main__':

    benchmarking_data_directory = workspace.get_dir()
    os.chdir(benchmarking_data_directory)

    t0a = time.time()
    suite = BenchSuite_MTC()
    suite.setup_cache()
    t0b = time.time()

    timings = {}
    for component_name in suite.params:

        logger.warning(f"$$$$$$$$ {component_name} #1 $$$$$$$$")
        suite.setup(component_name)
        t1a = time.time()
        suite.time_component(component_name)
        t1b = time.time()
        suite.teardown(component_name)

        logger.warning(f"$$$$$$$$ {component_name} #2 $$$$$$$$")
        suite.setup(component_name)
        t2a = time.time()
        suite.time_component(component_name)
        t2b = time.time()
        suite.teardown(component_name)

        timings[component_name] = (
            str(timedelta(seconds=t1b-t1a)),
            str(timedelta(seconds=t2b-t2a)),
        )

    logger.warning(f"Time Base Setup: {timedelta(seconds=t0b-t0a)}")
    for component_name in suite.params:
        logger.warning(f"Time {component_name}: {timings[component_name]}")
