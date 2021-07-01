import argparse
import os
import logging
import time
from datetime import timedelta
from activitysim.benchmarking.componentwise import run_component, add_run_args, prep_component, after_component
from activitysim.cli.create import get_example

logger = logging.getLogger("activitysim.benchmarking")


from activitysim.benchmarking.config_editing import modify_yaml

benchmarking_data_directory = "/tmp/asim-bench"
benchmarking_settings_mtc = dict(
    households_sample_size=1_000,
)

model_list_mtc = [
    'initialize_landuse',
    'initialize_households',
    'compute_accessibility',
    'school_location',
    'workplace_location',
    'auto_ownership_simulate',
    'free_parking',
    'cdap_simulate',
    'mandatory_tour_frequency',
    'mandatory_tour_scheduling',
    'joint_tour_frequency',
    'joint_tour_composition',
    'joint_tour_participation',
    'joint_tour_destination',
    'joint_tour_scheduling',
    'non_mandatory_tour_frequency',
    'non_mandatory_tour_destination',
    'non_mandatory_tour_scheduling',
    'tour_mode_choice_simulate',
    'atwork_subtour_frequency',
    'atwork_subtour_destination',
    'atwork_subtour_scheduling',
    'atwork_subtour_mode_choice',
    'stop_frequency',
    'trip_purpose',
    'trip_destination',
    'trip_purpose_and_destination',
    'trip_scheduling',
    'trip_mode_choice',
    'write_data_dictionary',
    'track_skim_usage',
    'write_trip_matrices',
    'write_tables',
]

class BenchSuite_MTC:

    example_name = "example_mtc_full"

    # any settings to override in the example's usual settings file
    benchmark_settings = {
        'households_sample_size': 1_000,
    }

    # the component names to be benchmarked
    params = [
        "workplace_location",
    ]

    def setup_cache(self):
        get_example(
            example_name=self.example_name,
            destination='.',
        )
        last_component_to_benchmark = 0
        for component_name in self.params:
            last_component_to_benchmark = max(
                model_list_mtc.index(component_name),
                last_component_to_benchmark
            )
        pre_run_model_list = model_list_mtc[:last_component_to_benchmark]
        modify_yaml(
            os.path.join("example_mtc_full", "configs", "settings.yaml"),
            **self.benchmark_settings,
            models=pre_run_model_list,
            checkpoints=True,
        )
        modify_yaml(
            os.path.join("example_mtc_full", "configs", "network_los.yaml"),
            read_skim_cache=True,
        )
        from activitysim.cli.run import run, add_run_args
        cmd_line_args = [
            '--config', os.path.join("example_mtc_full", "configs"),
            '--data', os.path.join("example_mtc_full", "data"),
            '--output', os.path.join("example_mtc_full", "output"),
        ]
        parser = argparse.ArgumentParser()
        add_run_args(parser)
        args = parser.parse_args(cmd_line_args)
        return run(args)


    def setup_component(self, component_name):
        this_run_model_list = model_list_mtc[:model_list_mtc.index(component_name)+1]
        modify_yaml(
            os.path.join("example_mtc_full", "configs", "settings.yaml"),
            **benchmarking_settings_mtc,
            benchmarking='workplace_location',
            checkpoints=False,
            #models=this_run_model_list,
            resume_after=this_run_model_list[-2],
        )
        cmd_line_args = [
            '--config', os.path.join("example_mtc_full", "configs"),
            '--data', os.path.join("example_mtc_full", "data"),
            '--output', os.path.join("example_mtc_full", "output"),
        ]
        parser = argparse.ArgumentParser()
        add_run_args(parser)
        args = parser.parse_args(cmd_line_args)
        prep_component(args, component_name)

    def time_component(self, component_name):
        return run_component(component_name)

    def teardown_component(self, component_name):
        after_component()




if __name__ == '__main__':

    t0 = time.time()
    os.chdir(benchmarking_data_directory)

    component_name = "workplace_location"
    suite = BenchSuite_MTC()

    suite.setup_cache()
    t1 = time.time()
    logger.warning("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    logger.warning("$ 0 $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    logger.warning("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    suite.setup_component(component_name)
    t2a = time.time()
    suite.time_component(component_name)
    t2b = time.time()
    suite.teardown_component(component_name)
    logger.warning("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    logger.warning("$ 1 $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    logger.warning("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    suite.setup_component(component_name)
    t3a = time.time()
    suite.time_component(component_name)
    t3b = time.time()
    suite.teardown_component(component_name)

    logger.warning(f"Time Base Setup: {timedelta(seconds=t1-t0)}")

    logger.warning(f"Time Setup 1: {timedelta(seconds=t2a-t1)}")
    logger.warning(f"Time Setup 2: {timedelta(seconds=t3a-t2b)}")

    logger.warning(f"Time Run 1: {timedelta(seconds=t2b-t2a)}")
    logger.warning(f"Time Run 2: {timedelta(seconds=t3b-t3a)}")

