import argparse
import os
import logging

logger = logging.getLogger("activitysim.benchmarking")


from activitysim.benchmarking.config_editing import modify_settings

benchmarking_data_directory = "/tmp/asim-bench"
benchmarking_settings_mtc = dict(
    households_sample_size=20_000,
    benchmarking='workplace_location',
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

def pull_mtc():
    # replicate function of `activitysim create -e example_mtc_full`
    from activitysim.cli.create import get_example
    get_example(
        "example_mtc_full",
        os.path.join(benchmarking_data_directory, "example_mtc_full")
    )

def setup_mtc_component(component_name):
    pre_run_model_list = model_list_mtc[:model_list_mtc.index(component_name)]
    modify_settings(
        os.path.join(benchmarking_data_directory, "example_mtc_full", "configs", "settings.yaml"),
        **benchmarking_settings_mtc,
        models=pre_run_model_list,
        checkpoints=pre_run_model_list[-1:],
    )
    from activitysim.cli.run import run, add_run_args
    cmd_line_args = [
        '--config', os.path.join(benchmarking_data_directory, "example_mtc_full", "configs"),
        '--data', os.path.join(benchmarking_data_directory, "example_mtc_full", "data"),
        '--output', os.path.join(benchmarking_data_directory, "example_mtc_full", "output"),
    ]
    parser = argparse.ArgumentParser()
    add_run_args(parser)
    args = parser.parse_args(cmd_line_args)
    return run(args)


def run_mtc_component(component_name):
    this_run_model_list = model_list_mtc[:model_list_mtc.index(component_name)+1]
    modify_settings(
        os.path.join(benchmarking_data_directory, "example_mtc_full", "configs", "settings.yaml"),
        **benchmarking_settings_mtc,
        checkpoints=False,
        models=this_run_model_list,
        resume_after=this_run_model_list[-2],
    )
    from activitysim.benchmarking.componentwise import run_component, add_run_args
    cmd_line_args = [
        '--config', os.path.join(benchmarking_data_directory, "example_mtc_full", "configs"),
        '--data', os.path.join(benchmarking_data_directory, "example_mtc_full", "data"),
        '--output', os.path.join(benchmarking_data_directory, "example_mtc_full", "output"),
    ]
    parser = argparse.ArgumentParser()
    add_run_args(parser)
    args = parser.parse_args(cmd_line_args)
    return run_component(args)





if __name__ == '__main__':
    #pull_mtc()
    #setup_mtc_component("workplace_location")
    run_mtc_component("workplace_location")
    logger.warning("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    logger.warning("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    logger.warning("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    run_mtc_component("workplace_location")

