import argparse
from genericpath import isfile
import os
import pkg_resources
import yaml
import pandas as pd
import shutil
from activitysim.cli.run import add_run_args, run
from activitysim.core.util import named_product


SAMPLING_PARAMS = {
    "DESTINATION_SAMPLE_SIZE": [0.1, 1 / 3, 2 / 3, 0],
    "ORIGIN_SAMPLE_SIZE": [0.1, 1 / 3, 2 / 3, 0],
    "ORIGIN_SAMPLE_METHOD": [None],  # , 'kmeans']
}


def base_path(dirname):
    resource = os.path.join("examples", "prototype_mtc", dirname)
    return pkg_resources.resource_filename("activitysim", resource)


def extended_path(dirname):
    resource = os.path.join("examples", "prototype_mtc_extended", dirname)
    return pkg_resources.resource_filename("activitysim", resource)


def run_model():
    parser = argparse.ArgumentParser()
    add_run_args(parser)
    args = parser.parse_args()

    # add in the arguments
    args.config = [
        extended_path("configs_mp"),
        extended_path("configs"),
        base_path("configs"),
    ]
    args.output = extended_path("output")
    args.data = base_path("data")
    run(args)


def count_lines_enumerate(file_name):
    fp = open(file_name, "r")
    line_count = list(enumerate(fp))[-1][0]
    return line_count


def update_configs(scene, model_settings, config_path):
    n_zones = count_lines_enumerate(base_path("data/land_use.csv"))

    d_zones = 1 if scene.DESTINATION_SAMPLE_SIZE > 1 else n_zones
    o_zones = 1 if scene.ORIGIN_SAMPLE_SIZE > 1 else n_zones

    d_size = round(scene.DESTINATION_SAMPLE_SIZE * d_zones)
    o_size = round(scene.ORIGIN_SAMPLE_SIZE * o_zones)
    method = scene.ORIGIN_SAMPLE_METHOD

    # Update the model settings
    model_settings["DESTINATION_SAMPLE_SIZE"] = d_size
    model_settings["ORIGIN_SAMPLE_SIZE"] = o_size
    model_settings["ORIGIN_SAMPLE_METHOD"] = method

    with open(config_path, "w") as file:
        yaml.dump(model_settings, file)

    return model_settings


def copy_output(iter, model_settings):
    scene_dir_name = "scenarios_output/scene-{}_dsamp-{}_osamp-{}_method-{}".format(
        iter,
        model_settings["DESTINATION_SAMPLE_SIZE"],
        model_settings["ORIGIN_SAMPLE_SIZE"],
        model_settings["ORIGIN_SAMPLE_METHOD"],
    )

    if os.path.exists(extended_path(scene_dir_name)):
        shutil.rmtree(extended_path(scene_dir_name))
    os.makedirs(extended_path(scene_dir_name))

    files_list = [x for x in os.listdir(extended_path('output')) if 'pipeline' not in x and 'cache' not in x]

    for file in files_list:
        copyargs = {
            "src": extended_path(os.path.join("output", file)),
            "dst": extended_path(os.path.join(scene_dir_name, file)),
        }
        if os.path.isfile(copyargs["src"]):
            shutil.copy(**copyargs)
        else:
            if os.path.exists(copyargs["dst"]):
                shutil.rmtree(copyargs["dst"])
            shutil.copytree(**copyargs)
    return


def run_scenarios():
    config_path = extended_path("configs/disaggregate_accessibility.yaml")
    with open(config_path) as file:
        model_settings = yaml.load(file, Loader=yaml.FullLoader)

    scenarios = pd.DataFrame(named_product(**SAMPLING_PARAMS))
    for iter, scene in scenarios.iterrows():
        # Update model settings
        model_settings = update_configs(scene, model_settings, config_path)
        # Run the model
        print(
            f"Running model {iter} of {len(scenarios.index)}: {chr(10)}"
            + f"{chr(10)}".join(
                [f"{var}={model_settings[var]}" for var in SAMPLING_PARAMS.keys()]
            )
        )
        try:
            run_model()
            # Copy results to named folder
            copy_output(iter, model_settings)
        except:
            print(f"Failed on scene {iter}")


if __name__ == "__main__":
    run_scenarios()
