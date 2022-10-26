import argparse
import os
import pkg_resources
import yaml
import pandas as pd
import shutil
from activitysim.cli.run import add_run_args, run
from activitysim.core.util import named_product

START_ITER = 0
SAMPLING_PARAMS = {
    "DESTINATION_SAMPLE_SIZE": [0.1, 1 / 3, 2 / 3, 0],
    "ORIGIN_SAMPLE_SIZE": [0.1, 1 / 3, 2 / 3, 0],
    "ORIGIN_SAMPLE_METHOD": [None],  # , 'kmeans']
}


def integer_params(params):
    n_zones = count_lines_enumerate(base_path("data_2/land_use.csv"))

    d_zones = 1 if params.DESTINATION_SAMPLE_SIZE > 1 else n_zones
    o_zones = 1 if params.ORIGIN_SAMPLE_SIZE > 1 else n_zones

    params.DESTINATION_SAMPLE_SIZE = round(params.DESTINATION_SAMPLE_SIZE * d_zones)
    params.ORIGIN_SAMPLE_SIZE = round(params.ORIGIN_SAMPLE_SIZE * o_zones)

    return params


def base_path(dirname):
    resource = os.path.join("examples", "placeholder_sandag_2_zone", dirname)
    return pkg_resources.resource_filename("activitysim", resource)


def extended_path(dirname):
    resource = os.path.join("examples", "placeholder_sandag_2_zone_extended", dirname)
    return pkg_resources.resource_filename("activitysim", resource)


def run_model():
    parser = argparse.ArgumentParser()
    add_run_args(parser)
    args = parser.parse_args()

    # add in the arguments
    args.config = [
        extended_path("configs_mp"),
        extended_path("configs"),
        base_path("configs_2_zone"),
        base_path("placeholder_psrc/configs"),
    ]
    args.output = extended_path("output")
    args.data = [extended_path("data"), base_path("data_2")]
    run(args)


def count_lines_enumerate(file_name):
    fp = open(file_name, "r")
    line_count = list(enumerate(fp))[-1][0]
    return line_count


def update_configs(scene, model_settings, config_path):
    # Update the model settings
    scene = integer_params(scene)
    model_settings["DESTINATION_SAMPLE_SIZE"] = scene.DESTINATION_SAMPLE_SIZE
    model_settings["ORIGIN_SAMPLE_SIZE"] = scene.ORIGIN_SAMPLE_SIZE
    model_settings["ORIGIN_SAMPLE_METHOD"] = scene.ORIGIN_SAMPLE_METHOD

    with open(config_path, "w") as file:
        yaml.dump(model_settings, file)

    return model_settings


def make_scene_name(it, params):
    d_samp = params["DESTINATION_SAMPLE_SIZE"]
    o_samp = params["ORIGIN_SAMPLE_SIZE"]
    method = params["ORIGIN_SAMPLE_METHOD"]

    scene_name = "scene-{}_dsamp-{}_osamp-{}_method-{}".format(
        it + START_ITER,
        d_samp,
        o_samp,
        method,
    )

    return scene_name


def copy_output(scene_name, model_settings):

    scene_dir_name = os.path.join("scenarios_output", scene_name)

    if os.path.exists(extended_path(scene_dir_name)):
        shutil.rmtree(extended_path(scene_dir_name))
    os.makedirs(extended_path(scene_dir_name))

    log = pd.read_csv("scenarios_output/log.csv")
    filt = (
        (log.DESTINATION_SAMPLE_SIZE == model_settings["DESTINATION_SAMPLE_SIZE"])
        & (log.ORIGIN_SAMPLE_SIZE == model_settings["ORIGIN_SAMPLE_SIZE"])
        & (log.ORIGIN_SAMPLE_METHOD == model_settings["ORIGIN_SAMPLE_METHOD"])
    )
    log.loc[filt, "COMPLETED_ID"] = scene_name
    log.to_csv("scenarios_output/log.csv", index=False)

    files_list = [
        x
        for x in os.listdir(extended_path("output"))
        if "pipeline" not in x and "cache" not in x
    ]

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

    if not os.path.exists(extended_path("scenarios_output")):
        os.makedirs(extended_path("scenarios_output"))

    if os.path.exists(extended_path("scenarios_output/log.csv")):
        log = pd.read_csv(extended_path("scenarios_output/log.csv"))
        # assert scenarios[['DESTINATION_SAMPLE_SIZE','ORIGIN_SAMPLE_SIZE', 'ORIGIN_SAMPLE_METHOD']].equals(
        #     log[['DESTINATION_SAMPLE_SIZE','ORIGIN_SAMPLE_SIZE', 'ORIGIN_SAMPLE_METHOD']]
        # )
        scenarios = log
    else:
        scenarios = pd.DataFrame(named_product(**SAMPLING_PARAMS))
        scenarios["COMPLETED_ID"] = ""
        scenarios["SKIP"] = False
        scenarios.to_csv(extended_path("scenarios_output/log.csv"), index=False)

    for it, scene in scenarios.iterrows():
        # Update model settings
        model_settings = update_configs(scene, model_settings, config_path)

        # Check if already run
        scene_name = make_scene_name(it, scene)
        scene_dir_name = extended_path(os.path.join("scenarios_output", scene_name))

        if (
            any(scenarios.COMPLETED_ID == scene_name)
            and os.path.exists(scene_dir_name)
            or scene.SKIP
        ):
            continue

        # Run the model
        print(
            f"Running model {it} of {len(scenarios.index)}: {chr(10)}"
            + f"{chr(10)}".join(
                [f"{var}={model_settings[var]}" for var in SAMPLING_PARAMS.keys()]
            )
        )
        try:
            run_model()
            # Copy results to named folder
            copy_output(scene_name, model_settings)
        except:
            print(f"Failed on scene {scene_name}")


if __name__ == "__main__":
    run_scenarios()
