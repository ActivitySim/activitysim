import os

import yaml

from ..wrapping import workstep


@workstep("data_dictionary")
def load_data_dictionary(
    config_dirs,
    data_dict_filename="data_dictionary.yaml",
    cwd=".",
):
    if isinstance(config_dirs, str):
        config_dirs = [config_dirs]
    dd = {}
    for config_dir in config_dirs:
        if os.path.isdir(os.path.join(cwd, config_dir)):
            f = os.path.join(cwd, config_dir, data_dict_filename)
            if os.path.exists(f):
                with open(f, "rt") as stream:
                    dd.update(yaml.safe_load(stream))
    return dd
