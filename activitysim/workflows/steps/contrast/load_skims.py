import os
from pypyr.context import Context
from ..progression import reset_progress_step
from ..error_handler import error_logging
from pathlib import Path
from activitysim.standalone.skims import load_skims
from activitysim.standalone.utils import chdir
import logging

logger = logging.getLogger(__name__)

@error_logging
def run_step(context: Context) -> None:
    reset_progress_step(description="load skims")

    context.assert_key_has_value(key='common_directory', caller=__name__)
    common_directory = context.get_formatted('common_directory')
    config_dirs = context.get_formatted('config_dirs')
    if isinstance(config_dirs, str):
        config_dirs = [config_dirs]
    data_dir = context.get_formatted('data_dir')
    with chdir(common_directory):
        network_los_file = None
        for config_dir in config_dirs:
            network_los_file = os.path.join(config_dir, 'network_los.yaml')
            if os.path.exists(network_los_file):
                break
        if network_los_file is None:
            raise FileNotFoundError("<<config_dir>>/network_los.yaml")
        if isinstance(data_dir, (str, Path)) and isinstance(network_los_file, (str, Path)):
            skims = load_skims(network_los_file, data_dir)
        else:
            skims = {}
            for k in data_dir.keys():
                skims[k] = load_skims(network_los_file, data_dir[k])
                # TODO: allow for different network_los_file

    context['skims'] = skims
