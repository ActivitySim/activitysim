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

def f_setup_cache(EXAMPLE_NAME, COMPONENT_NAMES, BENCHMARK_SETTINGS):

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
    settings_filename = os.path.join(model_dir(EXAMPLE_NAME), "configs", "settings.yaml")
    with open(settings_filename, 'rt') as f:
        models = yaml.load(f, Loader=yaml.loader.SafeLoader).get('models')

    last_component_to_benchmark = 0
    for cname in COMPONENT_NAMES:
        last_component_to_benchmark = max(
            models.index(cname),
            last_component_to_benchmark
        )
    pre_run_model_list = models[:last_component_to_benchmark]
    modify_yaml(
        os.path.join(model_dir(EXAMPLE_NAME), "configs", "settings.yaml"),
        **BENCHMARK_SETTINGS,
        models=pre_run_model_list,
        checkpoints=True,
        trace_hh_id=None,
        chunk_training_mode='off',
    )
    modify_yaml(
        os.path.join(model_dir(EXAMPLE_NAME), "configs", "network_los.yaml"),
        read_skim_cache=True,
    )
    componentwise.pre_run(model_dir(EXAMPLE_NAME))


def local_dir():
    if benchmarking_directory is not None:
        return benchmarking_directory
    return os.getcwd()


def model_dir(example_name):
    return os.path.join(local_dir(), "models", example_name)


def generate_component_timings(
        componentname,
        EXAMPLE_NAME,
        PRELOAD_INJECTABLES,
        REPEAT,
        NUMBER,
        TIMEOUT,
):

    class ComponentTiming:
        component_name = componentname
        repeat = REPEAT
        number = NUMBER
        timeout = TIMEOUT
        def setup(self):
            componentwise.setup_component(self.component_name, model_dir(EXAMPLE_NAME), PRELOAD_INJECTABLES)
        def teardown(self):
            componentwise.teardown_component(self.component_name)
        def time_component(self):
            componentwise.run_component(self.component_name)
        #time_component.benchmark_name = f"{__name__}.time_component.{componentname}"
        time_component.pretty_name = f"{EXAMPLE_NAME}:{componentname}"

    ComponentTiming.__name__ = f"time_{componentname}"

    return ComponentTiming

