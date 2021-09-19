import argparse
import sys
import os
import logging
import time
import yaml
import multiprocessing
from datetime import timedelta
from functools import partial
from activitysim.benchmarking import componentwise, modify_yaml, workspace
from activitysim.cli.create import get_example

logger = logging.getLogger("activitysim.benchmarking")
benchmarking_directory = workspace.get_dir()

def f_setup_cache(
        EXAMPLE_NAME,
        COMPONENT_NAMES,
        BENCHMARK_SETTINGS,
        CONFIGS_DIRS=("configs",),
        DATA_DIR='data',
        OUTPUT_DIR='output',
        SETTINGS_FILENAME="settings.yaml",
        SKIP_COMPONENT_NAMES=None,
        NUM_PROCESSES=None,
        PIPELINE_HASH=None,
):

    if workspace.get_dir() is None:
        from asv.console import log
        for k,v in os.environ.items():
            log.error(f" env {k}: {v}")
        raise RuntimeError("workspace unavailable")
    os.makedirs(os.path.join(local_dir(), "models"), exist_ok=True)
    get_example(
        example_name=EXAMPLE_NAME,
        destination=os.path.join(local_dir(), "models"),
        benchmarking=True,
    )
    models = None
    settings_filename = os.path.join(model_dir(EXAMPLE_NAME), SETTINGS_FILENAME)
    for config_settings_dir in CONFIGS_DIRS:
        settings_filename = os.path.join(model_dir(EXAMPLE_NAME), config_settings_dir, SETTINGS_FILENAME)
        if os.path.exists(settings_filename):
            if NUM_PROCESSES is not None:
                modify_yaml(settings_filename, num_processes=NUM_PROCESSES)
            with open(settings_filename, 'rt') as f:
                models = yaml.load(f, Loader=yaml.loader.SafeLoader).get('models')
            break
    if models is None and SETTINGS_FILENAME != "settings.yaml":
        for config_settings_dir in CONFIGS_DIRS:
            settings_filename = os.path.join(model_dir(EXAMPLE_NAME), config_settings_dir, "settings.yaml")
            if os.path.exists(settings_filename):
                with open(settings_filename, 'rt') as f:
                    models = yaml.load(f, Loader=yaml.loader.SafeLoader).get('models')
                break
    if models is None:
        raise ValueError(f"missing list of models from configs/{SETTINGS_FILENAME}")
    last_component_to_benchmark = 0
    for cname in COMPONENT_NAMES:
        last_component_to_benchmark = max(
            models.index(cname),
            last_component_to_benchmark
        )
    pre_run_model_list = models[:last_component_to_benchmark]
    if SKIP_COMPONENT_NAMES is not None:
        for cname in SKIP_COMPONENT_NAMES:
            if cname in pre_run_model_list:
                pre_run_model_list.remove(cname)
    modify_yaml(
        settings_filename,
        **BENCHMARK_SETTINGS,
        models=pre_run_model_list,
        checkpoints=True,
        trace_hh_id=None,
        chunk_training_mode='off',
    )
    for config_network_los_dir in CONFIGS_DIRS:
        network_los_filename = os.path.join(model_dir(EXAMPLE_NAME), config_network_los_dir, "network_los.yaml")
        if os.path.exists(network_los_filename):
            modify_yaml(
                network_los_filename,
                read_skim_cache=True,
                write_skim_cache=True,
            )
            break
    os.makedirs(os.path.join(model_dir(EXAMPLE_NAME), OUTPUT_DIR), exist_ok=True)
    use_prepared_pipeline = False
    asv_commit = os.environ.get('ASV_COMMIT', 'ASV_COMMIT_UNKNOWN')
    token_file = os.path.join(model_dir(EXAMPLE_NAME), OUTPUT_DIR, 'benchmark-setup-token.txt')
    if os.path.exists(token_file):
        with open(token_file, 'rt') as f:
            token = f.read()
        if token == asv_commit:
            use_prepared_pipeline = True
    if not use_prepared_pipeline:
        try:
            componentwise.pre_run(model_dir(EXAMPLE_NAME), CONFIGS_DIRS, DATA_DIR, OUTPUT_DIR, SETTINGS_FILENAME)
        except Exception as err:
            with open(token_file, 'wt') as f:
                f.write(f"error {err}")
            raise
        else:
            with open(token_file, 'wt') as f:
                f.write(asv_commit)


def local_dir():
    if benchmarking_directory is not None:
        return benchmarking_directory
    return os.getcwd()


def model_dir(example_name):
    return os.path.join(local_dir(), "models", example_name)


def generate_component_timings(
        componentname,
        EXAMPLE_NAME,
        CONFIGS_DIRS,
        DATA_DIR,
        OUTPUT_DIR,
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
            componentwise.setup_component(
                self.component_name, model_dir(EXAMPLE_NAME), PRELOAD_INJECTABLES,
                CONFIGS_DIRS, DATA_DIR, OUTPUT_DIR,
            )
        def teardown(self):
            componentwise.teardown_component(self.component_name)
        def time_component(self):
            componentwise.run_component(self.component_name)
        #time_component.benchmark_name = f"{__name__}.time_component.{componentname}"
        time_component.pretty_name = f"{EXAMPLE_NAME}:{componentname}"

    ComponentTiming.__name__ = f"time_{componentname}"

    return ComponentTiming


def generate_complete(
        EXAMPLE_NAME,
        CONFIGS_DIRS,
        DATA_DIR,
        OUTPUT_DIR,
        TIMEOUT,
        COMPONENT_NAMES,
        BENCHMARK_SETTINGS,
        SETTINGS_FILENAME="settings_mp.yaml",
):

    class time_mp_complete:
        repeat = 1
        number = 1
        timeout = TIMEOUT*100
        component_names = COMPONENT_NAMES
        benchmark_settings = BENCHMARK_SETTINGS

        def setup(self):
            print("<Running MP Complete> SETUP")
            f_setup_cache(
                EXAMPLE_NAME, self.component_names, self.benchmark_settings,
                CONFIGS_DIRS, DATA_DIR, OUTPUT_DIR + "MP",
                SETTINGS_FILENAME=SETTINGS_FILENAME,
                NUM_PROCESSES=max(multiprocessing.cpu_count()-2, 2),
            )
            print("<End MP Complete> SETUP")

        def time_complete(self):
            print("<Running MP Complete>")
            INJECTABLES = ['data_dir', 'configs_dir', 'output_dir', 'settings_file_name']
            from activitysim.core import mp_tasks, inject, pipeline, config
            injectables = {k: inject.get_injectable(k) for k in INJECTABLES}
            mp_tasks.run_multiprocess(injectables)
            assert not pipeline.is_open()
            if config.setting('cleanup_pipeline_after_run', False):
                pipeline.cleanup_pipeline()
            print("<End MP Complete>")

        time_complete.pretty_name = f"{EXAMPLE_NAME}:MP-Complete"

    return time_mp_complete