import argparse
import logging
import multiprocessing
import os
import sys
import time
from datetime import timedelta
from functools import partial

import yaml
from activitysim.benchmarking import componentwise, modify_yaml, workspace
from activitysim.cli.create import get_example

logger = logging.getLogger("activitysim.benchmarking")
benchmarking_directory = workspace.get_dir()


def f_setup_cache(
    EXAMPLE_NAME,
    COMPONENT_NAMES,
    BENCHMARK_SETTINGS,
    CONFIGS_DIRS=("configs",),
    DATA_DIR="data",
    OUTPUT_DIR="output",
    SETTINGS_FILENAME="settings.yaml",
    SKIP_COMPONENT_NAMES=None,
    NUM_PROCESSES=None,
    SKIM_CACHE=True,
):
    models = None
    try:
        if workspace.get_dir() is None:
            from asv.console import log

            for k, v in os.environ.items():
                log.error(f" env {k}: {v}")
            raise RuntimeError("workspace unavailable")
        os.makedirs(os.path.join(local_dir(), "models"), exist_ok=True)
        get_example(
            example_name=EXAMPLE_NAME,
            destination=os.path.join(local_dir(), "models"),
            benchmarking=True,
        )
        settings_filename = os.path.join(model_dir(EXAMPLE_NAME), SETTINGS_FILENAME)
        for config_settings_dir in CONFIGS_DIRS:
            settings_filename = os.path.join(
                model_dir(EXAMPLE_NAME), config_settings_dir, SETTINGS_FILENAME
            )
            if os.path.exists(settings_filename):
                with open(settings_filename, "rt") as f:
                    models = yaml.load(f, Loader=yaml.loader.SafeLoader).get("models")
                break
        if models is None and SETTINGS_FILENAME != "settings.yaml":
            for config_settings_dir in CONFIGS_DIRS:
                settings_filename = os.path.join(
                    model_dir(EXAMPLE_NAME), config_settings_dir, "settings.yaml"
                )
                if os.path.exists(settings_filename):
                    with open(settings_filename, "rt") as f:
                        models = yaml.load(f, Loader=yaml.loader.SafeLoader).get(
                            "models"
                        )
                    break
        if models is None:
            raise ValueError(f"missing list of models from configs/{SETTINGS_FILENAME}")
        last_component_to_benchmark = 0
        for cname in COMPONENT_NAMES:
            try:
                last_component_to_benchmark = max(
                    models.index(cname), last_component_to_benchmark
                )
            except ValueError:
                if cname not in models:
                    pass
                else:
                    raise
        pre_run_model_list = models[:last_component_to_benchmark]
        if SKIP_COMPONENT_NAMES is not None:
            for cname in SKIP_COMPONENT_NAMES:
                if cname in pre_run_model_list:
                    pre_run_model_list.remove(cname)
        settings_changes = dict(
            models=pre_run_model_list,
            checkpoints=True,
            trace_hh_id=None,
            chunk_training_mode="off",
        )
        if NUM_PROCESSES is not None:
            settings_changes["num_processes"] = NUM_PROCESSES
        modify_yaml(
            settings_filename,
            **BENCHMARK_SETTINGS,
            **settings_changes,
        )
        for config_network_los_dir in CONFIGS_DIRS:
            network_los_filename = os.path.join(
                model_dir(EXAMPLE_NAME), config_network_los_dir, "network_los.yaml"
            )
            if os.path.exists(network_los_filename):
                modify_yaml(
                    network_los_filename,
                    read_skim_cache=SKIM_CACHE,
                    write_skim_cache=SKIM_CACHE,
                )
                break
        os.makedirs(os.path.join(model_dir(EXAMPLE_NAME), OUTPUT_DIR), exist_ok=True)

        # Running the model through all the steps and checkpointing everywhere is
        # expensive and only needs to be run once.  Once it is done we will write
        # out a completion token file to indicate to future benchmark attempts
        # that this does not need to be repeated.  Developers should manually
        # delete the token (or the whole model file) when a structural change
        # in the model happens such that re-checkpointing is needed (this should
        # happen rarely).
        token_file = os.path.join(
            model_dir(EXAMPLE_NAME), OUTPUT_DIR, "benchmark-setup-token.txt"
        )
        if not os.path.exists(token_file):
            try:
                componentwise.pre_run(
                    model_dir(EXAMPLE_NAME),
                    CONFIGS_DIRS,
                    DATA_DIR,
                    OUTPUT_DIR,
                    SETTINGS_FILENAME,
                )
            except Exception as err:
                with open(token_file, "wt") as f:
                    f.write(f"error {err}")
                raise
            else:
                with open(token_file, "wt") as f:
                    # We write the commit into the token, in case that is useful
                    # to developers to decide if the checkpointed pipeline is
                    # out of date.
                    asv_commit = os.environ.get("ASV_COMMIT", "ASV_COMMIT_UNKNOWN")
                    f.write(asv_commit)

    except Exception as err:
        import traceback

        traceback.print_exc()
        raise


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
                self.component_name,
                model_dir(EXAMPLE_NAME),
                PRELOAD_INJECTABLES,
                CONFIGS_DIRS,
                DATA_DIR,
                OUTPUT_DIR,
            )

        def teardown(self):
            componentwise.teardown_component(self.component_name)

        def time_component(self):
            componentwise.run_component(self.component_name)

        # time_component.benchmark_name = f"{__name__}.time_component.{componentname}"
        time_component.pretty_name = f"{EXAMPLE_NAME}:{componentname}"

    ComponentTiming.__name__ = f"{componentname}"

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
    SKIM_CACHE=True,
    MAX_PROCESSES=10,
    HOUSEHOLD_SAMPLE=None,
):
    class mp_complete:
        repeat = 1
        number = 1
        timeout = TIMEOUT * 100
        component_names = COMPONENT_NAMES
        benchmark_settings = BENCHMARK_SETTINGS.copy()
        skim_cache = SKIM_CACHE

        def setup(self):
            # The output directory is changed by appending MP, to ensure
            # that we do not overwrite the checkpointed pipeline used by
            # single-component benchmarks.
            f_setup_cache(
                EXAMPLE_NAME,
                self.component_names,
                self.benchmark_settings,
                CONFIGS_DIRS,
                DATA_DIR,
                OUTPUT_DIR + "MP",
                SETTINGS_FILENAME=SETTINGS_FILENAME,
                NUM_PROCESSES=max(
                    min(multiprocessing.cpu_count() - 2, MAX_PROCESSES),
                    2,
                ),
                SKIM_CACHE=self.skim_cache,
            )

        def time_complete(self):
            INJECTABLES = [
                "data_dir",
                "configs_dir",
                "output_dir",
                "settings_file_name",
            ]
            from activitysim.core import config, inject, mp_tasks, pipeline

            injectables = {k: inject.get_injectable(k) for k in INJECTABLES}
            mp_tasks.run_multiprocess(injectables)
            assert not pipeline.is_open()
            if config.setting("cleanup_pipeline_after_run", False):
                pipeline.cleanup_pipeline()

        time_complete.pretty_name = f"{EXAMPLE_NAME}:MP-Complete"

    return mp_complete


def apply_template(
    GLOBALS,
    EXAMPLE_NAME,
    CONFIGS_DIRS,
    DATA_DIR,
    OUTPUT_DIR,
    PRELOAD_INJECTABLES,
    REPEAT,
    NUMBER,
    TIMEOUT,
    COMPONENT_NAMES,
    BENCHMARK_SETTINGS,
    SKIM_CACHE=True,
    MP_SAMPLE_SIZE=0,
    MAX_PROCESSES=10,
):
    # def setup_cache():
    #     f_setup_cache(
    #         EXAMPLE_NAME, COMPONENT_NAMES, BENCHMARK_SETTINGS,
    #         CONFIGS_DIRS, DATA_DIR, OUTPUT_DIR,
    #         SKIM_CACHE=SKIM_CACHE,
    #     )
    #
    # GLOBALS["setup_cache"] = setup_cache

    for cname in COMPONENT_NAMES:
        GLOBALS[cname] = generate_component_timings(
            cname,
            EXAMPLE_NAME,
            CONFIGS_DIRS,
            DATA_DIR,
            OUTPUT_DIR,
            PRELOAD_INJECTABLES,
            REPEAT,
            NUMBER,
            TIMEOUT,
        )

    BENCHMARK_SETTINGS_COMPLETE = BENCHMARK_SETTINGS.copy()
    BENCHMARK_SETTINGS_COMPLETE["households_sample_size"] = MP_SAMPLE_SIZE

    GLOBALS[f"mp_complete"] = generate_complete(
        EXAMPLE_NAME,
        CONFIGS_DIRS,
        DATA_DIR,
        OUTPUT_DIR,
        TIMEOUT,
        COMPONENT_NAMES,
        BENCHMARK_SETTINGS_COMPLETE,
        SKIM_CACHE=SKIM_CACHE,
        MAX_PROCESSES=MAX_PROCESSES,
    )
