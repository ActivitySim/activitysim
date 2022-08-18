# ActivitySim
# See full license in LICENSE.txt.
import argparse
import logging
import os
import sys
import warnings

import numpy as np

from activitysim.core import chunk, config, inject, mem, pipeline, tracing

logger = logging.getLogger(__name__)


INJECTABLES = ["data_dir", "configs_dir", "output_dir", "settings_file_name"]


def add_run_args(parser, multiprocess=True):
    """Run command args"""
    parser.add_argument(
        "-w",
        "--working_dir",
        type=str,
        metavar="PATH",
        help="path to example/project directory (default: %s)" % os.getcwd(),
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        action="append",
        metavar="PATH",
        help="path to config dir",
    )
    parser.add_argument(
        "-o", "--output", type=str, metavar="PATH", help="path to output dir"
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        action="append",
        metavar="PATH",
        help="path to data dir",
    )
    parser.add_argument(
        "-r", "--resume", type=str, metavar="STEPNAME", help="resume after step"
    )
    parser.add_argument(
        "-p", "--pipeline", type=str, metavar="FILE", help="pipeline file name"
    )
    parser.add_argument(
        "-s", "--settings_file", type=str, metavar="FILE", help="settings file name"
    )
    parser.add_argument(
        "-g", "--chunk_size", type=int, metavar="BYTES", help="chunk size"
    )
    parser.add_argument(
        "--chunk_training_mode",
        type=str,
        help="chunk training mode, one of [training, adaptive, production, disabled]",
    )
    parser.add_argument(
        "--households_sample_size", type=int, metavar="N", help="households sample size"
    )

    if multiprocess:
        parser.add_argument(
            "-m",
            "--multiprocess",
            default=False,
            const=-1,
            metavar="(N)",
            nargs="?",
            type=int,
            help="run multiprocess. Adds configs_mp settings"
            " by default. Optionally give a number of processes,"
            " which will override the settings file.",
        )


def validate_injectable(name):
    try:
        dir_paths = inject.get_injectable(name)
    except RuntimeError:
        # injectable is missing, meaning is hasn't been explicitly set
        # and defaults cannot be found.
        sys.exit(
            "Error: please specify either a --working_dir "
            "containing 'configs', 'data', and 'output' folders "
            "or all three of --config, --data, and --output"
        )

    dir_paths = [dir_paths] if isinstance(dir_paths, str) else dir_paths

    for dir_path in dir_paths:
        if not os.path.exists(dir_path):
            sys.exit("Could not find %s '%s'" % (name, os.path.abspath(dir_path)))

    return dir_paths


def handle_standard_args(args, multiprocess=True):
    def inject_arg(name, value, cache=False):
        assert name in INJECTABLES
        inject.add_injectable(name, value, cache=cache)

    if args.working_dir:
        # activitysim will look in the current working directory for
        # 'configs', 'data', and 'output' folders by default
        os.chdir(args.working_dir)

    # settings_file_name should be cached or else it gets squashed by config.py
    if args.settings_file:
        inject_arg("settings_file_name", args.settings_file, cache=True)

    if args.config:
        inject_arg("configs_dir", args.config)

    if args.data:
        inject_arg("data_dir", args.data)

    if args.output:
        inject_arg("output_dir", args.output)

    if multiprocess and args.multiprocess:
        config_paths = validate_injectable("configs_dir")

        if not os.path.exists("configs_mp"):
            logger.warning("could not find 'configs_mp'. skipping...")
        else:
            logger.info("adding 'configs_mp' to config_dir list...")
            config_paths.insert(0, "configs_mp")
            inject_arg("configs_dir", config_paths)

        config.override_setting("multiprocess", True)
        if args.multiprocess > 0:
            config.override_setting("num_processes", args.multiprocess)

    if args.chunk_size:
        config.override_setting("chunk_size", int(args.chunk_size))
    if args.chunk_training_mode is not None:
        config.override_setting("chunk_training_mode", args.chunk_training_mode)
    if args.households_sample_size is not None:
        config.override_setting("households_sample_size", args.households_sample_size)

    for injectable in ["configs_dir", "data_dir", "output_dir"]:
        validate_injectable(injectable)

    if args.pipeline:
        inject.add_injectable("pipeline_file_name", args.pipeline)

    if args.resume:
        config.override_setting("resume_after", args.resume)


def cleanup_output_files():

    tracing.delete_trace_files()

    tracing.delete_output_files("h5")
    tracing.delete_output_files("csv")
    tracing.delete_output_files("txt")
    tracing.delete_output_files("yaml")
    tracing.delete_output_files("prof")
    tracing.delete_output_files("omx")


def run(args):
    """
    Run the models. Specify a project folder using the '--working_dir' option,
    or point to the config, data, and output folders directly with
    '--config', '--data', and '--output'. Both '--config' and '--data' can be
    specified multiple times. Directories listed first take precedence.

    returns:
        int: sys.exit exit code
    """

    # register abm steps and other abm-specific injectables
    # by default, assume we are running activitysim.abm
    # other callers (e.g. populationsim) will have to arrange to register their own steps and injectables
    # (presumably) in a custom run_simulation.py instead of using the 'activitysim run' command
    if not inject.is_injectable("preload_injectables"):
        from activitysim import (  # register abm steps and other abm-specific injectables
            abm,
        )

    tracing.config_logger(basic=True)
    handle_standard_args(args)  # possibly update injectables

    # legacy support for run_list setting nested 'models' and 'resume_after' settings
    if config.setting("run_list"):
        warnings.warn(
            "Support for 'run_list' settings group will be removed.\n"
            "The run_list.steps setting is renamed 'models'.\n"
            "The run_list.resume_after setting is renamed 'resume_after'.\n"
            "Specify both 'models' and 'resume_after' directly in settings config file.",
            FutureWarning,
        )
        run_list = config.setting("run_list")
        if "steps" in run_list:
            assert not config.setting(
                "models"
            ), f"Don't expect 'steps' in run_list and 'models' as stand-alone setting!"
            config.override_setting("models", run_list["steps"])

        if "resume_after" in run_list:
            assert not config.setting(
                "resume_after"
            ), f"Don't expect 'resume_after' both in run_list and as stand-alone setting!"
            config.override_setting("resume_after", run_list["resume_after"])

    # If you provide a resume_after argument to pipeline.run
    # the pipeline manager will attempt to load checkpointed tables from the checkpoint store
    # and resume pipeline processing on the next submodel step after the specified checkpoint
    resume_after = config.setting("resume_after", None)

    # cleanup if not resuming
    if not resume_after:
        cleanup_output_files()
    elif config.setting("cleanup_trace_files_on_resume", False):
        tracing.delete_trace_files()

    tracing.config_logger(basic=False)  # update using possibly new logging configs
    config.filter_warnings()
    logging.captureWarnings(capture=True)

    # directories
    for k in ["configs_dir", "settings_file_name", "data_dir", "output_dir"]:
        logger.info("SETTING %s: %s" % (k, inject.get_injectable(k, None)))

    log_settings = inject.get_injectable("log_settings", {})
    for k in log_settings:
        logger.info("SETTING %s: %s" % (k, config.setting(k)))

    # OMP_NUM_THREADS: openmp
    # OPENBLAS_NUM_THREADS: openblas
    # MKL_NUM_THREADS: mkl
    for env in ["MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
        logger.info(f"ENV {env}: {os.getenv(env)}")

    np_info_keys = [
        "atlas_blas_info",
        "atlas_blas_threads_info",
        "atlas_info",
        "atlas_threads_info",
        "blas_info",
        "blas_mkl_info",
        "blas_opt_info",
        "lapack_info",
        "lapack_mkl_info",
        "lapack_opt_info",
        "mkl_info",
    ]

    for cfg_key in np_info_keys:
        info = np.__config__.get_info(cfg_key)
        if info:
            for info_key in ["libraries"]:
                if info_key in info:
                    logger.info(f"NUMPY {cfg_key} {info_key}: {info[info_key]}")

    t0 = tracing.print_elapsed_time()

    try:
        if config.setting("multiprocess", False):
            logger.info("run multiprocess simulation")

            from activitysim.core import mp_tasks

            injectables = {k: inject.get_injectable(k) for k in INJECTABLES}
            mp_tasks.run_multiprocess(injectables)

            assert not pipeline.is_open()

            if config.setting("cleanup_pipeline_after_run", False):
                pipeline.cleanup_pipeline()

        else:
            logger.info("run single process simulation")

            pipeline.run(models=config.setting("models"), resume_after=resume_after)

            if config.setting("cleanup_pipeline_after_run", False):
                pipeline.cleanup_pipeline()  # has side effect of closing open pipeline
            else:
                pipeline.close_pipeline()

            mem.log_global_hwm()  # main process
    except Exception:
        # log time until error and the error traceback
        tracing.print_elapsed_time("all models until this error", t0)
        logger.exception("activitysim run encountered an unrecoverable error")
        raise

    chunk.consolidate_logs()
    mem.consolidate_logs()

    tracing.print_elapsed_time("all models", t0)

    return 0


if __name__ == "__main__":

    from activitysim import abm  # register injectables

    parser = argparse.ArgumentParser()
    add_run_args(parser)
    args = parser.parse_args()

    parser.parse_args(["--sum", "7", "-1", "42"])
    sys.exit(run(args))
