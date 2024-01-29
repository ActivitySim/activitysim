from __future__ import annotations

import glob
import logging
import logging.handlers
import os
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from activitysim.benchmarking import workspace
from activitysim.cli.create import get_example
from activitysim.cli.run import INJECTABLES, config
from activitysim.core import tracing, workflow

logger = logging.getLogger(__name__)


def reload_settings(state, settings_filename, **kwargs):
    settings = state.filesystem.read_settings_file(settings_filename, mandatory=True)
    for k in kwargs:
        settings[k] = kwargs[k]
    state.add_injectable("settings", settings)
    return settings


def component_logging(state: workflow.State, component_name):
    root_logger = logging.getLogger()

    CLOG_FMT = "%(asctime)s %(levelname)7s - %(name)s: %(message)s"

    logfilename = state.get_log_file_path(f"asv-{component_name}.log")

    # avoid creation of multiple file handlers for logging components
    # as we will re-enter this function for every component run
    for entry in root_logger.handlers:
        if (isinstance(entry, logging.handlers.RotatingFileHandler)) and (
            entry.formatter._fmt == CLOG_FMT
        ):
            return

    state.logging.config_logger(basic=True)
    file_handler = logging.handlers.RotatingFileHandler(
        filename=logfilename,
        mode="a",
        maxBytes=50_000_000,
        backupCount=5,
    )
    formatter = logging.Formatter(
        fmt=CLOG_FMT,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)


def setup_component(
    state,
    component_name,
    working_dir=".",
    preload_injectables=(),
    configs_dirs=("configs"),
    data_dir="data",
    output_dir="output",
    settings_filename="settings.yaml",
    **other_settings,
):
    """
    Prepare to benchmark a model component.

    This function sets up everything, opens the pipeline, and
    reloads table state from checkpoints of prior components.
    All this happens here, before the model component itself
    is actually executed inside the timed portion of the loop.
    """
    if isinstance(configs_dirs, str):
        configs_dirs = [configs_dirs]
    state.add_injectable(
        "configs_dir", [os.path.join(working_dir, i) for i in configs_dirs]
    )
    state.add_injectable("data_dir", os.path.join(working_dir, data_dir))
    state.add_injectable("output_dir", os.path.join(working_dir, output_dir))

    reload_settings(
        state,
        settings_filename,
        benchmarking=component_name,
        checkpoints=False,
        **other_settings,
    )

    component_logging(state, component_name)
    logger.info("connected to component logger")
    config.filter_warnings()
    logging.captureWarnings(capture=True)

    # register abm steps and other abm-specific injectables outside of
    # benchmark timing loop
    if "preload_injectables" not in state.context:
        logger.info("preload_injectables yes import")
        from activitysim import abm  # noqa: F401
    else:
        logger.info("preload_injectables no import")

    # Extract the resume_after argument based on the model immediately
    # prior to the component being benchmarked.
    models = state.settings.models
    try:
        component_index = models.index(component_name)
    except ValueError:
        # the last component to be benchmarked isn't included in the
        # pre-checkpointed model list, we just resume from the end
        component_index = len(models)
    if component_index:
        resume_after = models[component_index - 1]
    else:
        resume_after = None

    if state.settings.multiprocess:
        raise NotImplementedError(
            "multiprocess component benchmarking is not yet implemented"
        )
        # Component level timings for multiprocess benchmarking
        # are not generated using this code that re-runs individual
        # components.  Instead, those benchmarks are generated in
        # aggregate during setup and then extracted from logs later.
    else:
        state.checkpoint.restore(resume_after, mode="r")

    for k in preload_injectables:
        if state.get_injectable(k, None) is not None:
            logger.info("pre-loaded %s", k)

    # Directories Logging
    for k in ["configs_dir", "settings_file_name", "data_dir", "output_dir"]:
        logger.info(f"DIRECTORY {k}: {state.get_injectable(k, None)}")

    # Settings Logging
    log_settings = [
        "checkpoints",
        "chunk_training_mode",
        "chunk_size",
        "chunk_method",
        "trace_hh_id",
        "households_sample_size",
        "check_for_variability",
        "use_shadow_pricing",
        "want_dest_choice_sample_tables",
        "log_alt_losers",
        "sharrow",
    ]
    for k in log_settings:
        logger.info(f"SETTING {k}: {config.setting(k)}")

    logger.info("setup_component completed: %s", component_name)


def run_component(state, component_name):
    logger.info("run_component: %s", component_name)
    try:
        if state.settings.multiprocess:
            raise NotImplementedError(
                "multiprocess component benchmarking is not yet implemented"
            )
            # Component level timings for multiprocess benchmarking
            # are not generated using this code that re-runs individual
            # components.  Instead, those benchmarks are generated in
            # aggregate during setup and then extracted from logs later.
        else:
            state.run.by_name(component_name)
    except Exception as err:
        logger.exception("run_component exception: %s", component_name)
        raise
    else:
        logger.info("run_component completed: %s", component_name)
    return 0


def teardown_component(state, component_name):
    logger.info("teardown_component: %s", component_name)

    # use the pipeline module to clear out all the tables, so
    # the next benchmark run has a clean slate.
    # anything needed should be reloaded from the pipeline checkpoint file
    pipeline_tables = state.registered_tables()
    for table_name in pipeline_tables:
        logger.info("dropping table %s", table_name)
        state.drop_table(table_name)

    if state.settings.multiprocess:
        raise NotImplementedError("multiprocess benchmarking is not yet implemented")
    else:
        state.checkpoint.close_store()
    logger.critical(
        "teardown_component completed: %s\n\n%s\n\n", component_name, "~" * 88
    )
    return 0


def pre_run(
    state,
    model_working_dir,
    configs_dirs=None,
    data_dir="data",
    output_dir="output",
    settings_file_name=None,
):
    """
    Pre-run the models, checkpointing everything.

    By checkpointing everything, it is possible to run each benchmark
    by recreating the state of the pipeline immediately prior to that
    component.

    Parameters
    ----------
    model_working_dir : str
        Path to the model working directory, generally inside the
        benchmarking workspace.
    configs_dirs : Iterable[str], optional
        Override the config dirs, similar to using -c on the command line
        for a model run.
    data_dir : str, optional
        Override the data directory similar to using -d on the command line
        for a model run.
    output_dir : str, optional
        Override the output directory similar to using -o on the command line
        for a model run.
    settings_file_name : str, optional
        Override the settings file name, similar to using -s on the command line
        for a model run.
    """
    if configs_dirs is None:
        state.add_injectable("configs_dir", os.path.join(model_working_dir, "configs"))
    else:
        configs_dirs_ = [os.path.join(model_working_dir, i) for i in configs_dirs]
        state.add_injectable("configs_dir", configs_dirs_)
    state.add_injectable("data_dir", os.path.join(model_working_dir, data_dir))
    state.add_injectable("output_dir", os.path.join(model_working_dir, output_dir))

    if settings_file_name is not None:
        state.add_injectable("settings_file_name", settings_file_name)

    # Always pre_run from the beginning
    config.override_setting("resume_after", None)

    # register abm steps and other abm-specific injectables
    if "preload_injectables" not in state.context:
        from activitysim import abm  # noqa: F401

        # register abm steps and other abm-specific injectables

    if settings_file_name is not None:
        state.add_injectable("settings_file_name", settings_file_name)

    # cleanup
    # cleanup_output_files()

    state.logging.config_logger(basic=False)
    config.filter_warnings()
    logging.captureWarnings(capture=True)

    # directories
    for k in ["configs_dir", "settings_file_name", "data_dir", "output_dir"]:
        logger.info("SETTING %s: %s" % (k, state.get_injectable(k, None)))

    log_settings = state.get_injectable("log_settings", {})
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

    logger.info(f"MODELS: {config.setting('models')}")

    if state.settings.multiprocess:
        logger.info("run multi-process complete simulation")
    else:
        logger.info("run single process simulation")
        state.run(models=state.settings.models)
        state.checkpoint.close_store()

    tracing.print_elapsed_time("prerun required models for checkpointing", t0)

    return 0


def run_multiprocess(state: workflow.State):
    logger.info("run multiprocess simulation")
    state.tracing.delete_trace_files()
    state.tracing.delete_output_files("h5")
    state.tracing.delete_output_files("csv")
    state.tracing.delete_output_files("txt")
    state.tracing.delete_output_files("yaml")
    state.tracing.delete_output_files("prof")
    state.tracing.delete_output_files("omx")

    from activitysim.core import mp_tasks

    injectables = {k: state.get_injectable(k) for k in INJECTABLES}
    mp_tasks.run_multiprocess(state, injectables)

    # assert not pipeline.is_open()
    #
    # if state.settings.cleanup_pipeline_after_run:
    #     pipeline.cleanup_pipeline()


########


def local_dir():
    benchmarking_directory = workspace.get_dir()
    if benchmarking_directory is not None:
        return benchmarking_directory
    return os.getcwd()


def model_dir(*subdirs):
    return os.path.join(local_dir(), "models", *subdirs)


def template_setup_cache(
    example_name,
    component_names,
    benchmark_settings,
    benchmark_network_los,
    config_dirs=("configs",),
    data_dir="data",
    output_dir="output",
    settings_filename="settings.yaml",
    skip_component_names=None,
    config_overload_dir="dynamic_configs",
):
    """
    Prepare an example model for benchmarking.

    The algorithm for benchmarking components requires an existing pipeline
    file with checkpoints after every component, which allows the code to
    recreate the state of pipeline tables immediately prior to each component's
    execution.  The checkpoints file is very large, and can be recreated
    by running the model, so it is not stored/downloaded but just rebuilt as
    needed.

    This template function creates that pipeline store if it does not exist by
    getting the example model and running once through from the beginning through
    the last component to be benchmarked.  After this is done, a token is written
    out to `benchmark-setup-token.txt` in the output directory, flagging that
    the complete checkpointed pipeline has been created without errors and is
    ready to use.  If the template setup finds this token file, all this work
    is assumed to be already done and is skipped.

    Parameters
    ----------
    example_name : str
        The name of the example to benchmark, as used in
        the `activitysim create` command.
    component_names : Sequence
        The names of the model components to be individually
        benchmarked.  This list does not need to include all
        the components usually included in the example.
    benchmark_settings : Mapping
        Settings values to override from their usual values
        in the example.
    benchmark_network_los : Mapping
        Network LOS values to override from their usual values
        in the example.
    config_dirs : Sequence
    data_dir : str
    output_dir : str
    settings_filename : str
    skip_component_names : Sequence, optional
        Skip running these components when setting up the
        benchmarks (i.e. in pre-run).
    config_overload_dir : str, default 'dynamic_configs'
    """
    try:
        os.makedirs(model_dir(), exist_ok=True)
        get_example(
            example_name=example_name,
            destination=model_dir(),
            benchmarking=True,
        )
        os.makedirs(model_dir(example_name, config_overload_dir), exist_ok=True)

        # Find the settings file and extract the complete set of models included
        try:
            existing_settings, settings_filenames = state.filesystem.read_settings_file(
                settings_filename,
                mandatory=True,
                include_stack=True,
                configs_dir_list=[model_dir(example_name, c) for c in config_dirs],
            )
        except Exception:
            logger.error(f"os.getcwd:{os.getcwd()}")
            raise
        if "models" not in existing_settings:
            raise ValueError(
                f"missing list of models from {config_dirs}/{settings_filename}"
            )
        models = existing_settings["models"]
        use_multiprocess = existing_settings.get("multiprocess", False)
        for k in existing_settings:
            print(f"existing_settings {k}:", existing_settings[k])

        settings_changes = dict(
            benchmarking=True,
            checkpoints=True,
            trace_hh_id=None,
            chunk_training_mode="disabled",
            inherit_settings=True,
        )
        settings_changes.update(benchmark_settings)

        # Pre-run checkpointing or Multiprocess timing runs only need to
        # include models up to the penultimate component to be benchmarked.
        last_component_to_benchmark = 0
        for cname in component_names:
            try:
                last_component_to_benchmark = max(
                    models.index(cname), last_component_to_benchmark
                )
            except ValueError:
                if cname not in models:
                    logger.warning(
                        f"want to benchmark {example_name}.{cname} but it is not in the list of models to run"
                    )
                else:
                    raise
        if use_multiprocess:
            last_component_to_benchmark += 1
        pre_run_model_list = models[:last_component_to_benchmark]
        if skip_component_names is not None:
            for cname in skip_component_names:
                if cname in pre_run_model_list:
                    pre_run_model_list.remove(cname)
        settings_changes["models"] = pre_run_model_list

        if "multiprocess_steps" in existing_settings:
            multiprocess_steps = existing_settings["multiprocess_steps"]
            while (
                multiprocess_steps[-1].get("begin", "missing-begin")
                not in pre_run_model_list
            ):
                multiprocess_steps = multiprocess_steps[:-1]
                if len(multiprocess_steps) == 0:
                    break
            settings_changes["multiprocess_steps"] = multiprocess_steps

        with open(
            model_dir(example_name, config_overload_dir, settings_filename), "wt"
        ) as yf:
            try:
                yaml.safe_dump(settings_changes, yf)
            except Exception:
                logger.error(f"settings_changes:{str(settings_changes)}")
                logger.exception("oops")
                raise
        with open(
            model_dir(example_name, config_overload_dir, "network_los.yaml"), "wt"
        ) as yf:
            benchmark_network_los["inherit_settings"] = True
            yaml.safe_dump(benchmark_network_los, yf)

        os.makedirs(model_dir(example_name, output_dir), exist_ok=True)

        state = workflow.State.make_default(Path(model_dir(example_name)))

        # Running the model through all the steps and checkpointing everywhere is
        # expensive and only needs to be run once.  Once it is done we will write
        # out a completion token file to indicate to future benchmark attempts
        # that this does not need to be repeated.  Developers should manually
        # delete the token (or the whole model file) when a structural change
        # in the model happens such that re-checkpointing is needed (this should
        # happen rarely).
        use_config_dirs = (config_overload_dir, *config_dirs)
        token_file = model_dir(example_name, output_dir, "benchmark-setup-token.txt")
        if not os.path.exists(token_file) and not use_multiprocess:
            try:
                pre_run(
                    state,
                    model_dir(example_name),
                    use_config_dirs,
                    data_dir,
                    output_dir,
                    settings_filename,
                )
            except Exception as err:
                with open(
                    model_dir(example_name, output_dir, "benchmark-setup-error.txt"),
                    "wt",
                ) as f:
                    f.write(f"error {err}")
                    f.write(traceback.format_exc())
                raise
            else:
                with open(token_file, "wt") as f:
                    # We write the commit into the token, in case that is useful
                    # to developers to decide if the checkpointed pipeline is
                    # out of date.
                    asv_commit = os.environ.get("ASV_COMMIT", "ASV_COMMIT_UNKNOWN")
                    f.write(asv_commit)
        if use_multiprocess:
            # Multiprocessing timing runs are actually fully completed within
            # the setup_cache step, and component-level timings are written out
            # to log files by activitysim during this run.
            asv_commit = os.environ.get("ASV_COMMIT", "ASV_COMMIT_UNKNOWN")
            try:
                pre_run(
                    state,
                    model_dir(example_name),
                    use_config_dirs,
                    data_dir,
                    output_dir,
                    settings_filename,
                )
                run_multiprocess(state)
            except Exception as err:
                with open(
                    model_dir(
                        example_name, output_dir, f"-mp-run-error-{asv_commit}.txt"
                    ),
                    "wt",
                ) as f:
                    f.write(f"error {err}")
                    f.write(traceback.format_exc())
                raise

    except Exception as err:
        logger.error(
            f"error in template_setup_cache({example_name}):\n" + traceback.format_exc()
        )
        raise


def template_component_timings(
    module_globals,
    component_names,
    example_name,
    config_dirs,
    data_dir,
    output_dir,
    preload_injectables,
    repeat_=(1, 20, 10.0),  # min_repeat, max_repeat, max_time_seconds
    number_=1,
    timeout_=36000.0,  # ten hours,
    version_="1",
):
    """
    Inject ComponentTiming classes into a module namespace for benchmarking a model.

    Arguments with a trailing underscore get passed through to airspeed velocity, see
    https://asv.readthedocs.io/en/stable/benchmarks.html?highlight=repeat#timing-benchmarks
    for more info on these.

    Parameters
    ----------
    module_globals : Mapping
        The module globals namespace, into which the timing classes are written.
    component_names : Iterable[str]
        Names of components to benchmark.
    example_name : str
        Name of the example model being benchmarked, as it appears in the
        exammple_manifest.yaml file.
    config_dirs : Tuple[str]
        Config directories to use when running the model being benchmarked.
    data_dir, output_dir : str
        Data and output directories to use when running the model being
        benchmarked.
    preload_injectables : Tuple[str]
        Names of injectables to pre-load (typically skims).
    repeat_ : tuple
        The values for (min_repeat, max_repeat, max_time_seconds).  See ASV docs
        for more information.
    number_ : int, default 1
        The number of iterations in each sample.  Generally this should stay
        set to 1 for ActivitySim timing.
    timeout_ : number, default 36000.0,
        How many seconds before the benchmark is assumed to have crashed.  The
        typical default for airspeed velocity is 60 but that is wayyyyy too short for
        ActivitySim, so the default here is set to ten hours.
    version_ : str
        Used to determine when to invalidate old benchmark results. Benchmark results
        produced with a different value of the version than the current value will be
        ignored.
    """

    for componentname in component_names:

        class ComponentTiming:
            component_name = componentname
            warmup_time = 0
            min_run_count = 1
            processes = 1
            repeat = repeat_
            number = number_
            timeout = timeout_

            def setup(self):
                setup_component(
                    self.component_name,
                    model_dir(example_name),
                    preload_injectables,
                    config_dirs,
                    data_dir,
                    output_dir,
                    sharrow=True,
                )

            def teardown(self):
                teardown_component(self.component_name)

            def time_component(self):
                run_component(self.component_name)

            time_component.pretty_name = f"{example_name}:{componentname}"
            time_component.version = version_

        ComponentTiming.__name__ = f"{componentname}"

        module_globals[componentname] = ComponentTiming


def template_component_timings_mp(
    state: workflow.State,
    module_globals,
    component_names,
    example_name,
    output_dir,
    pretty_name,
    version_="1",
):
    """
    Inject ComponentTiming classes into a module namespace for benchmarking a model.

    This "MP" version for multiprocessing doesn't actually measure the time taken,
    but instead it parses the run logs from a single full run of the mode, to
    extract the per-component timings.  Most of the configurability has been removed
    compared to the single-process version of this function.

    Parameters
    ----------
    module_globals : Mapping
        The module globals namespace, into which the timing classes are written.
    component_names : Iterable[str]
        Names of components to benchmark.
    example_name : str
        Name of the example model being benchmarked, as it appears in the
        exammple_manifest.yaml file.
    output_dir : str
        Output directory to use when running the model being benchmarked.
    pretty_name : str
        A "pretty" name for this set of benchmarks.
    version_ : str
        Used to determine when to invalidate old benchmark results. Benchmark results
        produced with a different value of the version than the current value will be
        ignored.
    """

    for componentname in component_names:

        class ComponentTiming:
            component_name = componentname

            def track_component(self):
                durations = []
                state.add_injectable("output_dir", model_dir(example_name, output_dir))
                logfiler = state.get_log_file_path(f"timing_log.mp_households_*.csv")
                for logfile in glob.glob(logfiler):
                    df = pd.read_csv(logfile)
                    dfq = df.query(f"component_name=='{self.component_name}'")
                    if len(dfq):
                        durations.append(dfq.iloc[-1].duration)
                if len(durations):
                    return np.mean(durations)
                else:
                    raise ValueError("no results available")

            track_component.pretty_name = f"{pretty_name}:{componentname}"
            track_component.version = version_
            track_component.unit = "s"

        ComponentTiming.__name__ = f"{componentname}"

        module_globals[componentname] = ComponentTiming
