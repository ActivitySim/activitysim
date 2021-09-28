import glob
import os
import logging
import logging.handlers
import numpy as np
import pandas as pd
import yaml
import traceback
import multiprocessing

from ..cli.create import get_example
from ..core.pipeline import print_elapsed_time, open_pipeline, mem, run_model
from ..core import inject, tracing
from ..cli.run import handle_standard_args, config, warnings, cleanup_output_files, pipeline, INJECTABLES, chunk, add_run_args
from .config_editing import modify_yaml
from . import workspace

logger = logging.getLogger(__name__)


def reload_settings(settings_filename, **kwargs):
    settings = config.read_settings_file(settings_filename, mandatory=True)
    for k in kwargs:
        settings[k] = kwargs[k]
    inject.add_injectable("settings", settings)
    return settings


def component_logging(component_name):
    root_logger = logging.getLogger()

    CLOG_FMT = '%(asctime)s %(levelname)7s - %(name)s: %(message)s'

    logfilename = config.log_file_path(f"asv-{component_name}.log")

    # avoid creation of multiple file handlers for logging components
    # as we will re-enter this function for every component run
    for entry in root_logger.handlers:
        if (isinstance(entry, logging.handlers.RotatingFileHandler)) and \
                (entry.formatter._fmt == CLOG_FMT):
            return

    tracing.config_logger(basic=True)
    file_handler = logging.handlers.RotatingFileHandler(
        filename=logfilename,
        mode='a', maxBytes=50_000_000, backupCount=5,
    )
    formatter = logging.Formatter(
        fmt=CLOG_FMT,
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)


def setup_component(
        component_name,
        working_dir='.',
        preload_injectables=(),
        configs_dirs=('configs'),
        data_dir='data',
        output_dir='output',
        settings_filename='settings.yaml',
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
    inject.add_injectable('configs_dir', [os.path.join(working_dir, i) for i in configs_dirs])
    inject.add_injectable('data_dir', os.path.join(working_dir, data_dir))
    inject.add_injectable('output_dir', os.path.join(working_dir, output_dir))

    reload_settings(
        settings_filename,
        benchmarking=component_name,
        checkpoints=False,
    )

    component_logging(component_name)
    logger.info("connected to component logger")
    config.filter_warnings()
    logging.captureWarnings(capture=True)

    # register abm steps and other abm-specific injectables outside of
    # benchmark timing loop
    if not inject.is_injectable('preload_injectables'):
        logger.info("preload_injectables yes import")
        from activitysim import abm
    else:
        logger.info("preload_injectables no import")

    # Extract the resume_after argument based on the model immediately
    # prior to the component being benchmarked.
    models = config.setting('models')
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

    if config.setting('multiprocess', False):
        raise NotImplementedError("multiprocess component benchmarking is not yet implemented")
    else:
        open_pipeline(resume_after, mode='r')

    for k in preload_injectables:
        if inject.get_injectable(k, None) is not None:
            logger.info("pre-loaded %s", k)

    # Directories Logging
    for k in ['configs_dir', 'settings_file_name', 'data_dir', 'output_dir']:
        logger.info(f'DIRECTORY {k}: {inject.get_injectable(k, None)}')

    # Settings Logging
    log_settings = [
        'checkpoints',
        'chunk_training_mode',
        'chunk_size',
        'chunk_method',
        'trace_hh_id',
        'households_sample_size',
        'check_for_variability',
        'use_shadow_pricing',
        'want_dest_choice_sample_tables',
        'log_alt_losers',
    ]
    for k in log_settings:
        logger.info(f'SETTING {k}: {config.setting(k)}')

    logger.info("setup_component completed: %s", component_name)


def run_component(component_name):
    logger.info("run_component: %s", component_name)
    try:
        if config.setting('multiprocess', False):
            raise NotImplementedError("multiprocess component benchmarking is not yet implemented")
            # logger.info('run multiprocess simulation')
            #
            # from activitysim.core import mp_tasks
            # injectables = {k: inject.get_injectable(k) for k in INJECTABLES}
            # mp_tasks.run_multiprocess(injectables)
            #
            # assert not pipeline.is_open()
            #
            # if config.setting('cleanup_pipeline_after_run', False):
            #     pipeline.cleanup_pipeline()
        else:
            run_model(component_name)
    except Exception as err:
        logger.exception("run_component exception: %s", component_name)
        raise
    else:
        logger.info("run_component completed: %s", component_name)
    return 0


def teardown_component(component_name):
    logger.info("teardown_component: %s", component_name)

    # use the pipeline module to clear out all the orca tables, so
    # the next benchmark run has a clean slate.
    # anything needed should be reloaded from the pipeline checkpoint file
    pipeline_tables = pipeline.registered_tables()
    for table_name in pipeline_tables:
        logger.info("dropping table %s", table_name)
        pipeline.drop_table(table_name)

    if config.setting('multiprocess', False):
        raise NotImplementedError("multiprocess benchmarking is not yet implemented")
    else:
        pipeline.close_pipeline()
    logger.critical(
        "teardown_component completed: %s\n\n%s\n\n",
        component_name,
        "~" * 88
    )
    return 0


def pre_run(
        model_working_dir,
        configs_dirs=None,
        data_dir='data',
        output_dir='output',
        settings_file_name=None,
):
    """
    Pre-run the models, checkpointing everything.
    """
    if configs_dirs is None:
        inject.add_injectable('configs_dir', os.path.join(model_working_dir, 'configs'))
    else:
        configs_dirs_ = [os.path.join(model_working_dir, i) for i in configs_dirs]
        inject.add_injectable('configs_dir', configs_dirs_)
    inject.add_injectable('data_dir', os.path.join(model_working_dir, data_dir))
    inject.add_injectable('output_dir', os.path.join(model_working_dir, output_dir))

    if settings_file_name is not None:
        inject.add_injectable('settings_file_name', settings_file_name)

    # Always pre_run from the beginning
    config.override_setting('resume_after', None)

    # register abm steps and other abm-specific injectables
    if not inject.is_injectable('preload_injectables'):
        from activitysim import abm  # register abm steps and other abm-specific injectables

    if settings_file_name is not None:
        inject.add_injectable('settings_file_name', settings_file_name)

    # cleanup
    #cleanup_output_files()

    tracing.config_logger(basic=False)
    config.filter_warnings()
    logging.captureWarnings(capture=True)

    # directories
    for k in ['configs_dir', 'settings_file_name', 'data_dir', 'output_dir']:
        logger.info('SETTING %s: %s' % (k, inject.get_injectable(k, None)))

    log_settings = inject.get_injectable('log_settings', {})
    for k in log_settings:
        logger.info('SETTING %s: %s' % (k, config.setting(k)))

    # OMP_NUM_THREADS: openmp
    # OPENBLAS_NUM_THREADS: openblas
    # MKL_NUM_THREADS: mkl
    for env in ['MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
        logger.info(f"ENV {env}: {os.getenv(env)}")

    np_info_keys = [
        'atlas_blas_info',
        'atlas_blas_threads_info',
        'atlas_info',
        'atlas_threads_info',
        'blas_info',
        'blas_mkl_info',
        'blas_opt_info',
        'lapack_info',
        'lapack_mkl_info',
        'lapack_opt_info',
        'mkl_info']

    for cfg_key in np_info_keys:
        info = np.__config__.get_info(cfg_key)
        if info:
            for info_key in ['libraries']:
                if info_key in info:
                    logger.info(f"NUMPY {cfg_key} {info_key}: {info[info_key]}")

    t0 = tracing.print_elapsed_time()

    logger.info(f"MODELS: {config.setting('models')}")

    if config.setting('multiprocess', False):
        logger.info('run multi-process complete simulation')
    else:
        logger.info('run single process simulation')
        pipeline.run(models=config.setting('models'))
        pipeline.close_pipeline()

    tracing.print_elapsed_time('prerun required models for checkpointing', t0)

    return 0


def run_multiprocess():
    logger.info('run multiprocess simulation')

    from activitysim.core import mp_tasks
    injectables = {k: inject.get_injectable(k) for k in INJECTABLES}
    mp_tasks.run_multiprocess(injectables)

    assert not pipeline.is_open()

    if config.setting('cleanup_pipeline_after_run', False):
        pipeline.cleanup_pipeline()






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
        data_dir='data',
        output_dir='output',
        settings_filename="settings.yaml",
        skip_component_names=None,
        config_overload_dir='dynamic_configs',
):
    """

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
        from ..core.config import read_settings_file
        try:
            existing_settings, settings_filenames = read_settings_file(
                settings_filename,
                mandatory=True,
                include_stack=True,
                configs_dir_list=[model_dir(example_name, c) for c in config_dirs],
            )
        except:
            logger.error(f"os.getcwd:{os.getcwd()}")
            raise
        if 'models' not in existing_settings:
            raise ValueError(f"missing list of models from {config_dirs}/{settings_filename}")
        models = existing_settings['models']
        use_multiprocess = existing_settings.get('multiprocess', False)
        for k in existing_settings:
            print(f"existing_settings {k}:", existing_settings[k])

        settings_changes = dict(
            benchmarking=True,
            checkpoints=True,
            trace_hh_id=None,
            chunk_training_mode='disabled',
            inherit_settings=True,
        )
        settings_changes.update(benchmark_settings)

        # Pre-run checkpointing or Multiprocess timing runs only need to
        # include models up to the penultimate component to be benchmarked.
        last_component_to_benchmark = 0
        for cname in component_names:
            try:
                last_component_to_benchmark = max(
                    models.index(cname),
                    last_component_to_benchmark
                )
            except ValueError:
                if cname not in models:
                    logger.warning(f"want to benchmark {example_name}.{cname} but it is not in the list of models to run")
                else:
                    raise
        if use_multiprocess:
            last_component_to_benchmark += 1
        pre_run_model_list = models[:last_component_to_benchmark]
        if skip_component_names is not None:
            for cname in skip_component_names:
                if cname in pre_run_model_list:
                    pre_run_model_list.remove(cname)
        settings_changes['models'] = pre_run_model_list

        if 'multiprocess_steps' in existing_settings:
            multiprocess_steps = existing_settings['multiprocess_steps']
            while multiprocess_steps[-1].get('begin', "missing-begin") not in pre_run_model_list:
                multiprocess_steps = multiprocess_steps[:-1]
                if len(multiprocess_steps) == 0:
                    break
            settings_changes['multiprocess_steps'] = multiprocess_steps

        with open(model_dir(example_name, config_overload_dir, settings_filename), 'wt') as yf:
            try:
                yaml.safe_dump(settings_changes, yf)
            except:
                logger.error(f"settings_changes:{str(settings_changes)}")
                logger.exception("oops")
                raise
        with open(model_dir(example_name, config_overload_dir, "network_los.yaml"), 'wt') as yf:
            benchmark_network_los['inherit_settings'] = True
            yaml.safe_dump(benchmark_network_los, yf)

        os.makedirs(model_dir(example_name, output_dir), exist_ok=True)

        # Running the model through all the steps and checkpointing everywhere is
        # expensive and only needs to be run once.  Once it is done we will write
        # out a completion token file to indicate to future benchmark attempts
        # that this does not need to be repeated.  Developers should manually
        # delete the token (or the whole model file) when a structural change
        # in the model happens such that re-checkpointing is needed (this should
        # happen rarely).
        use_config_dirs = (config_overload_dir, *config_dirs)
        token_file = model_dir(example_name, output_dir, 'benchmark-setup-token.txt')
        if not os.path.exists(token_file) and not use_multiprocess:
            try:
                pre_run(model_dir(example_name), use_config_dirs, data_dir, output_dir, settings_filename)
            except Exception as err:
                with open(model_dir(example_name, output_dir, 'benchmark-setup-error.txt'), 'wt') as f:
                    f.write(f"error {err}")
                    f.write(traceback.format_exc())
                raise
            else:
                with open(token_file, 'wt') as f:
                    # We write the commit into the token, in case that is useful
                    # to developers to decide if the checkpointed pipeline is
                    # out of date.
                    asv_commit = os.environ.get('ASV_COMMIT', 'ASV_COMMIT_UNKNOWN')
                    f.write(asv_commit)
        if use_multiprocess:
            # Multiprocessing timing runs are actually fully completed within
            # the setup_cache step, and component-level timings are written out
            # to log files by activitysim during this run.
            pre_run(model_dir(example_name), use_config_dirs, data_dir, output_dir, settings_filename)
            run_multiprocess()

    except Exception as err:
        logger.error(
            f"error in template_setup_cache({example_name}):\n"+traceback.format_exc()
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
        repeat_=(1,20,10.0), # min_repeat, max_repeat, max_time_seconds
        number_=1,
        timeout_=36000.0,  # ten hours,
        version_='1',
):

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
                    self.component_name, model_dir(example_name), preload_injectables,
                    config_dirs, data_dir, output_dir,
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
        module_globals,
        component_names,
        example_name,
        output_dir,
        pretty_name,
        version_='1',
):

    for componentname in component_names:

        class ComponentTiming:
            component_name = componentname
            def track_component(self):
                durations = []
                for logfile in glob.glob(model_dir(example_name, output_dir, "timing_log.mp_households_*.csv")):
                    df = pd.read_csv(logfile)
                    durations.append(df.query(f"component_name=='{self.component_name}'").iloc[-1].duration)
                return np.mean(durations)
            track_component.pretty_name = f"{pretty_name}:{componentname}"
            track_component.version = version_
            track_component.unit = 's'

        ComponentTiming.__name__ = f"{componentname}"

        module_globals[componentname] = ComponentTiming

