import os
import logging
import logging.handlers
import numpy as np
from ..core.pipeline import print_elapsed_time, open_pipeline, mem, run_model
from ..core import inject, tracing
from ..cli.run import handle_standard_args, config, warnings, cleanup_output_files, pipeline, INJECTABLES, chunk, add_run_args

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
        raise NotImplementedError("multiprocess benchmarking is not yet implemented")
    else:
        open_pipeline(resume_after, mode='r')

    for k in preload_injectables:
        if inject.get_injectable(k, None) is not None:
            logger.info("pre-loaded %s", k)

    logger.info("setup_component completed: %s", component_name)


def run_component(component_name):
    logger.info("run_component: %s", component_name)
    try:
        if config.setting('multiprocess', False):
            raise NotImplementedError("multiprocess benchmarking is not yet implemented")
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
        raise NotImplementedError("multiprocess benchmarking is not yet implemented")
    else:
        logger.info('run single process simulation')
        pipeline.run(models=config.setting('models'))
        pipeline.close_pipeline()

    tracing.print_elapsed_time('prerun required models for checkpointing', t0)

    return 0
