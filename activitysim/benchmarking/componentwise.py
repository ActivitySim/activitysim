import os
import logging
import numpy as np
from ..core.pipeline import print_elapsed_time, open_pipeline, mem, run_model
from ..core import inject, tracing
from ..cli.run import handle_standard_args, config, warnings, cleanup_output_files, pipeline, INJECTABLES, chunk, add_run_args

logger = logging.getLogger(__name__)


def reload_settings(**kwargs):
    settings = config.read_settings_file('settings.yaml', mandatory=True)
    for k in kwargs:
        settings[k] = kwargs[k]
    inject.add_injectable("settings", settings)
    return settings


def setup_component(component_name):
    """
    Prepare to benchmark a model component.

    This function sets up everything, opens the pipeline, and
    reloads table state from checkpoints of prior components.
    All this happens here, before the model component itself
    is actually executed inside the timed portion of the loop.
    """
    reload_settings(
        benchmarking=component_name,
    )

    # register abm steps and other abm-specific injectables outside of
    # benchmark timing loop
    if not inject.is_injectable('preload_injectables'):
        from activitysim import abm

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
        open_pipeline(resume_after)


def run_component(component_name):
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
    return 0


def teardown_component():
    if config.setting('multiprocess', False):
        raise NotImplementedError("multiprocess benchmarking is not yet implemented")
    else:
        pipeline.close_pipeline()
    return 0


def pre_run(model_working_dir):
    """
    Pre-run the models, checkpointing everything.
    """
    inject.add_injectable('configs_dir', os.path.join(model_working_dir, 'configs'))
    inject.add_injectable('data_dir', os.path.join(model_working_dir, 'data'))
    inject.add_injectable('output_dir', os.path.join(model_working_dir, 'output'))

    # register abm steps and other abm-specific injectables
    if not inject.is_injectable('preload_injectables'):
        from activitysim import abm  # register abm steps and other abm-specific injectables

    # Always pre_run from the beginning
    config.override_setting('resume_after', None)

    # cleanup
    cleanup_output_files()

    tracing.config_logger(basic=True)
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

    if config.setting('multiprocess', False):
        raise NotImplementedError("multiprocess benchmarking is not yet implemented")
    else:
        logger.info('run single process simulation')
        pipeline.run(models=config.setting('models'))
        pipeline.close_pipeline()

    tracing.print_elapsed_time('prerun required models for checkpointing', t0)

    return 0
