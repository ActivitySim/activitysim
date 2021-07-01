import os
import logging
import numpy as np
from ..core.pipeline import print_elapsed_time, open_pipeline, mem, run_model
from ..core import inject, tracing
from ..cli.run import handle_standard_args, config, warnings, cleanup_output_files, pipeline, INJECTABLES, chunk, add_run_args

logger = logging.getLogger(__name__)


def benchmark_component(model, resume_after=None):
    """

    Parameters
    ----------
    model : str
        list of model_names
    resume_after : str or None
        model_name of checkpoint to load checkpoint and AFTER WHICH to resume model run

    returns:
        nothing, but with pipeline open
    """
    if config.setting('multiprocess', False):
        raise NotImplementedError("multiprocess benchmarking is not yet implemented")

    else: # single process benchmarking
        open_pipeline(resume_after)
        run_model(model)




def reload_settings(**kwargs):
    settings = config.read_settings_file('settings.yaml', mandatory=True)
    for k in kwargs:
        settings[k] = kwargs[k]
    inject.add_injectable("settings", settings)
    return settings


def prep_component(args, component_name):
    """
    Run the models. Specify a project folder using the '--working_dir' option,
    or point to the config, data, and output folders directly with
    '--config', '--data', and '--output'. Both '--config' and '--data' can be
    specified multiple times. Directories listed first take precedence.

    returns:
        int: sys.exit exit code
    """
    reload_settings(
        benchmarking=component_name,
    )

    # register abm steps and other abm-specific injectables
    # by default, assume we are running activitysim.abm
    # other callers (e.g. populationsim) will have to arrange to register their own steps and injectables
    # (presumably) in a custom run_simulation.py instead of using the 'activitysim run' command
    if not inject.is_injectable('preload_injectables'):
        from activitysim import abm  # register abm steps and other abm-specific injectables

    tracing.config_logger(basic=True)
    handle_standard_args(args)  # possibly update injectables

    # If you provide a resume_after argument to pipeline.run
    # the pipeline manager will attempt to load checkpointed tables from the checkpoint store
    # and resume pipeline processing on the next submodel step after the specified checkpoint
    models = config.setting('models')
    component_index = models.index(component_name)
    if component_index:
        resume_after = models[models.index(component_name) - 1]
    else:
        resume_after = None

    # cleanup if not resuming
    if not resume_after:
        cleanup_output_files()
    elif config.setting('cleanup_trace_files_on_resume', False):
        tracing.delete_trace_files()

    tracing.config_logger(basic=False)  # update using possibly new logging configs
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

def after_component():
    if config.setting('multiprocess', False):
        raise NotImplementedError("multiprocess benchmarking is not yet implemented")
    else:
        if config.setting('cleanup_pipeline_after_run', False):
            pipeline.cleanup_pipeline()  # has side effect of closing open pipeline
        else:
            pipeline.close_pipeline()
    return 0