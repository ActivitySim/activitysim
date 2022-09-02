import sys
import os
import logging
import numpy as np
import argparse

from activitysim.core import (inject,
                              tracing,
                              config,
                              pipeline,
                              mem,
                              chunk)

from activitysim.cli.run import (handle_standard_args,
                                 cleanup_output_files,
                                 add_run_args)

logger = logging.getLogger(__name__)

# Modified 'run' function from activitysim.cli.run to override the models list in settings.yaml with MODELS list above
# and run only the disaggregate_accessibility step but retain all other main model settings.

# This must be in a separate file or else it will not properly call the injection decorators

# This also enables the model to be run as either a model step, or a one-off model.
# example model run is in examples/example_mtc_accessibility/disaggregate_accessibility_model.py
def run_disaggregate_accessibility(args):
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
    if not inject.is_injectable('preload_injectables'):
        from activitysim import abm  # register abm steps and other abm-specific injectables

    tracing.config_logger(basic=True)

    # possibly update injectables
    handle_standard_args(args)

    # cleanup
    cleanup_output_files()

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

    t0 = tracing.print_elapsed_time()

    INJECTABLES = ['data_dir', 'configs_dir', 'output_dir', 'settings_file_name']
    MODELS = ['initialize_landuse', 'compute_disaggregate_accessibility', 'write_tables']

    try:
        # if config.setting('multiprocess', False):
        #     logger.info('run multiprocess simulation')
        #
        #     from activitysim.core import mp_tasks
        #     injectables = {k: inject.get_injectable(k) for k in INJECTABLES}
        #     mp_tasks.run_multiprocess(injectables)
        #
        #     assert not pipeline.is_open()
        #
        #     if config.setting('cleanup_pipeline_after_run', False):
        #         pipeline.cleanup_pipeline()
        #
        # else:
            logger.info('run single process simulation')

            pipeline.run(models=MODELS, resume_after=None)

            if config.setting('cleanup_pipeline_after_run', False):
                pipeline.cleanup_pipeline()  # has side effect of closing open pipeline
            else:
                pipeline.close_pipeline()

            mem.log_global_hwm()  # main process
    except Exception:
        # log time until error and the error traceback
        tracing.print_elapsed_time('all models until this error', t0)
        logger.exception('activitysim run encountered an unrecoverable error')
        raise

    chunk.consolidate_logs()
    mem.consolidate_logs()

    tracing.print_elapsed_time('all models', t0)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_run_args(parser)
    args = parser.parse_args()

    # Create subprocess output folder
    sub_output_path = os.path.join(args.output, 'disaggregate_accessibilities')
    args.output = sub_output_path
    if not os.path.exists(sub_output_path):
        os.mkdir(sub_output_path)
    sys.exit(run_disaggregate_accessibility(args))