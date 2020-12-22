# ActivitySim
# See full license in LICENSE.txt.
import sys
import os
import logging
import argparse

from activitysim.core import inject
from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import pipeline
from activitysim.core import chunk

logger = logging.getLogger(__name__)


INJECTABLES = ['data_dir', 'configs_dir', 'output_dir', 'settings_file_name']


def add_run_args(parser, multiprocess=True):
    """Run command args
    """
    parser.add_argument('-w', '--working_dir',
                        type=str,
                        metavar='PATH',
                        help='path to example/project directory (default: %s)' % os.getcwd())
    parser.add_argument('-c', '--config',
                        type=str,
                        action='append',
                        metavar='PATH',
                        help='path to config dir')
    parser.add_argument('-o', '--output',
                        type=str,
                        metavar='PATH',
                        help='path to output dir')
    parser.add_argument('-d', '--data',
                        type=str,
                        action='append',
                        metavar='PATH',
                        help='path to data dir')
    parser.add_argument('-r', '--resume',
                        type=str,
                        metavar='STEPNAME',
                        help='resume after step')
    parser.add_argument('-p', '--pipeline',
                        type=str,
                        metavar='FILE',
                        help='pipeline file name')
    parser.add_argument('-s', '--settings_file',
                        type=str,
                        metavar='FILE',
                        help='settings file name')

    if multiprocess:
        parser.add_argument('-m', '--multiprocess',
                            default=False,
                            action='store_true',
                            help='run multiprocess. Adds configs_mp settings'
                            ' by default.')


def validate_injectable(name):
    try:
        dir_paths = inject.get_injectable(name)
    except RuntimeError:
        # injectable is missing, meaning is hasn't been explicitly set
        # and defaults cannot be found.
        sys.exit('Error: please specify either a --working_dir '
                 "containing 'configs', 'data', and 'output' folders "
                 'or all three of --config, --data, and --output')

    dir_paths = [dir_paths] if isinstance(dir_paths, str) else dir_paths

    for dir_path in dir_paths:
        if not os.path.exists(dir_path):
            sys.exit("Could not find %s '%s'" % (name, os.path.abspath(dir_path)))

    logger.info('using %s: %s' % (name, dir_paths))

    return dir_paths


def handle_standard_args(args, multiprocess=True):

    def inject_arg(name, value):
        assert name in INJECTABLES
        inject.add_injectable(name, value)

    if args.working_dir:
        # activitysim will look in the current working directory for
        # 'configs', 'data', and 'output' folders by default
        os.chdir(args.working_dir)

    if args.settings_file:
        inject_arg('settings_file_name', args.settings_file)

    if args.config:
        inject_arg('configs_dir', args.config)

    if args.data:
        inject_arg('data_dir', args.data)

    if args.output:
        inject_arg('output_dir', args.output)

    if multiprocess and args.multiprocess:
        config_paths = validate_injectable('configs_dir')

        if not os.path.exists('configs_mp'):
            logger.warning("could not find 'configs_mp'. skipping...")
        else:
            logger.info("adding 'configs_mp' to config_dir list...")
            config_paths.insert(0, 'configs_mp')
            inject_arg('configs_dir', config_paths)

        config.override_setting('multiprocess', args.multiprocess)

    for injectable in ['configs_dir', 'data_dir', 'output_dir']:
        validate_injectable(injectable)

    if args.pipeline:
        inject.add_injectable('pipeline_file_name', args.pipeline)

    if args.resume:
        config.override_setting('resume_after', args.resume)


def cleanup_output_files():

    active_log_files = \
        [h.baseFilename for h in logger.root.handlers if isinstance(h, logging.FileHandler)]
    tracing.delete_output_files('log', ignore=active_log_files)

    tracing.delete_output_files('h5')
    tracing.delete_output_files('csv')
    tracing.delete_output_files('txt')
    tracing.delete_output_files('yaml')
    tracing.delete_output_files('prof')
    tracing.delete_output_files('omx')


def log_settings():

    settings = [
        'households_sample_size',
        'chunk_size',
        'multiprocess',
        'num_processes',
        'resume_after',
    ]

    for k in settings:
        logger.info('setting %s: %s' % (k, config.setting(k)))


def run(args):
    """
    Run the models. Specify a project folder using the '--working_dir' option,
    or point to the config, data, and output folders directly with
    '--config', '--data', and '--output'. Both '--config' and '--data' can be
    specified multiple times. Directories listed first take precedence.

    """

    from activitysim import abm  # register injectables

    tracing.config_logger(basic=True)
    handle_standard_args(args)  # possibly update injectables
    tracing.config_logger(basic=False)  # update using possibly new logging configs
    config.filter_warnings()
    logging.captureWarnings(capture=True)

    log_settings()

    t0 = tracing.print_elapsed_time()

    # If you provide a resume_after argument to pipeline.run
    # the pipeline manager will attempt to load checkpointed tables from the checkpoint store
    # and resume pipeline processing on the next submodel step after the specified checkpoint
    resume_after = config.setting('resume_after', None)

    # cleanup if not resuming
    if not resume_after:
        cleanup_output_files()
    elif config.setting('cleanup_trace_files_on_resume', False):
        tracing.delete_trace_files()

    if config.setting('multiprocess', False):
        logger.info('run multiprocess simulation')

        from activitysim.core import mp_tasks
        run_list = mp_tasks.get_run_list()
        injectables = {k: inject.get_injectable(k) for k in INJECTABLES}
        mp_tasks.run_multiprocess(run_list, injectables)
    else:
        logger.info('run single process simulation')

        pipeline.run(models=config.setting('models'), resume_after=resume_after)
        pipeline.close_pipeline()
        chunk.log_write_hwm()

    tracing.print_elapsed_time('all models', t0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_run_args(parser)
    args = parser.parse_args()
    sys.exit(run(args))
