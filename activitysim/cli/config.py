# ActivitySim
# See full license in LICENSE.txt.
import sys
import os
import logging
import argparse

from activitysim.core import inject
from activitysim.core import tracing
from activitysim.core import config

from activitysim.cli import run

logger = logging.getLogger(__name__)


def handle_asim_args(args):

    if args.multiprocess:
        config.override_setting('multiprocess', args.multiprocess)

    if args.config:
        inject.add_injectable('configs_dir', args.config)

    if args.data:
        inject.add_injectable('data_dir', args.data)

    if args.output:
        inject.add_injectable('output_dir', args.output)

    for injectable in ['configs_dir', 'data_dir', 'output_dir']:
        run.validate_injectable(injectable)

    if args.resume:
        config.override_setting('resume_after', args.resume)

    if args.households:
        config.override_setting('households_sample_size', args.households)

    if args.trace_hh_id:
        config.override_setting('trace_hh_id', args.trace_hh_id or None)


def add_asim_args(parser):
    """Run command args
    """
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
    parser.add_argument('-m', '--multiprocess',
                        default=False,
                        action='store_true',
                        help='run multiprocess.')

    parser.add_argument('-hh', '--households',
                        type=int,
                        help='set households_sample_size.')
    parser.add_argument('-t', '--trace_hh_id',
                        type=int,
                        help='set trace_hh_id.')


def setup():

    parser = argparse.ArgumentParser()
    add_asim_args(parser)
    args = parser.parse_args()

    tracing.config_logger(basic=True)
    # run.handle_standard_args(args)  # possibly update injectables
    handle_asim_args(args)  # possibly update injectables
    tracing.config_logger(basic=False)  # update using possibly new logging configs
    config.filter_warnings()
    logging.captureWarnings(capture=True)
