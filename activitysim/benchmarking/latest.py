# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import shlex
import subprocess
import traceback

from asv import util
from asv.commands import common_args
from asv.commands.run import Run
from asv.console import log


def _do_build(args):
    env, conf, repo, commit_hash = args
    try:
        with log.set_level(logging.WARN):
            env.install_project(conf, repo, commit_hash)
    except util.ProcessError:
        return (env.name, False)
    return (env.name, True)


def _do_build_multiprocess(args):
    """
    multiprocessing callback to build the project in one particular
    environment.
    """
    try:
        return _do_build(args)
    except BaseException as exc:
        raise util.ParallelFailure(str(exc), exc.__class__, traceback.format_exc())


class Latest(Run):
    @classmethod
    def setup_arguments(cls, subparsers):
        parser = subparsers.add_parser(
            "latest",
            help="Run a benchmark suite on the HEAD commit",
            description="Run a benchmark suite.",
        )

        common_args.add_bench(parser)
        parser.add_argument(
            "--profile",
            "-p",
            action="store_true",
            help="""In addition to timing, run the benchmarks through
            the `cProfile` profiler and store the results.""",
        )
        common_args.add_parallel(parser)
        common_args.add_show_stderr(parser)
        parser.add_argument(
            "--quick",
            "-q",
            action="store_true",
            help="""Do a "quick" run, where each benchmark function is
            run only once.  This is useful to find basic errors in the
            benchmark functions faster.  The results are unlikely to
            be useful, and thus are not saved.""",
        )
        common_args.add_environment(parser)
        parser.add_argument(
            "--set-commit-hash",
            default=None,
            help="""Set the commit hash to use when recording benchmark
            results. This makes results to be saved also when using an
            existing environment.""",
        )
        common_args.add_launch_method(parser)
        parser.add_argument(
            "--dry-run",
            "-n",
            action="store_true",
            default=None,
            help="""Do not save any results to disk.""",
        )
        common_args.add_machine(parser)
        parser.add_argument(
            "--skip-existing-successful",
            action="store_true",
            help="""Skip running benchmarks that have previous successful
            results""",
        )
        parser.add_argument(
            "--skip-existing-failed",
            action="store_true",
            help="""Skip running benchmarks that have previous failed
            results""",
        )
        parser.add_argument(
            "--skip-existing-commits",
            action="store_true",
            help="""Skip running benchmarks for commits that have existing
            results""",
        )
        parser.add_argument(
            "--skip-existing",
            "-k",
            action="store_true",
            help="""Skip running benchmarks that have previous successful
            or failed results""",
        )
        parser.add_argument(
            "--interleave-processes",
            action="store_true",
            default=False,
            help="""Interleave benchmarks with multiple processes across
            commits. This can avoid measurement biases from commit ordering,
            can take longer.""",
        )
        parser.add_argument(
            "--no-interleave-processes",
            action="store_false",
            dest="interleave_processes",
        )
        parser.add_argument(
            "--no-pull", action="store_true", help="Do not pull the repository"
        )

        parser.set_defaults(func=cls.run_from_args)

        return parser

    @classmethod
    def run_from_conf_args(cls, conf, args, **kwargs):
        return cls.run(
            conf=conf,
            range_spec="HEAD^!",
            steps=None,
            bench=args.bench,
            attribute=args.attribute,
            parallel=args.parallel,
            show_stderr=args.show_stderr,
            quick=args.quick,
            profile=args.profile,
            env_spec=args.env_spec,
            set_commit_hash=args.set_commit_hash,
            dry_run=args.dry_run,
            machine=args.machine,
            skip_successful=args.skip_existing_successful or args.skip_existing,
            skip_failed=args.skip_existing_failed or args.skip_existing,
            skip_existing_commits=args.skip_existing_commits,
            record_samples=True,
            append_samples=True,
            pull=not args.no_pull,
            interleave_processes=args.interleave_processes,
            launch_method=args.launch_method,
            **kwargs
        )


class Batch(Run):
    @classmethod
    def setup_arguments(cls, subparsers):
        parser = subparsers.add_parser(
            "batch",
            help="Run a set of benchmark suites based on a batch file. "
            "Simply give the file name, which should be a text file "
            "containing a number of activitysim benchmark commands.",
            description="Run a set of benchmark suites based on a batch file.",
        )

        parser.add_argument(
            "file",
            action="store",
            type=str,
            help="""Set the file name to use for reading multiple commands.""",
        )

        parser.set_defaults(func=cls.run_from_args)

        return parser

    @classmethod
    def run_from_conf_args(cls, conf, args, **kwargs):
        with open(args.file, "rt") as f:
            for line in f.readlines():
                subprocess.run(["activitysim", "benchmark", *shlex.split(line)])
