
import re
import sys

def benchmark(args):
    """
    Compute the airspeed velocity of an unladen activitysim.
    """
    # for now we simply complete a handoff to the asv tool.
    # TODO: setup workspace if not defined.
    from asv.main import main
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    del sys.argv[1]
    sys.exit(main())

def add_benchmark_args(parser):
    from asv.commands import common_args
    parser.add_argument(
        'range', nargs='?', default=None,
        help="""Range of commits to benchmark.  For a git
        repository, this is passed as the first argument to ``git
        log``.  See 'specifying ranges' section of the
        `gitrevisions` manpage for more info.  Also accepts the
        special values 'NEW', 'ALL', 'EXISTING', and 'HASHFILE:xxx'.
        'NEW' will benchmark all commits since the latest
        benchmarked on this machine.  'ALL' will benchmark all
        commits in the project. 'EXISTING' will benchmark against
        all commits for which there are existing benchmarks on any
        machine. 'HASHFILE:xxx' will benchmark only a specific set
        of hashes given in the file named 'xxx', which must have
        one hash per line. By default, will benchmark the head of
        each configured of the branches.""")
    parser.add_argument(
        "--steps", "-s", type=common_args.positive_int, default=None,
        help="""Maximum number of steps to benchmark.  This is
        used to subsample the commits determined by range to a
        reasonable number.""")
    common_args.add_bench(parser)
    parser.add_argument(
        "--profile", "-p", action="store_true",
        help="""In addition to timing, run the benchmarks through
        the `cProfile` profiler and store the results.""")
    common_args.add_parallel(parser)
    common_args.add_show_stderr(parser)
    parser.add_argument(
        "--quick", "-q", action="store_true",
        help="""Do a "quick" run, where each benchmark function is
        run only once.  This is useful to find basic errors in the
        benchmark functions faster.  The results are unlikely to
        be useful, and thus are not saved.""")
    common_args.add_environment(parser)
    parser.add_argument(
        "--set-commit-hash", default=None,
        help="""Set the commit hash to use when recording benchmark
        results. This makes results to be saved also when using an
        existing environment.""")
    common_args.add_launch_method(parser)
    parser.add_argument(
        "--dry-run", "-n", action="store_true",
        default=None,
        help="""Do not save any results to disk.""")
    common_args.add_machine(parser)
    parser.add_argument(
        "--skip-existing-successful", action="store_true",
        help="""Skip running benchmarks that have previous successful
        results""")
    parser.add_argument(
        "--skip-existing-failed", action="store_true",
        help="""Skip running benchmarks that have previous failed
        results""")
    parser.add_argument(
        "--skip-existing-commits", action="store_true",
        help="""Skip running benchmarks for commits that have existing
        results""")
    parser.add_argument(
        "--skip-existing", "-k", action="store_true",
        help="""Skip running benchmarks that have previous successful
        or failed results""")
    common_args.add_record_samples(parser)
    parser.add_argument(
        "--interleave-processes", action="store_true", default=False,
        help="""Interleave benchmarks with multiple processes across
        commits. This can avoid measurement biases from commit ordering,
        can take longer.""")
    parser.add_argument(
        "--no-interleave-processes", action="store_false", dest="interleave_processes")
    parser.add_argument(
        "--no-pull", action="store_true",
        help="Do not pull the repository")
    return parser

