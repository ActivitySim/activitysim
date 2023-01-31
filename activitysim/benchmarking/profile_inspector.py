import base64
import contextlib
import io
import json
import os
import tempfile
import traceback
import zlib

from asv.commands import Command
from asv.console import log
from asv.plugins.snakeviz import SnakevizGui


def benchmark_snakeviz(json_record, benchmark=None):
    """
    A utility to directly display saved profiling data in Snakeviz.

    Parameters
    ----------
    json_record : Path-like
        The archived json file that contains profile data for benchmarks.
    benchmark : str, optional
        The name of the benchmark to display.
    """
    from asv import util

    with open(json_record, "rt") as f:
        json_content = json.load(f)
    profiles = json_content.get("profiles", {})
    if benchmark is None or benchmark not in profiles:
        if profiles:
            log.info("\n\nAvailable profiles:")
            for k in profiles.keys():
                log.info(f"- {k}")
        else:
            log.info(f"\n\nNo profiles stored in {json_record}")
        if benchmark is None:
            return
        raise KeyError()
    profile_data = zlib.decompress(base64.b64decode(profiles[benchmark].encode()))
    prefix = benchmark.replace(".", "__") + "."
    with temp_profile(profile_data, prefix) as profile_path:
        log.info(f"Profiling data cached to {profile_path}")
        import pstats

        prof = pstats.Stats(profile_path)
        prof.strip_dirs().dump_stats(profile_path + "b")
        try:
            SnakevizGui.open_profiler_gui(profile_path + "b")
        except KeyboardInterrupt:
            pass
        except Exception:

            traceback.print_exc()
            input(input("Press Enter to continue..."))
        finally:
            os.remove(profile_path + "b")


@contextlib.contextmanager
def temp_profile(profile_data, prefix=None):
    profile_fd, profile_path = tempfile.mkstemp(prefix=prefix)
    try:
        with io.open(profile_fd, "wb", closefd=True) as fd:
            fd.write(profile_data)

        yield profile_path
    finally:
        os.remove(profile_path)


class ProfileInspector(Command):
    @classmethod
    def setup_arguments(cls, subparsers):
        parser = subparsers.add_parser(
            "snakeviz",
            help="""Run snakeviz on a particular benchmark that has been profiled.""",
            description="Inspect a benchmark profile",
        )

        parser.add_argument(
            "json_record",
            help="""The json file in the benchmark results to read profile data from.""",
        )

        parser.add_argument(
            "benchmark",
            help="""The benchmark to profile.  Must be a
            fully-specified benchmark name. For parameterized benchmark, it
            must include the parameter combination to use, e.g.:
            benchmark_name(param0, param1, ...)""",
            default=None,
            nargs="?",
        )

        parser.set_defaults(func=cls.run_from_args)

        return parser

    @classmethod
    def run_from_conf_args(cls, conf, args, **kwargs):
        return cls.run(json_record=args.json_record, benchmark=args.benchmark)

    @classmethod
    def run(cls, json_record, benchmark):
        benchmark_snakeviz(json_record, benchmark)
