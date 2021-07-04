import os
import sys
import json

ASV_CONFIG = {
    # The version of the config file format.  Do not change, unless
    # you know what you are doing.
    "version": 1,

    # The name of the project being benchmarked
    "project": "activitysim",

    #The project's homepage
    "project_url": "https://activitysim.github.io/",

    # The URL or local path of the source code repository for the
    # project being benchmarked
    "repo": ".",

    # The tool to use to create environments.
    "environment_type": "conda",

    # the base URL to show a commit for the project.
    "show_commit_url": "http://github.com/ActivitySim/activitysim/commit/",

    # The Pythons you'd like to test against.  If not provided, defaults
    # to the current version of Python used to run `asv`.
    # "pythons": ["2.7", "3.6"],

    # The list of conda channel names to be searched for benchmark
    # dependency packages in the specified order
    "conda_channels": ["conda-forge"],

    # The matrix of dependencies to test.  Each key is the name of a
    # package (in PyPI) and the values are version numbers.  An empty
    # list or empty string indicates to just test against the default
    # (latest) version. null indicates that the package is to not be
    # installed. If the package to be tested is only available from
    # PyPi, and the 'environment_type' is conda, then you can preface
    # the package name by 'pip+', and the package will be installed via
    # pip (with all the conda available packages installed first,
    # followed by the pip installed packages).
    "matrix": {
        "pyarrow": [],
        "numpy": [],
        "openmatrix": [],
        "pandas": ["1.2"],
        "pyyaml": [],
        "pytables": [],
        "toolz": [],
        "orca": [],
        "psutil": [],
        "requests": [],
        "numba": [],
        "coverage": [],
        "pytest": [],
        "ruamel.yaml": [],
        "cytoolz": []
    },

    # The directory (relative to the current directory) to cache the Python
    # environments in.  If not provided, defaults to "env"
    "env_dir": "../activitysim-asv/env",

    # The directory (relative to the current directory) that raw benchmark
    # results are stored in.  If not provided, defaults to "results".
    "results_dir": "../activitysim-asv/results",

    # The directory (relative to the current directory) that the html tree
    # should be written to.  If not provided, defaults to "html".
    "html_dir": "../activitysim-asv/html",
}

def make_asv_argparser(parser):
    """
    The entry point for asv.

    Most of this work is handed off to the airspeed velocity library.
    """
    from asv.commands import common_args, Command, util, command_order

    def help(args):
        parser.print_help()
        sys.exit(0)

    common_args.add_global_arguments(parser, suppress_defaults=False)

    subparsers = parser.add_subparsers(
        title='benchmarking with airspeed velocity',
        description='valid subcommands')

    help_parser = subparsers.add_parser(
        "help", help="Display usage information")
    help_parser.set_defaults(afunc=help)

    commands = dict((x.__name__, x) for x in util.iter_subclasses(Command))

    hide_commands = ["quickstart", ]

    for command in command_order:
        if str(command) in hide_commands:
            continue
        subparser = commands[str(command)].setup_arguments(subparsers)
        common_args.add_global_arguments(subparser)
        subparser.add_argument(
            "--workspace", "-w",
            help="benchmarking workspace directory",
            default=".",
        )
        del commands[command]

    for name, command in sorted(commands.items()):
        if str(command) in hide_commands:
            continue
        subparser = command.setup_arguments(subparsers)
        subparser.add_argument(
            "--workspace", "-w",
            help="benchmarking workspace directory",
            default=".",
        )
        common_args.add_global_arguments(subparser)

    parser.set_defaults(afunc=benchmark)
    return parser, subparsers



def benchmark(args):
    from asv.console import log
    from asv import util

    log.enable(args.verbose)

    log.info("<== benchmarking activitysim ==>")

    # workspace
    args.workspace = os.path.abspath(args.workspace)

    if os.path.abspath(os.path.expanduser("~")) == args.workspace:
        log.error("don't run benchmarks in the user's home directory \n"
                  "try changing directories before calling `activitysim benchmark` "
                  "or use the --workspace option \n")
        sys.exit(1)

    if not os.path.isdir(args.workspace):
        raise NotADirectoryError(args.workspace)
    log.info(f" workspace: {args.workspace}")
    os.chdir(args.workspace)

    from ..benchmarking import workspace
    workspace.set_dir(args.workspace)

    from .. import __path__ as pkg_path
    log.info(f" activitysim installation: {pkg_path[0]}")

    repo_dir = os.path.normpath(
        os.path.join(pkg_path[0], "..")
    )
    git_dir = os.path.normpath(
        os.path.join(repo_dir, ".git")
    )
    local_git = os.path.exists(git_dir)
    log.info(f" local git repo available: {local_git}")

    asv_config = ASV_CONFIG.copy()
    if local_git:
        asv_config["repo"] = os.path.relpath(args.workspace, repo_dir)
    else:
        asv_config["repo"] = "https://github.com/ActivitySim/activitysim.git"

    conf_file = os.path.normpath(
        os.path.join(args.workspace, "asv.conf.json")
    )
    with open(conf_file, 'wt') as jf:
        json.dump(asv_config, jf)

    if args.config and args.config != "asv.conf.json":
        raise ValueError("activitysim manages the asv config json file itself, do not use --config")
    args.config = os.path.abspath(conf_file)


    try:
        result = args.func(args)
    except util.UserError as e:
        log.error(str(e))
        sys.exit(1)
    finally:
        log.flush()

    if result is None:
        result = 0

    sys.exit(result)
