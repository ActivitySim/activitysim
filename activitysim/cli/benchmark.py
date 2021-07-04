import os
import sys


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
        del commands[command]

    for name, command in sorted(commands.items()):
        if str(command) in hide_commands:
            continue
        subparser = command.setup_arguments(subparsers)
        common_args.add_global_arguments(subparser)

    parser.set_defaults(afunc=benchmark)
    return parser, subparsers



def benchmark(args):
    from asv.console import log
    from asv import util

    print("benchmarking activitysim...")

    log.enable(args.verbose)

    args.config = os.path.abspath(args.config)

    # Use the path to the config file as the cwd for the remainder of
    # the run
    dirname = os.path.dirname(args.config)
    os.chdir(dirname)

    print("working directory:", dirname)

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
