import argparse


class CLI:
    def __init__(self, version, description):
        self.version = version
        self.description = description
        self.parser = argparse.ArgumentParser(description=self.description)

        self.parser.add_argument(
            "--version", "-V", action="version", version=self.version
        )

        # print help if no subcommand is provided
        self.parser.set_defaults(afunc=lambda x: self.parser.print_help())

        self.subparsers = self.parser.add_subparsers(
            title="subcommands", help="available subcommand options"
        )

    def add_subcommand(self, name, args_func, exec_func, description):
        subparser = self.subparsers.add_parser(name, description=description)
        args_func(subparser)
        subparser.set_defaults(afunc=exec_func)

    def execute(self):
        args = self.parser.parse_args()
        args.afunc(args)
