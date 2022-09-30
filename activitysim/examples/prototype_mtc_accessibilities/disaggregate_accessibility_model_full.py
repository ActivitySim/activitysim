import argparse
import os
import pkg_resources
import sys

from activitysim.cli.run import add_run_args, run


def disaggregate_accessibility():
    def base_path(dirname):
        resource = os.path.join('examples', 'test_prototype_mtc_full', dirname)
        return pkg_resources.resource_filename('activitysim', resource)

    def extended_path(dirname):
        resource = os.path.join('examples', 'prototype_mtc_accessibilities', dirname)
        return pkg_resources.resource_filename('activitysim', resource)

    parser = argparse.ArgumentParser()
    add_run_args(parser)
    args = parser.parse_args()

    # add in the arguments
    args.config = [extended_path('configs'), base_path('configs')]
    args.config.insert(0, extended_path('configs_mp'))  # separate line to comment out
    args.output = extended_path('output')
    args.data = [extended_path('data'), base_path('data')]

    sys.exit(run(args))


if __name__ == "__main__":
    disaggregate_accessibility()
