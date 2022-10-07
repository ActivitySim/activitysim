import argparse
import os
import pkg_resources
import sys

from activitysim.cli.run import add_run_args, run


def disaggregate_accessibility():
    # Constructs full paths
    def base_path(dirname):
        resource = os.path.join('examples', 'placeholder_sandag_2_zone', dirname)
        return pkg_resources.resource_filename('activitysim', resource)

    def extended_path(dirname):
        resource = os.path.join('examples', 'placeholder_sandag_2_zone_extended', dirname)
        return pkg_resources.resource_filename('activitysim', resource)

    parser = argparse.ArgumentParser()
    add_run_args(parser)
    args = parser.parse_args()

    # add in the arguments
    args.config = [extended_path('configs'), base_path('configs_2_zone'), base_path('placeholder_psrc/configs')]
    args.config.insert(0, extended_path('configs_mp'))  # separate line to comment out
    args.output = extended_path('output')
    args.data = [extended_path('data'), base_path('data_2')]

    sys.exit(run(args))

if __name__ == "__main__":
    disaggregate_accessibility()
