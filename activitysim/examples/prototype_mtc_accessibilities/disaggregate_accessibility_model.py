import argparse
import os
import pkg_resources

from activitysim.abm.models.disaggregate_accessibility import run_disaggregate_accessibility
from activitysim.cli.run import add_run_args

def disaggregate_accessibility():
    def model_path(filename):
        resource = os.path.join('abm', 'models', filename)
        return pkg_resources.resource_filename('activitysim', resource)

    def base_path(dirname):
        resource = os.path.join('examples', 'prototype_mtc', dirname)
        return pkg_resources.resource_filename('activitysim', resource)

    def extended_path(dirname):
        resource = os.path.join('examples', 'prototype_mtc_accessibilities', dirname)
        return pkg_resources.resource_filename('activitysim', resource)

    parser = argparse.ArgumentParser()
    add_run_args(parser)
    args = parser.parse_args()

    # add in the arguments
    args.config = [extended_path('configs'), base_path('configs')]
    args.output = extended_path('output')
    args.data = base_path('data')

    run_disaggregate_accessibility(args)

if __name__ == "__main__":
    disaggregate_accessibility()
