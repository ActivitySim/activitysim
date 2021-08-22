# ActivitySim
# See full license in LICENSE.txt.
import os
import subprocess
import pkg_resources

import pandas as pd
import pandas.testing as pdt

from activitysim.core import inject


def teardown_function(func):
    inject.clear_cache()
    inject.reinject_decorated_tables()


def example_path(dirname):
    resource = os.path.join('examples', 'example_sandag', dirname)
    return pkg_resources.resource_filename('activitysim', resource)


def mtc_example_path(dirname):
    resource = os.path.join('examples', 'example_mtc', dirname)
    return pkg_resources.resource_filename('activitysim', resource)


def psrc_example_path(dirname):
    resource = os.path.join('examples', 'example_psrc', dirname)
    return pkg_resources.resource_filename('activitysim', resource)


def test_sandag():

    def test_path(dirname):
        return os.path.join(os.path.dirname(__file__), dirname)

    def regress(zone):

        # ## regress tours
        regress_tours_df = pd.read_csv(test_path(f'regress/final_{zone}_zone_tours.csv'))
        tours_df = pd.read_csv(test_path(f'output/final_{zone}_zone_tours.csv'))
        print(f"regress tours")
        pdt.assert_frame_equal(tours_df, regress_tours_df), "regress tours"

        # ## regress trips
        regress_trips_df = pd.read_csv(test_path(f'regress/final_{zone}_zone_trips.csv'))
        trips_df = pd.read_csv(test_path(f'output/final_{zone}_zone_trips.csv'))
        print(f"regress trips")
        pdt.assert_frame_equal(trips_df, regress_trips_df), "regress trips"

    # run tests with and without multi processing
    zones = ['1', '2', '3']
    test_combos = [(z, mp) for z in zones for mp in [False, True]]
    for test_combo in test_combos:
        zone, multiprocess = test_combo

        file_path = os.path.join(os.path.dirname(__file__), 'simulation.py')

        if zone == '2':
            base_configs = psrc_example_path(f'configs')
        else:
            base_configs = mtc_example_path(f'configs')

        run_args = ['-c', test_path(f'configs_{zone}_zone'),
                    '-c', example_path(f'configs_{zone}_zone'),
                    '-c', base_configs,
                    '-d', example_path(f'data_{zone}'),
                    '-o', test_path('output')]

        if multiprocess:
            run_args = run_args + ['-s', 'settings_mp.yaml']

        subprocess.run(['coverage', 'run', '-a', file_path] + run_args, check=True)

        regress(zone)


if __name__ == '__main__':
    test_sandag()
