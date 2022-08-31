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


def test_mtc_accessibilities():

    def example_path(dirname):
        resource = os.path.join('examples', 'example_mtc_accessibilities', dirname)
        return pkg_resources.resource_filename('activitysim', resource)

    def example_mtc_path(dirname):
        resource = os.path.join('examples', 'example_mtc', dirname)
        return pkg_resources.resource_filename('activitysim', resource)

    def test_path(dirname):
        return os.path.join(os.path.dirname(__file__), dirname)

    def regress():
        regress_trips_df = pd.read_csv(test_path('regress/final_trips.csv'))
        final_trips_df = pd.read_csv(test_path('output/final_trips.csv'))
        pdt.assert_frame_equal(final_trips_df, regress_trips_df)

    sim_file_path = os.path.join(os.path.dirname(__file__), 'simulation.py')
    acc_file_path = os.path.join(os.path.dirname(__file__), 'disaggregate_accessibility_model.py')

    # TODO run disagg accessibilities then run the model. Or run as a model step?
    subprocess.run(['run', '-a', acc_file_path], check=True)

    subprocess.run(['coverage', 'run', '-a', sim_file_path,
                    '-c', test_path('configs'), '-c', example_path('configs'),
                    '-c', example_mtc_path('configs'),
                    '-d', example_mtc_path('data'),
                    '-o', test_path('output')], check=True)

    regress()


if __name__ == '__main__':
    # TODO NOT WORKING YET
    test_mtc_accessibilities()
