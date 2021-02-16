# ActivitySim
# See full license in LICENSE.txt.
import os
import subprocess

from activitysim.core import inject


def teardown_function(func):
    inject.clear_cache()
    inject.reinject_decorated_tables()


def test_psrc():

    file_path = os.path.join(os.path.dirname(__file__), 'run_psrc.py')

    subprocess.check_call(['coverage', 'run', file_path])


if __name__ == '__main__':

    test_psrc()
