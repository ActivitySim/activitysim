# ActivitySim
# See full license in LICENSE.txt.
import os
import subprocess


def test_mp_run():

    file_path = os.path.join(os.path.dirname(__file__), 'run_mp.py')

    subprocess.check_call(['coverage', 'run', file_path])


if __name__ == '__main__':

    test_mp_run()
