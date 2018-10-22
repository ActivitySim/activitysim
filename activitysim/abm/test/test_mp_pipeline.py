# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402

import os
import subprocess


def test_mp_run():

    file_path = os.path.join(os.path.dirname(__file__), 'run_mp.py')

    subprocess.check_call(['coverage', 'run', file_path])


if __name__ == '__main__':

    test_mp_run()
