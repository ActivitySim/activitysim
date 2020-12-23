from ez_setup import use_setuptools
use_setuptools()  # nopep8

from setuptools import setup, find_packages

import os
import re

with open(os.path.join('activitysim', '__init__.py')) as f:
    info = re.search(r'__.*', f.read(), re.S)
    exec(info[0])

setup(
    name='activitysim',
    version=__version__,
    description=__doc__,
    author='contributing authors',
    author_email='ben.stabler@rsginc.com',
    license='BSD-3',
    url='https://github.com/activitysim/activitysim',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: BSD License'
    ],
    packages=find_packages(exclude=['*.tests']),
    include_package_data=True,
    entry_points={'console_scripts': ['activitysim=activitysim.cli.main:main']},
    install_requires=[
        'pyarrow >= 2.0',
        'numpy >= 1.16.1',
        'openmatrix >= 0.3.4.1',
        'pandas >= 1.1.0',
        'pyyaml >= 5.1',
        'tables >= 3.5.1',
        'toolz >= 0.8.1',
        'zbox >= 1.2',
        'psutil >= 4.1',
        'requests >= 2.7',
        'numba >= 0.51.2',
    ]
)
