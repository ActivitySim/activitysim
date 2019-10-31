from ez_setup import use_setuptools
use_setuptools()  # nopep8

from setuptools import setup, find_packages

setup(
    name='activitysim',
    version='0.9.1',
    description='Activity-Based Travel Modeling',
    author='contributing authors',
    author_email='ben.stabler@rsginc.com',
    license='BSD-3',
    url='https://github.com/activitysim/activitysim',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'License :: OSI Approved :: BSD License'
    ],
    packages=find_packages(exclude=['*.tests']),
    include_package_data=True,
    install_requires=[
        'numpy >= 1.16.1',
        'openmatrix >= 0.3.4.1',
        'pandas >= 0.24.1',
        'pyyaml >= 5.1',
        'tables >= 3.5.1',
        'toolz >= 0.8.1',
        'zbox >= 1.2',
        'psutil >= 4.1',
        'future >= 0.16.0'
    ]
)
