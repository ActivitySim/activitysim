from ez_setup import use_setuptools
use_setuptools()  # nopep8

from setuptools import setup, find_packages

with open('README.rst') as file:
    long_description = file.read()

setup(
    name='activitysim',
    version='0.4',
    description='Activity-Based Travel Modeling',
    author='contributing authors',
    author_email='ben.stabler@rsginc.com',
    license='BSD-3',
    url='https://github.com/udst/activitysim',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 2.7',
        'License :: OSI Approved :: BSD License'
    ],
    long_description=long_description,
    packages=find_packages(exclude=['*.tests']),
    install_requires=[
        'numpy >= 1.12.0',
        'openmatrix >= 0.2.4',
        'orca >= 1.1',
        'pandas >= 0.20.3',
        'pyyaml >= 3.0',
        'tables >= 3.3.0',
        'toolz >= 0.7',
        'zbox >= 1.2',
        'psutil >= 4.1'
    ]
)
