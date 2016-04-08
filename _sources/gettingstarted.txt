Getting Started
===============

Installation
------------

.. note::
   In the instructions below we will direct you to run various commands.
   On Mac and Linux these should go in your standard terminal.
   On Windows you may use the standard command prompt, the Anaconda
   command prompt, or even Git Bash (if you have that installed).

Anaconda
~~~~~~~~

ActivitySim is a Python library that uses a number of packages from the
scientific Python ecosystem.
The easiest way to get your own scientific Python installation is to
install Anaconda_, which contains many of the libraries upon which
ActivitySim depends.

Once you have a Python installation, install the dependencies listed below and
then install ActivitySim.

Dependencies
~~~~~~~~~~~~

ActivitySim depends on the following libraries, some of which are pre-installed
with Anaconda:

* `numpy <http://numpy.org>`__ >= 1.8.0 \*
* `openmatrix <https://pypi.python.org/pypi/OpenMatrix/0.2.3>`__ >= 0.2.2 \*\*\*
* `orca <https://synthicity.github.io/orca/>`__ >= 1.1 \*\*\*
* `pandas <http://pandas.pydata.org>`__ >= 0.18.0 \*
* `pyyaml <http://pyyaml.org/wiki/PyYAML>`__ >= 3.0 \*
* `tables <http://www.pytables.org/moin>`__ >= 3.1.0 \*
* `toolz <http://toolz.readthedocs.org/en/latest/>`__ or
  `cytoolz <https://github.com/pytoolz/cytoolz>`__ >= 0.7 \*\*
* `zbox <https://github.com/jiffyclub/zbox>`__ >= 1.2 \*\*\*

| \* Pre-installed with Anaconda
| \*\* Available via conda_ or pip_.
| \*\*\* Available via pip_.
|

ActivitySim
~~~~~~~~~~~

conda
^^^^^

Conda is the easiest way to install ActivitySim because it can install
binary sources for all of ActivitySim's dependencies on any platform.
Conda is installed with Anaconda, so if you're using Anaconda you can run
the following command to install ActivitySim and all its dependencies::

    conda install --channel synthicity --channel jiffyclub activitysim

The extra channels are necessary so that conda can find some packages that
are not on the standard conda channels.

pip
^^^

If you're not using Anaconda/Conda, ActivitySim can also be installed
`from PyPI <https://pypi.python.org/pypi/activitysim>`__ using pip_.
Pip will attempt to install any dependencies that are not already installed.

::

    pip install activitysim

Development
^^^^^^^^^^^

Development versions of ActivitySim can be installed by downloading the source
from the `GitHub repo <https://github.com/udst/activitysim>`__.
Once downloaded ``cd`` into the ``activitysim`` directory and run the
command ``python setup.py install``. Or run ``python setup.py develop`` if you
wish to make changes to the package source and see the results without reinstalling.)

.. _Anaconda: http://docs.continuum.io/anaconda/index.html
.. _conda: http://conda.pydata.org/
.. _pip: https://pip.pypa.io/en/stable/
