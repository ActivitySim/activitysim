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
* `pandas <http://pandas.pydata.org>`__ >= 0.13.1 \*
* `tables <http://www.pytables.org/moin>`__ >= 3.1.0 \*
* `toolz <http://toolz.readthedocs.org/en/latest/>`__ or
  `cytoolz <https://github.com/pytoolz/cytoolz>`__ >= 0.7 \*\*
* `zbox <https://github.com/jiffyclub/zbox>`__ >= 1.2 \*\*\*

| \* Pre-installed with Anaconda
| \*\* Available via conda_ or pip_.
| \*\*\* Available via pip_.

ActivitySim
~~~~~~~~~~~

ActivitySim can be installed by downloading the source from
the `GitHub repo <https://github.com/synthicity/activitysim>`__.
Once downloaded ``cd`` into the ``activitysim`` directory and run the
command ``python setup.py install``.

As ActivitySim matures we'll make official releases so that it can be installed
via pip and/or conda.

.. _Anaconda: http://docs.continuum.io/anaconda/index.html
.. _conda: http://conda.pydata.org/
.. _pip: https://pip.pypa.io/en/stable/
