
Getting Started
===============

This page describes how to get started with ActivitySim.

.. note::
   ActivitySim is under development
   

.. index:: installation


Installation
------------

1. Install `Anaconda 64bit Python 2.7 <https://www.continuum.io/downloads>`__.  It is best to use :ref:`anaconda_notes` with ActivitySim.
2. If you access the internet from behind a firewall, then you need to configure your proxy server. To do so, create a .condarc file in your Anaconda installation folder, such as:

::

  proxy_servers:
    http: http://myproxy.org:8080
    https: https://myproxy.org:8080
  ssl_verify: false
 
3. Create and activate an Anaconda environment (basically a Python install just for this project)
  
::
    
  conda create -n asimtest python=2.7
    
  #Windows
  activate asimtest
    
  #Mac
  source activate asimtest   

4. Get and install other required libraries on the activated conda Python environment using `pip <https://pypi.org/project/pip>`__:

::
    
  #required packages for running ActivitySim
  pip install cytoolz numpy pandas tables pyyaml psutil
  pip install orca openmatrix zbox
    
  #optional required packages for testing and building documentation
  pip install pytest pytest-cov coveralls pycodestyle
  pip install sphinx numpydoc sphinx_rtd_theme

5. If you access the internet from behind a firewall, then you need to configure your proxy server when downloading packages. For example:
     
::

  pip install --trusted-host pypi.python.org --proxy=myproxy.org:8080  openmatrix


6. Get and install the ActivitySim package on the activated conda Python environment:

::

  #new install
  pip install activitysim
  
  #update to a new release
  pip install -U activitysim


.. _anaconda_notes :

Anaconda
~~~~~~~~

.. note::

  ActivitySim is a 64bit Python 2.7 library that uses a number of packages from the
  scientific Python ecosystem, most notably `pandas <http://pandas.pydata.org>`__ 
  and `numpy <http://numpy.org>`__. ActivitySim does not currently support Python 3.
   
  The recommended way to get your own scientific Python installation is to
  install Anaconda 2 64bit, which contains many of the libraries upon which
  ActivitySim depends + some handy Python installation management tools.  

  Anaconda includes the ``conda`` command line tool, which does a number of useful 
  things, including creating `environments <http://conda.pydata.org/docs/using/envs.html>`__ 
  (i.e. stand-alone Python installations/instances/sandboxes) that are the recommended 
  way to work with multiple versions of Python on one machine.  Using conda 
  environments keeps multiple Python setups from conflicting with one another.
  
  You need to activate the activitysim environment each time you start a new command 
  session.  You can remove an environment with ``conda remove -n asimtest --all`` and 
  check the current active environment with ``conda info -e``.

  If numexpr (which numpy requires) fails to install, you may need 
  the `Microsoft Visual C++ Compiler for Python <http://aka.ms/vcpython27>`__. 

Run the Example
---------------

To setup and run the :ref:`example`, do the following:

* Copy ``mtc_asim.h5`` and ``skims.omx`` from ``activitysim\abm\test\data`` to ``example\data``.
* Open a command prompt in the ``example`` folder
* Run the following commands:
  
::

  #Windows
  activate asimtest
    
  #Mac
  source activate asimtest
  
  #run example
  python run_populationsim.py
   
* Review the outputs in the ``output`` folder

Hardware
--------

The computing hardware required to run a model implemented in the ActivitySim framework generally depends on:

* the number of households to be simulated for disaggregate model steps
* the number of model zones (for each zone system) for aggregate model steps
* the number and size of network skims by mode and time-of-day
* the desired runtimes

ActivitySim framework models utilize a significant amount of RAM since they store data in-memory to reduce 
access time in order to minimize runtimes.  For example, the example MTC Travel Model One model has 2.7 million 
households, 1475 zones, 826 skims and runs for approximately 19 hours in a single process/thread.  The full 
scale (all households, zones, skims, and sub-models) Travel Model One example is run on a Windows 
server with 164GB of RAM and 40 CPUs.  Parallelization is **NOT YET IMPLEMENTED** but is planned for 
Fall 2018 and will make full use of the available CPUs.
