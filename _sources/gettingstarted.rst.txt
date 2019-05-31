
Getting Started
===============

This page describes how to get started with ActivitySim.

.. note::
   ActivitySim is under development
   

.. index:: installation


Installation
------------

1. Install `Anaconda 64bit Python 3 <https://www.anaconda.com/distribution/>`__.  It is best to use :ref:`anaconda_notes` with ActivitySim.
2. If you access the internet from behind a firewall, then you need to configure your proxy server. To do so, create a .condarc file in your Anaconda installation folder, such as:

::

  proxy_servers:
    http: http://myproxy.org:8080
    https: https://myproxy.org:8080
  ssl_verify: false
 
3. Create and activate an Anaconda environment (basically a Python install just for this project) using Anaconda Prompt or the terminal.  
  
::
    
  conda create -n asimtest python=3.7
  #conda create -n asimtest python=2.7

  #Windows
  activate asimtest
    
  #Mac
  source activate asimtest   

4. Get and install other required libraries on the activated conda Python environment using `pip <https://pypi.org/project/pip>`__:

::
    
  #required packages for running ActivitySim
  conda install cytoolz numpy pandas psutil future
  conda install -c anaconda pytables pyyaml 
  pip install openmatrix zbox
    
  #optional required packages for testing and building documentation
  conda install pytest pytest-cov coveralls pycodestyle
  conda install sphinx numpydoc sphinx_rtd_theme

5. If you access the internet from behind a firewall, then you need to configure your proxy server when downloading packages. 

For `conda` for example, create a `.condarc` file in your Anaconda installation folder with the following:

::
  
  proxy_servers:
    http: http://myproxy.org:8080
    https: https://myproxy.org:8080
  ssl_verify: false

For `pip` for example:
     
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

  ActivitySim is a 64bit Python 2 or 3 library that uses a number of packages from the
  scientific Python ecosystem, most notably `pandas <http://pandas.pydata.org>`__ 
  and `numpy <http://numpy.org>`__. ActivitySim currently supports Python 2, but Python 2
  will be `retired <https://pythonclock.org/>`__ at the end of 2019 so Python 3 is recommended.
   
  The recommended way to get your own scientific Python installation is to
  install 64 bit Anaconda, which contains many of the libraries upon which
  ActivitySim depends + some handy Python installation management tools.  

  Anaconda includes the ``conda`` command line tool, which does a number of useful 
  things, including creating `environments <http://conda.pydata.org/docs/using/envs.html>`__ 
  (i.e. stand-alone Python installations/instances/sandboxes) that are the recommended 
  way to work with multiple versions of Python on one machine.  Using conda 
  environments keeps multiple Python setups from conflicting with one another.
  
  You need to activate the activitysim environment each time you start a new command 
  session.  You can remove an environment with ``conda remove -n asimtest --all`` and 
  check the current active environment with ``conda info -e``.
  
  For more information on Anaconda, see Anaconda's `getting started 
  <https://docs.anaconda.com/anaconda/user-guide/getting-started>`__ guide.

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
  python simulation.py
   
* Review the outputs in the ``output`` folder

.. note::
   Common configuration settings can be overidden at runtime.  See ``python simulation.py -h``.

Hardware
--------

The computing hardware required to run a model implemented in the ActivitySim framework generally depends on:

* the number of households to be simulated for disaggregate model steps
* the number of model zones (for each zone system) for aggregate model steps
* the number and size of network skims by mode and time-of-day
* the desired runtimes

ActivitySim framework models use a significant amount of RAM since they store data in-memory to reduce 
access time in order to minimize runtime.  For example, the example MTC Travel Model One model has 2.7 million 
households, 7.5 people, 1475 zones, 826 network skims and has been run between one hour and one day depending 
on the amount of RAM and number of processors allocated.

.. note::
   ActivitySim has been run in the cloud, on both Windows and Linux OS using 
   `Microsoft Azure <https://azure.microsoft.com/en-us/>`__.  Example configurations, 
   Azure scripts, runtimes, and costs are in the ``example_azure`` folder.