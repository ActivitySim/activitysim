.. ActivitySim documentation users guide

.. _userguide:

Users Guide
===========

The mission of the ActivitySim project is to create and maintain advanced, open-source,
activity-based travel behavior modeling software based on best software development
practices for distribution at no charge to the public.

The ActivitySim project is led by a consortium of Metropolitan Planning Organizations
(MPOs) and other transportation planning agencies, which provides technical direction
and resources to support project development. New member agencies are welcome to join
the consortium. All member agencies help make decisions about development priorities
and benefit from contributions of other agency partners.  Additional information about
the development and management of the ActivitySim is on the `project site <http://www.activitysim.org>`__.

ActivitySim is a common `codebase <https://github.com/activitysim>`__ and individual implementations
can vary in a lot of ways – in terms of space (one-zone or two-zone) and spatial fidelity; model components,
or individual submodels; activity type segmentation (purposes, scheduling); mode alternatives; and other
characteristics. Some `example implementations <https://activitysim.github.io/activitysim/v1.2.0/examples.html>`__ are
openly available to any user.

Note that these model files referenced and instructions provided in this User’s Guide are not complete models;
only the ActivitySim component is included in the files and this User’s Guide. In practice, ActivitySim is a
part of a model system that includes other components, such as network processing, skimming, and assignment,
that generates inputs needed for ActivitySim and processes outputs of ActivitySim.


Contents
--------

.. toctree::
   :maxdepth: 2

   modelsetup
   ways_to_run
   performance/index
   run_primary_example
   model_anatomy
   ../howitworks
   model_dev
   visualization
   example_models
   example_performance
   .. toctree::
   :maxdepth: 1
   other_examples
