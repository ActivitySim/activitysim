==================
Command Line Tools
==================


activitysim run
---------------

.. argparse::
    :module: activitysim.cli.main
    :func: parser
    :prog: activitysim
    :path: run


activitysim create
------------------

.. argparse::
    :module: activitysim.cli.main
    :func: parser
    :prog: activitysim
    :path: create

    -d --destination : @replace
         Path to new project directory.  If this directory already exists, the
         newly created example will be copied to a subdirectory within the
         existing directory, and named according to the example name.  Otherwise,
         a new directory is created with this name and the newly created example
         will be copied directly into it.

