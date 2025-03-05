# How to issue an ActivitySim release

00. Check that the main branch is passing tests, especially the "core tests" on
    [GitHub Actions](https://github.com/ActivitySim/activitysim/actions/workflows/core_tests.yml). 
    It is generally the policy that the main branch should always be passing tests, 
    becuase PRs must pass tests before they can be merged.  However, it is 
    possible that tests may fail after a PR is merged, so it is important to
    double-check that the main branch is passing tests before issuing a release.

00. Start from a completely clean conda environment
    and git repository.  Assuming you have `conda` installed, you can do so
    by starting where ActivitySim is not yet cloned (e.g. in an empty
    directory) and running:
    ```sh
    conda create -p ./TEMP-ASIM-DEV python=3.10 git gh -c conda-forge --override-channels
    conda activate ./TEMP-ASIM-DEV
    gh auth login   # <--- (only needed if gh is not logged in)
    gh repo clone ActivitySim/activitysim
    cd activitysim
    ```

00. Update your Conda environment for testing.  We do not want to use an
    existing environment on your machine, as it may be out-of-sync
    and we want to make sure everything passes muster using the
    dependencies as they are available today.  The following command
    will update the active environment (we made this to be `TEMP-ASIM-DEV`
    if you followed the directions above).
    ```sh
    conda env update --file=conda-environments/activitysim-dev.yml
    ```
    If you add to the ActivitySim dependencies, make sure to also update
    the environments in `conda-environments`, which are used for testing
    and development.  If they are not updated, these environments will end
    up with dependencies loaded from *pip* instead of *conda-forge*.

00. Run black to ensure that the codebase passes all style checks.
    This check should only take a few seconds.  These checks are also done on
    GitHub Actions and are platform independent, so they should not be necessary to
    replicate locally, but are listed here for completeness.
    ```sh
    black --check --diff .
    ```

00. Run the regular test suite on Windows. Most GitHub Actions tests are done on
    Linux (it's faster to start up and run a new clean VM for testing) but most
    users are on Windows, and the test suite should also be run on Windows to
    ensure that it works on that platform as well.  If you
    are not preparing this release on Windows, you should be sure to run
    at least through this step on a Windows machine before finalizing a
    release.

    A few of the tests require pre-created data that is not included in the
    repository directly, but rather recreated on the fly before testing. The
    regular test suite takes some time to run, between about half an hour and
    two hours depending on the specs of your machine.
    ```sh
    python activitysim/examples/placeholder_multiple_zone/scripts/two_zone_example_data.py
    python activitysim/examples/placeholder_multiple_zone/scripts/three_zone_example_data.py
    pytest .
    ```

00. Test the full-scale regional examples. These examples are big, too
    large to run on GitHub Actions, and will take a lot of time (many hours) to
    download and run.
    ```sh
    mkdir tmp-asim
    cd activitysim/examples
    python create_run_all_examples.py > ../../tmp-asim/run_all_examples.bat
    cd ../../tmp-asim
    call run_all_examples.bat
    ```
    These tests will run through the gamut even if some of them crash, so
    if you don't sit and watch them go (please don't do this) you'll need
    to scan through the results to make sure there are no errors after the
    fact.
    ```sh
    python ../activitysim/examples/scan_examples_for_errors.py .
    ```

00. Test the notebooks in `activitysim/examples/prototype_mtc/notebooks`.
    There are also demo notebooks for estimation, but their functionality
    is completely tested in the unit tests run previously.

00. Tag the release commit with the new version number.  ActivitySim uses
    dynamic versioning, so the version number is not stored in a file but
    is instead read from the most recent git tag, so it is important to tag
    the repository with the correct version.  The following command will 
    generate a new tag with the version number "1.2.3" (for example):
    ```sh
    git -a v1.2.3 -m "Release v1.2.3"
    ```

00. Push the tagged commit to GitHub.
    ```sh
    git push --tags
    ```

00. Create a "release" on GitHub.  You can do this from the command line using
    the `gh` command line tool:
    ```sh
    gh release create v1.2.3
    ```
    But it may be easier to do this through the 
    [GitHub web interface](https://github.com/ActivitySim/activitysim/releases/new),
    where you can select the tag you just created and add a title and description.
    Both the interactive command line tool shown above and the web interface include
    the ability to create release notes automatically from the commit messages of
    all accepted PRs since the last release, but you may want to add additional
    notes to the release to highlight important changes or new features.

    The process of creating and tagging a release will automatically
    trigger various GitHub Actions scripts to build, test, and publish the
    new release to PyPI and conda forge, assuming there are no errors.

00. Build the ActivitySim Standalone Windows Installer.  This is done using 
    GitHub Actions, but it is not done automatically when a release is created, 
    instead it requires a manual workflow dispatch trigger.  You can do this by 
    going to the [build_installer workflow page](https://github.com/ActivitySim/activitysim/actions/workflows/build_installer.yml)
    and clicking on the "Run workflow" button.  You will need to provide the 
    version number and choose to add the built installer to the release.

00. Clean up your workspace, including removing the Conda environment used for
    testing (which will prevent you from accidentally using an old
    environment when you should have a fresh up-to-date one next time).
    ```sh
    conda deactivate
    conda env remove -p ./TEMP-ASIM-DEV
    ```
