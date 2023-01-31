# How to issue an ActivitySim release

> **WARNING: These instructions are incomplete.**

01. Check that the branch you intend to release is passing tests on Travis.
    If it's not passing there you should not release it.

00. Start from a completely clean conda environment
    and git repository.  Assuming you have `conda` installed, you can do so
    by starting where ActivitySim is not yet cloned (e.g. in an empty
    directory) and running:
    ```sh
    conda create -n TEMP-ASIM-DEV python=3.9 git gh -c conda-forge --override-channels
    conda activate TEMP-ASIM-DEV
    gh auth login   # <--- (only needed if gh is not logged in)
    gh repo clone ActivitySim/activitysim
    cd activitysim
    ```

00. Per project policy, code on the main branch should have been released,
    but if you are *preparing* a release then the code should be on the `develop`
    branch.  Switch to that branch now, and make sure it is synced to the
    version on GitHub:
    ```sh
    git switch develop
    git pull
    ```

00. Update your Conda environment for testing.  We do not want to use an
    existing environment on your machine, as it may be out-of-date
    and we want to make sure everything passes muster using the
    most up-to-date dependencies available.  The following command
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
    Travis and are platform independent, so they should not be necessary to
    replicate locally, but are listed here for completeness.
    ```sh
    black --check --diff .
    ```

00. Run the regular test suite on Windows. Travis tests are done on Linux,
    but most users are on Windows, and the test suite should also be run
    on Windows to ensure that it works on that platform as well.  If you
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
    large to run on Travis, and will take a lot of time (many hours) to
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

00. Use bump2version to tag the release commit and update the
    version number.  The following code will generate a "patch" release,
    incrementing the third value in the version number (i.e. "1.2.3"
    becomes "1.2.4").  Alternatively, make a "minor" or "major" release.
    The `--list` command will generate output to your console to confirm
    that the old and new version numbers are what you expect, before you
    push the commit (with the changed version in the code) and tags to
    GitHub.
    ```sh
    bump2version patch --list
    ```

    It is also possible to make a development pre-release. To do so,
    explicitly set the version number to the next patch plus a ".devN"
    suffix:

    ```sh
    bump2version patch --new-version 1.2.3.dev0 --list
    ```

    Then, when ready to make a "final" release, set the version by
    explicitly removing the suffix:
    ```sh
    bump2version patch --new-version 1.2.3 --list
    ```

00. Push the tagged commit to GitHub.
    ```sh
    git push --tags
    ```

00. For non-development releases, open a pull request to merge the proposed
    release into main. The following command will open a web browser for
    you to create the pull request.
    ```sh
    gh pr create --web
    ```
    After creating the PR, confirm with the ActivitySim PMC that the release
    is ready before actually merging it.

    Once final approval is granted, merge the PR into main.  The presence
    of the git tags added earlier will trigger automated build steps to
    prepare and deploy the release to pypi and conda-forge.

00. Create a "release" on GitHub.
    ```sh
    gh release create v1.2.3
    ```
    For a development pre-release, include the `--prerelease` argument.
    As the project's policy is that only formally released code is merged
    to the main branch, any pre-release should also be built against a
    non-default branch.  For example, to pre-release from the `develop`
    branch:
    ```sh
    gh release create v1.2.3.dev0 \
        --prerelease \
        --target develop \
        --notes "this pre-release is for a cool new feature" \
        --title "Development Pre-Release"
    ```

00. Clean up your workspace, including removing the Conda environment used for
    testing (which will prevent you from accidentally using an old
    environment when you should have a fresh up-to-date one next time).
    ```sh
    conda deactivate
    conda env remove -n TEMP-ASIM-DEV
    ```
