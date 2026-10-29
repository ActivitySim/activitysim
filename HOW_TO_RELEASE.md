# How to issue an ActivitySim release

1.  (Optional) Update dependencies. Pull in the latest versions of all
    dependencies that still satisfy current constraints:
    ```sh
    uv lock --upgrade
    ```
    Make any adjustments using `uv add ...` with version constraints as needed
    to pass tests. If a version constraint is created, there should be a github
    issue created that identifies the need in the future to debug and remove
    this constraint. Merge the updated `uv.lock` through a pull request, so that
    the full test suite runs on it before the release.

2.  Update the change log. In `docs/dev-guide/changes.md`, move the entries
    under "Upcoming Changes" into a new section for the release (for example
    `## v1.2`), and merge this change through a pull request.

3.  Check that the main branch is passing tests, especially the "core tests" on
    [GitHub Actions](https://github.com/ActivitySim/activitysim/actions/workflows/core_tests.yml).
    It is generally the policy that the main branch should always be passing tests,
    because PRs must pass tests before they can be merged.  However, it is
    possible that tests may fail after a PR is merged, so it is important to
    double-check that the main branch is passing tests before issuing a release.

    Pull requests run the regional model, estimation, and other slower tests on
    the oldest supported Python version only; every merge into `main` runs them
    on all supported versions. A failure that only affects a newer Python
    version therefore first shows up in the runs on `main`.

4.  Test the two supported example models, MTC and SANDAG, at full scale if
    possible. These runs are too large for GitHub Actions and take many hours.
    Each example repository has a script that downloads the full-scale data
    and runs the model:
    - [activitysim-prototype-mtc](https://github.com/ActivitySim/activitysim-prototype-mtc):
      `scripts/run-large-sharrow.py`
    - [sandag-abm3-example](https://github.com/ActivitySim/sandag-abm3-example):
      `scripts/run-large-sharrow.py` (see `scripts/README.md` there)

    On their own, these scripts use the ActivitySim release locked in each
    example repository. To test the code being released instead, run them with
    the environment of an up-to-date ActivitySim `main` checkout. For example,
    with both example repositories cloned next to the `activitysim` directory:
    ```sh
    cd activitysim-prototype-mtc
    uv run --project ../activitysim --locked python scripts/run-large-sharrow.py
    cd ../sandag-abm3-example
    uv run --project ../activitysim --locked --with wring python scripts/run-large-sharrow.py
    ```
    The scripts simulate a sample of the full-scale data (500,000 households
    for MTC, 100,000 for SANDAG). For a full-scale run, set
    `households_sample_size` to `0` in the script.

5.  Test the notebooks in `activitysim/examples/prototype_mtc/notebooks`.
    There are also demo notebooks for estimation; those are run automatically
    by the "Estimation Notebooks Test" job in the core tests.

6.  Tag the release commit with the new version number.  ActivitySim uses
    dynamic versioning, so the version number is not stored in a file but
    is instead read from the most recent git tag, so it is important to tag
    the repository with the correct version.  Tag the latest commit on `main`
    of ActivitySim/activitysim, not of a fork. The following commands will
    generate a new tag with the version number "1.2.3" (for example), assuming
    the `origin` remote points to ActivitySim/activitysim:
    ```sh
    git switch main
    git pull origin main
    git tag -a v1.2.3 -m "Release v1.2.3"
    ```

7.  Push the tag to GitHub.
    ```sh
    git push origin v1.2.3
    ```

8.  Create a "release" on GitHub.  You can do this from the command line using
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
    new release to PyPI, assuming there are no errors. It also publishes the
    documentation for the release to the `v1.2.3` folder of the `gh-pages` branch.

9.  Update the documentation's version menu and landing page. Both are files
    at the root of the `gh-pages` branch, edited by hand through a pull request
    into that branch once the release's documentation has been published there:
    - `switcher.json`: the menu has one entry per minor version, pointing at its
      latest release. For a new minor version, add an entry for it at the top
      (below "Development"), for example
      `{"name": "1.2.0 (Latest)", "version": "1.2.0", "url": "https://activitysim.github.io/activitysim/v1.2.0/index.html"}`,
      and rename the previous "(Latest)" entry to its minor version (for
      example "1.1"). For a patch release, update the existing entry for its
      minor version instead: its version and URL, and its name if that shows
      the full version (for example "1.2.0 (Latest)" becomes "1.2.1 (Latest)").
    - `index.html`: point the redirect at the new version (for example
      `url='v1.2.3'`).

10. Update the conda-forge package. Shortly after the release is published on
    PyPI, a bot opens a pull request on the
    [activitysim feedstock](https://github.com/conda-forge/activitysim-feedstock)
    and merges it automatically if its checks pass. If the dependencies or the
    supported Python versions changed, update the recipe to match
    `pyproject.toml`, either in the bot's pull request before it is merged or
    in a separate pull request.
