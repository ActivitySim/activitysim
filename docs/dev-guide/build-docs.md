
(write-docs)=
# Documentation

The core documentation for ActivitySim is built with [Sphinx](https://www.sphinx-doc.org).
The input files for this documentation can be written either in
[markdown](https://www.markdownguide.org) with filenames ending in `.md` (preferred
for new documentation pages) or
[reStructuredText](http://docutils.sourceforge.net/rst.html) with filenames ending in `.rst`.
In addition to converting *.md and *.rst files
to html format, Sphinx can also read the inline Python docstrings and convert
them into html as well.  ActivitySim's docstrings are written in
[numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard) format.

## Building the Documentation

Developers who want to test a build of the ActivitySim documentation locally can
do so using `sphinx`.  A pre-packaged conda environment is available to simplify this
process. On the command line, starting from the `activitysim` directory that constitutes the
main repository (i.e. you should see subdirectories including `activitysim`,
`conda-environments`, `docs`, and a few others) run these commands:

```bash
mkdir -p ../.env
mamba env update -p ../.env/DOCBUILD -f conda-environments/docbuild.yml
conda activate ../.env/DOCBUILD
cd docs
make clean
make html
```

This will build the docs in the `docs/_build/html` directory.  They can be viewed
in a web browser using the `file:///` protocol, or by double-clicking on the
`index.html` file (or any other .html file in that directory).

## Automatic Documentation Builds

Documentation can also be rendered online automatically by GitHub.  Several scripts
are included in this repository's GitHub Actions to do so when updates are made
to the `main` or `develop` branches in the primary `ActivitySim` repository.

If you are working in a *fork* of the primary `ActivitySim/activitysim` repository, you
can generate test builds of the documentation by pushing a commit to your branch
with the tag `[makedocs]` in the commit message.  Note to prevent conflicts this
only works on a fork, not within the primary `ActivitySim` repository, and only
on branches named something other than `develop`.  The documentation will then be
published on your own subdomain.  For example, if your fork is `tacocat/activitysim`,
and you are working on the `featuring-cilantro` branch, the GitHub will render your
documentation build at `https://tacocat.github.io/activitysim/featuring-cilantro`.
