(developer-installation)=
# Developer Installation

Installing ActivitySim as a developer is almost as easy as just using it,
but making some tweaks to the processes enables live code development and
testing.

## Package Manager

ActivitySim has a lot of dependencies.  It's easiest and fastest to install
them using a package manager like conda, or its faster and free sibling
[Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).
Depending on your security settings, you might need to install in a
container like docker, instructions for that are coming soon.

Note that if you are installing `mamba`, you only should install `mamba`
in the *base* environment. If you install `mamba` itself in other environments,
it will not function correctly.  If you've got an existing conda installation
and you want to install mamba into it, you can install mamba into the *base*
environment like this:

```sh
conda update conda -n base
conda install -n base -c conda-forge mamba
```

While you are at it, if you are a Jupyter user you might want to also install
`nb_conda_kernels` in your base conda environment alongside any other `jupyter`
libraries:

```sh
mamba install -n base nb_conda_kernels -c conda-forge
```

This will ensure your development environments are selectable as kernels in
Jupyter Notebook/Lab/Etc.

## Environment

It's convenient to start from a completely clean conda environment
and git repository. Assuming you have `mamba` installed, and you
want to install in a new directory called "workspace" run:

```sh
mkdir workspace
cd workspace
mamba env create -p ASIM-ENV --file https://raw.githubusercontent.com/ActivitySim/activitysim/main/conda-environments/activitysim-dev-base.yml
conda activate ./ASIM-ENV
git clone https://github.com/ActivitySim/sharrow.git
python -m pip install -e ./sharrow
git clone https://github.com/ActivitySim/activitysim.git
python -m pip install -e ./activitysim
```

Note the above commands will create an environment with all the
necessary dependencies, clone both ActivitySim and sharrow from GitHub,
and `pip install` each of these libraries in editable mode, which
will allow your code changes to be reflected when running ActivitySim
in this environment.

Depending on what you are working on, you may want to check out a branch
other than `develop`.  To do so, you can point the `git switch` command
above to any other existing branch name.  If you want to start an new
branch, first create it with `git branch cool-new-feature` and then switch
to it with `git switch cool-new-feature`.

Now your environment should be ready to use.  Happy coding!

```{important}
If you add to the ActivitySim dependencies, make sure to also update
the environments in `conda-environments`, which are used for testing
and development.
```
