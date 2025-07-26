(developer-installation)=
# Developer Installation

Installing ActivitySim as a developer is as easy as just using it with *uv*.

Depending on what you are working on, you may want to check out a branch
other than the default `main`. To do so, you can use a `git switch` command
to any other existing branch name. If you want to start an new
branch, first create it with `git branch cool-new-feature` and then switch
to it with `git switch cool-new-feature`.

By default, *uv* installs projects in editable mode, such that changes to the 
source code are immediately reflected in the environment. 

```{important}
If you add to the ActivitySim dependencies during development or remove, make 
sure to use the `uv add` and `uv remove` commands so that the `pyproject.toml`
and `uv.lock` files are updated correctly and your virtual environment is 
updated.
```
