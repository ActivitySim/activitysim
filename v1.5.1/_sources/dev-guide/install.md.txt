(developer-installation)=
# Developer Installation

Installing ActivitySim as a developer is as easy as just using it with *uv*.
The easiest wat to make sure you are up-to-date is to run `uv sync --locked`, 
which will make sure that all the dependencies are loaded correctly.

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

## Cloning from GitHub

Developers who want to work on ActivitySim should clone the repository from GitHub.
If you have the [GitHub command line tool](https://cli.github.com) installed, you 
*could* run this command to clone the consortium's repository:

```{sh}
gh repo clone ActivitySim/activitysim
```

Usually, you'll actually not want to do this.  Instead, **you should work on your
own fork of the repository.**  You can create a fork on GitHub using the web interface
or the command line tool's [fork](https://cli.github.com/manual/gh_repo_fork) command.
Then you would clone the repository by referencing your fork:

```{sh}
gh repo clone MyUsernameOrOrganization/activitysim
```

This way, you can make whatever changes you want in your fork and store them locally
on your computer or push them to GitHub, and they definitely won't step on anyone
else's work.  When you're ready to share your awesome new features, you can open a
pull request to do so.

Also, you may notice that the consortium repository has a huge history and includes
a lot of older data files that you probably don't need.  To reduce your download time
and disk space usage, you can tell Git that you don't want that whole history.  If you
just want to have access to the current code and a couple of years history, you can 
cut off the older history using the `shallow-since` option:

```{sh}
gh repo clone MyUsernameOrOrganization/activitysim -- --shallow-since="2025-01-01"
```

This can reduce the cloning download and disk usage by several gigabytes!  By default
you won't see all your branches appear in this more limited clone, just the `main` 
branch.  

### Fetching a Specific Branch

This is easily solved by downloading individual branches specifically by
name.  

```{sh}
# change into the git repo's directory if not already there
cd activitysim

# Fetch the specific branch named "patch-67"
git fetch origin refs/heads/patch-67:refs/remotes/origin/patch-67

# Check out the branch as a local branch to work on
git checkout -b patch-67 origin/patch-67
```

### Fetching All Branches in a Fork

If you want to work more expansively, you may want to fetch all the branches in
your fork, rather than getting specific branches one at a time.  You can change the
repo settings to do so by adding to the Git config file:

```{sh}
git config remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*"
git fetch --shallow-since="2025-01-01" origin
```
