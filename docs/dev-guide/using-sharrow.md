
# Using Sharrow

This page will walk through an exercise of running a model with `sharrow`.

## How it Works

Sharrow accelerates ActivitySim in part by using numba to create optimized and
pre-compiled versions of utility specification files, and caching those bits
of code to disk.

```{important}
Running the compiler needs to be done in single-process mode, otherwise the
various process all do the compiling and compete to write to the same cache
location on disk, which is likely to fail.  You can safely run in
multiprocessing mode after all the compilation for all model components is
complete.
```


## Getting the Code

We'll assume that activitysim and sharrow have been installed in editable
mode, per the [developer install instructions](install.md).

The code to implement `sharrow` in `activitysim` is in the
[sharrow-black](https://github.com/camsys/activitysim/tree/sharrow-black) branch
of the [camsys/activitysim](https://github.com/camsys/activitysim) repository.
If you have checked out from this repository directly, you can just switch to
the correct branch:

```sh
cd activitysim
git switch sharrow-black
cd ..
```

If your local Git repository is configured with only the main ActivitySim
parent repo as its remote, you can find the code in
[PR #579](https://github.com/ActivitySim/activitysim/pull/579), accessible on the
command line using the `gh` tool:

```sh
cd activitysim
gh pr checkout 579  # use `gh auth login` first if needed
cd ..
```


## Prototype MTC Examples

Testing with sharrow requires two steps: test mode and production mode.

In test mode, the code is run to compile all the spec files and
ascertain whether the functions are working correctly.  Production mode
can then just run the pre-compiled functions with sharrow, which is much
faster.

You can run both, plus the ActivitySim in "legacy" mode that does not use sharrow,
as well as a reference implementation (version 1.0.4), all together using the
"mini" dataset for testing in one workflow.

```sh
activitysim workflow sharrow-contrast/mtc_mini
```

Alternatively, you can use the full size skims.  To test this model with
100k households and full skims (1475 zones), you can run the "mtc_full" workflow:

```sh
activitysim workflow sharrow-contrast/mtc_full
```

To use the full synthetic population as well, run the multiprocess workflow:

```sh
activitysim workflow sharrow-contrast/mtc_mp
```

Lastly, a comprehensive performance testing suite on the prototype MTC model
(warning: this takes hours!)

```sh
activitysim workflow sharrow-contrast/mtc_comprehensive
```

All these performance tests assume you have sufficient RAM to run without chunking.
