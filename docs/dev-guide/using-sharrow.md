
# Using Sharrow

This page will walk through an exercise of running a model with `sharrow`.


## Getting the Code

We'll assume that activitysim and sharrow have been installed in editable
mode, per the [developer install instructions](install.md).

The code to implement `sharrow` in `activitysim` is in pull request 
[#542](https://github.com/ActivitySim/activitysim/pull/542). 

```sh
cd activitysim
gh pr checkout 542  # use `gh auth login` first if needed
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

All these performance tests assume you have suffient RAM to run without chunking.

