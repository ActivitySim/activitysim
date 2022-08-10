
# Workflows

ActivitySim workflows use the [`pypyr`](https://pypyr.io/) library. This
task runner is more flexible than orca, and relies on isolated 'context'
namespaces rather than storing tables and configurations in Python's
global state. Workflows are activated using the `activitysim workflow`
command line interface.

You can run a workflow from a local working directory, or one of
a number of pre-packaged workflows that are included with ActivitySim.

A collection of workflows used to compare the new *sharrow* code against
legacy implementations can be found [here](https://github.com/camsys/activitysim/tree/sharrow-black/activitysim/workflows/sharrow-contrast).

To invoke a pre-packaged workflow, you can provide the workflow's
name and location within the `activitysim/activitysim/workflows` directory.

```shell
activitysim workflow sharrow-contrast/mtc_mini
```
