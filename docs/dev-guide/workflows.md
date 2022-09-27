(workflows)=
# Workflows

ActivitySim workflows use the [`pypyr`](https://pypyr.io/) library. This
task runner is more flexible than orca, and relies on isolated 'context'
namespaces rather than storing tables and configurations in Python's
global state. Workflows are activated using the `activitysim workflow`
command line interface.

You can run a workflow from a local working directory, or one of
a number of pre-packaged workflows that are included with ActivitySim.


## Workflows for Performance Monitoring

A collection of workflows used to compare the new *sharrow* code against
legacy implementations can be found in the
[sharrow-contrast](https://github.com/camsys/activitysim/tree/sharrow-black/activitysim/workflows/sharrow-contrast)
workflow subdirectory. Each of these first runs the relevant example model in
test mode to compile the relevant functions, and then runs in production mode
to measure runtime and memory usage.  This is followed by another run in
"legacy" mode that does not use sharrow, as well as a run in a reference
implementation (version 1.1.3).  Finally, a report is generated comparing the
results of these various runs, to illustrate performance and validiate output
quality.

### Mini Examples

A collection of "mini" example workflows is provided for typical performance analysis.
Each mini example is based on a small or medium size subarea for each model,
and each runs in single-process mode to illustrate the performance and memory
usage succinctly.  These mini examples are sized so they should run on a typical
modeler's laptop (e.g. a modern CPU, 32 GB of RAM).

| Workflow            | Description                                                                            |
|---------------------|:---------------------------------------------------------------------------------------|
| mtc_mini            | Prototype MTC model on 50K households in San Francisco (190 zones)                     |
| arc_mini            | Prototype ARC model on 10K households in Fulton County (1,296 zones)                   |
| sandag_1zone_mini   | Placeholder SANDAG 1-Zone model on 50K households in a test region (90 zones)          |
| sandag_2zone_mini   | Placeholder SANDAG 2-Zone model on 50K households in a test region (90 TAZs, 690 MAZs) |
| sandag_3zone_mini   | Placeholder SANDAG 3-Zone model on 50K households in a test region (90 TAZs, 690 MAZs) |
| sandag_xborder_mini | Prototype SANDAG Cross-Border model on 50K tours in a test region (94 TAZs, 586 MAZs)  |
| psrc_mini           | Placeholder PSRC Model on 50K households in Seattle (8400 MAZs)                        |
| comprehensive-mini  | Runs all "mini" workflows listed above (warning, this takes a long time!)              |

To invoke a pre-packaged workflow, you can provide the workflow's
name and location within the `activitysim/workflows` directory.
For example, to run the workflow of all "mini" examples, run:

```shell
activitysim workflow sharrow-contrast/comprehensive-mini
```

### Full Size Examples

Running a "full size" example of an ActivitySim model generally requires more
compute resources than the smaller performance tests shown above.  These models
generally require [chunking](chunk_size) to run, and servers with substantial
amounts of RAM.

The "mp" workflows listed below will run each listed example in multiprocess mode.
They will attempt to automatically scale the number of processes and the chunk
size to the available resources of the server where they are run, but some models
require substantial resources and may not run correctly on inadequate hardware.

| Workflow        | Description                                                      |
|-----------------|:-----------------------------------------------------------------|
| mtc_mp          | Prototype MTC model                                              |
| arc_mp          | Prototype ARC model, 5,922 zones                                 |
| sandag_1zone_mp | Placeholder SANDAG 1-Zone model, 1.2M households and 4,984 zones |
