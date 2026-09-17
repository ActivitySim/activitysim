# Multiprocessing

ActivitySim implements an extensive set of multiprocessing capabilities to take
advantage of modern multicore CPUs.  The number of processes used is controlled
by the `multiprocessing` configuration setting in the `settings.yaml` file.

There are a number of settings that need to be configured to use multiprocessing
in ActivitySim.

- [`multiprocess`](activitysim.core.configuration.Settings.multiprocess)

  A boolean setting that enables or disables multiprocessing.

- [`num_processes`](activitysim.core.configuration.Settings.num_processes)

  This controls the number of processes used by ActivitySim in multiprocessing,
  unless the value is overloaded in a particular multiprocess step.

- [`multiprocess_steps`](activitysim.core.configuration.Settings.multiprocess_steps)

  This setting controls which model components are handled by multiprocessing,
  and when the model flow needs to begin and end each multiprocessing step. An
  individual multiprocessing step can include one or more model components, and
  can be configured to use a different number of processes than other
  multiprocessing step

These settings are described in more detail in the [Configuration](configuration)
section.

For the most part, the preparation of the `multiprocess_steps` configuration is
the domain of model developers, and users should not modify these settings.
Details for setting up multiprocessing are described in the
[Multiprocessing](multiprocessing_in_detail) section of the developer's guide.
The selection of what components to group together to run in multiprocessed steps is
a complex decision that depends on the mathematical structure of the components
being run, and generally cannot be arbitrarily changed.  However, users can
potentially tune the system for best performance by controlling the number of
processes used by ActivitySim in each multiprocessing step.


## How Many Processes

Understanding ideal number of processes to use in multiprocessing is complex.  Using
more processes might typically be expected to reduce overall model runtime, and
it usually does for the first few processes added.  However, as the number of
processes grows, the marginal benefit of adding more processes decreases, in large
part because there is additional overhead associated with each process, so using too many
processes can actually slow down the model.

Experiments by the ActivitySim consortium have generally found that the optimal
number of processes is usually around 10, even if the machine has many more
cores available.  Of course, this is a general rule of thumb, and the optimal
number of processes will vary depending on the specific model being run, the
hardware being used, and the specific configuration of the model.

```{tip}
The optimal number of processes should always be less than or equal to the
number of CPU cores available.  If running ActivitySim on a remote server or
cloud instance, it may be desirable to limit the number of processes to at least
one less than the number of available cores, to ensure that the system remains
responsive.
```
