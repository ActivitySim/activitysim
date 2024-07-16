# Runtime Performance

Achieving "good" runtime performance in ActivitySim depends on several factors,
including the model size (number of zones, number of households), model complexity,
(eg. number of components, length and complexity of utility specifications), and
hardware capabilities (amount of RAM, number of CPU cores, amount of on-chip cache,
speed of storage devices).

Much of the work in making ActivitySim run efficiently sits on the shoulders of
model developers, who design models and determine the level of detail and complexity
of the model components. However, for any given fully developed ActivitySim model,
there are several techniques that users can employ to improve runtime performance,
and to tune that performance to the specific hardware on which the model is being run.
These techniques are the focus of this section.

```{tip}
If you are unsure about the best settings for runtime performance of your model,
try running with

    multiprocessing: true
    num_processes: 10
    chunk_training_mode: explicit
    sharrow: true

If your machine has fewer than 11 cores, try decreasing `num_processes` to be
one less than the number of cores available.
```

```{eval-rst}
.. toctree::
   :maxdepth: 1

   multiprocessing
   multithreading
   chunking
   sharrow
```
