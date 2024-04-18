(skim-datasets)=
# Using Skim Dataset

ActivitySim 1.2 offers two internal frameworks for managing skim data.

- [`SkimDict`](activitysim.core.skim_dictionary.SkimDict), the legacy
  framework, that stores all skim data in one large omnibus array, with
  various offset lookups and tools to access values.
- [`SkimDataset`](activitysim.core.skim_dataset.SkimDataset), an
  [xarray.Dataset]() based framework, which mimics the
  [`SkimDict`](activitysim.core.skim_dictionary.SkimDict) interface, and
  adds a number of features optimized specifically for use with `sharrow`.
  This framework is automatically used when sharrow is enabled, and there
  is no user configuration to enable it separately.

## Skims in Shared Memory

These two internal frameworks manage shared memory differently when running
ActivitySim in multiprocessing mode.

For [`SkimDict`](activitysim.core.skim_dictionary.SkimDict), shared memory is
used only when running with multiprocessing active, and is allocated via the
[`allocate_shared_skim_buffers`](activitysim.core.mp_tasks.allocate_shared_skim_buffers)
function, which in turn invokes
[Network_LOS.allocate_shared_skim_buffers](activitysim.core.los.Network_LOS.allocate_shared_skim_buffers).

For [`SkimDataset`](activitysim.core.skim_dataset.SkimDataset), shared memory is
used regardless of whether multiprocessing is active or not.  The shared memory
allocation is done via the `Dataset.shm.to_shared_memory` method called at the end
of the [`load_skim_dataset_to_shared_memory`](activitysim.core.skim_dataset.load_skim_dataset_to_shared_memory)
function.

## Skim Dataset API

```{eval-rst}
.. automodule:: activitysim.core.skim_dataset
   :members:
```
