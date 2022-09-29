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
