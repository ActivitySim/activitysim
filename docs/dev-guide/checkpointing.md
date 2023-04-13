# Checkpointing

```{eval-rst}
.. currentmodule:: activitysim.core.workflow.checkpoint
```

ActivitySim provides a checkpointing mechanism, whereby the content of data tables
can be stored to disk in an intermediate state.  This intermediate state can
subsequently be restored from disk, setting up the data tables to resume
simulation from that point forward.

There are currently two data file formats available for checkpointing:

- [HDF5](https://www.hdfgroup.org/solutions/hdf5/), the longstanding default
  format for ActivitySim checkpointing, and
- [Apache Parquet](https://parquet.apache.org/), added as an option as of
  ActivitySim version 1.3.




## API

```{eval-rst}
.. autosummary::
    :toctree: _generated
    :recursive:

    GenericCheckpointStore
    HdfStore
    ParquetStore
```
