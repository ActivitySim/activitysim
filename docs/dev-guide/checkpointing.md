# Checkpointing

```{eval-rst}
.. currentmodule:: activitysim.core.workflow.checkpoint
```

ActivitySim provides a checkpointing mechanism, whereby the content of data tables
can be stored to disk in an intermediate state.  This intermediate state can
subsequently be restored from disk, setting up the data tables to resume
simulation from that point forward.


```{eval-rst}
.. autosummary::
    :toctree: _generated
    :recursive:

    GenericCheckpointStore
    HdfStore
    ParquetStore
```
