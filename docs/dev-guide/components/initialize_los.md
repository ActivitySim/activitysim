(component-initialize-los)=
# Initialize LOS

```{eval-rst}
.. currentmodule:: activitysim.abm.models.initialize_los
```

The initialize LOS model isn't really a model, but rather a series of data processing steps in the data pipeline.
The initialize LOS model does two things:

  * Loads skims and cache for later if desired
  * Loads network LOS inputs for transit virtual path building (see :ref:`transit_virtual_path_builder`), pre-computes tap-to-tap total utilities and cache for later if desired


The main interface to the initialize LOS step is the :py:func:`~activitysim.abm.models.initialize_los.initialize_los`
function.  The main interface to the initialize TVPB step is the :py:func:`~activitysim.abm.models.initialize_los.initialize_tvpb`

function.  These functions are registered as Inject steps in the example Pipeline.



### Examples

- [Prototype MTC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mtc/configs/free_parking.yaml)
- [Prototype ARC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_arc/configs/free_parking.yaml)

## Implementation

```{eval-rst}
.. autofunction:: initialize_los
```
