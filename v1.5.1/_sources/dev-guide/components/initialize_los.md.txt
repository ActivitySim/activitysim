(component-initialize-los)=
# Initialize LOS

```{eval-rst}
.. currentmodule:: activitysim.abm.models.initialize_los
```

The initialize LOS model isn't really a model, but rather a series of data processing steps in the data pipeline.
The initialize LOS model does two things:

  * Loads skims and cache for later if desired
  * Loads network LOS inputs for transit virtual path building (see [transit_virtual_path_builder](transit_virtual_path_builder), pre-computes tap-to-tap total utilities and cache for later if desired

The main interface to the initialize LOS step is the [initialize_los](activitysim.abm.models.initialize_los.initialize_los)
function.  The main interface to the initialize TVPB step is the [initialize_tvpb](activitysim.abm.models.initialize_los.initialize_tvpb)
function.  These functions are registered as Inject steps in the example Pipeline.


## Implementation

```{eval-rst}
.. autofunction:: initialize_los
.. autofunction:: compute_utilities_for_attribute_tuple
.. autofunction:: initialize_tvpb
```
