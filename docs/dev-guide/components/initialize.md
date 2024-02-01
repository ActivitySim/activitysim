(component-initialize)=
# Initialize

```{eval-rst}
.. currentmodule:: activitysim.abm.models.initialize
```

The initialize model isn't really a model, but rather a few data processing steps in the data pipeline.
The initialize data processing steps code variables used in downstream models, such as household and person
value-of-time.  This step also pre-loads the land_use, households, persons, and person_windows tables because
random seeds are set differently for each step and therefore the sampling of households depends on which step
they are initially loaded in.

The main interface to the initialize land use step is the [initialize_landuse](activitysim.abm.models.initialize.initialize_landuse)
function. The main interface to the initialize household step is the [initialize_households](activitysim.abm.models.initialize.initialize_households)
function.  The main interface to the initialize tours step is the [initialize_tours](activitysim.abm.models.initialize_tours.initialize_tours)
function.  These functions are registered as Inject steps in the example Pipeline.


## Implementation

```{eval-rst}
.. autofunction:: initialize_landuse
.. autofunction:: initialize_households
```
