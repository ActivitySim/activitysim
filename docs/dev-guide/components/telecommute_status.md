(component-telecommute_status)=
# Telecommute Status

```{eval-rst}
.. currentmodule:: activitysim.abm.models.telecommute_status
```

ActivitySim telecommute representation consists of two long term submodels - 
a person [work_from_home](work_from_home) model and 
a person [telecommute_frequency](telecommute_frequency) model. 
The work from home model predicts if a worker works exclusively from home, 
whereas the telecommute frequency model predicts number of days in a week a worker telecommutes, 
if they do not exclusively work from home. 
However, neither of them predicts whether a worker telecommutes or not on the simulation day. 
This telecommute status model extends the previous two models to predict for all workers whether 
they telecommute on the simulation day.

A simple implementation of the telecommute status model can be based on the worker's telecommute frequency.
For example, if a worker telecommutes 4 days a week, then there is a 80% probability for them 
to telecommute on the simulation day. 
The telecommute status model software can accommodate more complex model forms if needed.

There have been discussions about where to place the telecommute status model within the model sequence,
particularly regarding its interation with the Coordinated Daily Activity Pattern (CDAP) model.
Some have proposed expanding the CDAP definition of the "Mandatory" day pattern to include commuting, telecommuting and working from home, 
and then applying the telecommute status model to workers with a "Mandatory" day pattern.
While this idea had merit, it would require re-defining and re-estimating CDAP for many regions, which presents practical challenges.

During Phase 9B development, the Consortium collaboratively reached a consensus on a preferred design for explicitly modeling telecommuting. 
It was decided that the existing CDAP definitions would remain unchanged. The new design introduces the ability 
to model hybrid workers—those who work both in-home and out-of-home on the simulation day. To support this, 
the Consortium recommended adding two new models: a Telecommute Arrangement model and an In-Home Work Activity Duration model.

As of August 2025, these two models remain at the design stage and have not yet been implemented. Once deployed, 
they will supersede the current telecommute status model, which will no longer be needed. In the interim, 
the telecommute status model can be used to flag telecommuters in the simulation.

The main interface to the telecommute status model is the
[telecommute_status](activitysim.abm.models.telecommute_status) function. This
function is registered as an Inject step in the example Pipeline.

## Structure

- *Configuration File*: `telecommute_status.yaml`
- *Core Table*: `persons`
- *Result Table*: `is_telecommuting`


## Configuration

```{eval-rst}
.. autopydantic_model:: TelecommuteStatusSettings
    :inherited-members: BaseModel, PydanticReadable
    :show-inheritance:
```

### Examples

- [Example SANDAG ABM3](https://github.com/ActivitySim/sandag-abm3-example/tree/main/configs/resident/telecommute_status.yaml)


## Implementation

```{eval-rst}
.. autofunction:: telecommute_status
```
