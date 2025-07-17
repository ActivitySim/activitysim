(component-telecommute_status)=
# Telecommute Status

```{eval-rst}
.. currentmodule:: activitysim.abm.models.telecommute_status
```

ActivitySim telecommute representation consists of two long term submodels - 
a person [work_from_home](work_from_home) model and 
a person [telecommute_frequency](telecommute_frequency) model. 
The work from home model predicts if a worker works exclusively from home, 
whereas the telecommute frequency model predicts number of days in a week a worker telecommuters, 
if they do not exclusively work from home. 
However, neither of them predicts whether a worker telecommutes or not on the simulation day. 
This telecommute status model extends the previous two models to predict for all workers whether 
they telecommute on the simulation day.

A simple implementation of the telecommute status model can be based on the worker's telecommute frequency.
For example, if a worker telecommutes 4 days a week, then there is a 80% probability for them 
to telecommute on the simulation day. 
The telecommute status model software can accommodate more complex model forms if needed.

There have been discussions on exactly where the telecommute status model should be added
in the model sequence. Some suggest it should be applied to all workers before the CDAP model; 
some suggest it should be applied after the CDAP model only to workers who have work activities 
during the day regardless of in-home or out-of-home (which requires change in CDAP definition). 
The Consortium is currently engaged in an explicit telecommute design task as part of Phase 9B,
out of which more guidance on the model sequence will be established.

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
