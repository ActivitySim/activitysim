(component-joint-tour-participation)=
# Joint Tour Participation

```{eval-rst}
.. currentmodule:: activitysim.abm.models.joint_tour_participation
```
In the joint tour person participation model, each eligible person sequentially makes a
choice to participate or not participate in each joint tour.  Since the party composition model
determines what types of people are eligible to join a given tour, the person participation model
can operate in an iterative fashion, with each household member choosing to join or not to join
a travel party independent of the decisions of other household members. In the event that the
constraints posed by the result of the party composition model are not met, the person
participation model cycles through the household members multiple times until the required
types of people have joined the travel party.

This step also creates the ``joint_tour_participants`` table in the pipeline, which stores the
person ids for each person on the tour.

The main interface to the joint tour participation model is the
[joint_tour_participation](activitysim.abm.models.joint_tour_participation.joint_tour_participation)
function.  This function is registered as an Inject step in the example Pipeline.

## Structure

- *Configuration File*: `joint_tour_participation.yaml`
- *Core Table*: `tours`
- *Result Field*: `number_of_participants, person_id (for the point person)`


## Configuration

```{eval-rst}
.. autopydantic_model:: JointTourParticipationSettings
    :inherited-members: BaseModel, PydanticReadable
    :show-inheritance:
```

### Examples

- [Prototype MTC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mtc/configs/joint_tour_participation.yaml)
- [Prototype ARC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_arc/configs/joint_tour_participation.yaml)

## Implementation

```{eval-rst}
.. autofunction:: joint_tour_participation
.. autofunction:: participants_chooser
```
