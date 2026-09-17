(component-disaggregate-accessibility)=
# Disaggregate Accessibility

```{eval-rst}
.. currentmodule:: activitysim.abm.models.disaggregate_accessibility
```

The disaggregate accessibility model is an extension of the base accessibility model.
While the base accessibility model is based on a mode-specific decay function and uses fixed market
segments in the population (i.e., income), the disaggregate accessibility model extracts the actual
destination choice logsums by purpose (i.e., mandatory fixed school/work location and non-mandatory
tour destinations by purpose) from the actual model calculations using a user-defined proto-population.
This enables users to include features that may be more critical to destination
choice than just income (e.g., automobile ownership).

## Structure

*Inputs*
  * disaggregate_accessibility.yaml - Configuration settings for disaggregate accessibility model.
  * annotate.csv [optional] - Users can specify additional annotations specific to disaggregate accessibility. For example, annotating the proto-population tables.

*Outputs*
  * final_disaggregate_accessibility.csv [optional]
  * final_non_mandatory_tour_destination_accesibility.csv [optional]
  * final_workplace_location_accessibility.csv [optional]
  * final_school_location_accessibility.csv [optional]
  * final_proto_persons.csv [optional]
  * final_proto_households.csv [optional]
  * final_proto_tours.csv [optional]

The above tables are created in the model pipeline, but the model will not save
any outputs unless specified in settings.yaml - output_tables. Users can return
the proto population tables for inspection, as well as the raw logsum accessibilities
for mandatory school/work and non-mandatory destinations. The logsums are then merged
at the household level in final_disaggregate_accessibility.csv, which each tour purpose
logsums shown as separate columns.

*Usage*

The disaggregate accessibility model is run as a model step in the model list.
There are two necessary steps:

* `initialize_proto_population`
* `compute_disaggregate_accessibility`

The reason the steps must be separate is to enable multiprocessing.
The proto-population must be fully generated and initialized before activitysim
slices the tables into separate threads. These steps must also occur before
initialize_households in order to avoid conflict with the shadow_pricing model.

The model steps can be run either as part the activitysim model run, or setup
to run as a standalone run to pre-computing the accessibility values.
For standalone implementations, the final_disaggregate_accessibility.csv is read
into the pipeline and initialized with the initialize_household model step.

- *Configuration File*: `disaggregate_accessibility.yaml`
- *Core Table*:  Users define the variables to be generated for 'PROTO_HOUSEHOLDS', 'PROTO_PERSONS', and 'PROTO_TOURS' tables. These tables must include all basic fields necessary for running the actual model. Additional fields can be annotated in pre-processing using the annotation settings of this file.


## Configuration

```{eval-rst}
.. autopydantic_model::  DisaggregateAccessibilitySettings
    :inherited-members: BaseModel, PydanticReadable
    :show-inheritance:
```

### Examples

- [Prototype MTC_Extended](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mtc_extended/configs/disaggregate_accessibility.yaml)
- [Placeholder_SANDAG_2_Zone](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/placeholder_sandag/test/configs_2_zone/disaggregate_accessibility.yaml)

## Implementation

```{eval-rst}
.. autofunction:: disaggregate_accessibility
```
