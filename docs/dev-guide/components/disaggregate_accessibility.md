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
- *Variables*:
    - VARIABLES - The base variable, must be a value or a list. Results in the cartesian product (all non-repeating combinations) of the fields.
    - mapped_fields [optional] - For non-combinatorial fields, users can map a variable to the fields generated in VARIABLES (e.g., income category bins mapped to median dollar values).
    - filter_rows [optional] - Users can also filter rows using pandas expressions if specific variable combinations are not desired.
    - JOIN_ON [required only for PROTO_TOURS] - specify the persons variable to join the tours to (e.g., person_number).
  * MERGE_ON - User specified fields to merge the proto-population logsums onto the full synthetic population. The proto-population should be designed such that the logsums are able to be joined exactly on these variables specified to the full population. Users specify the to join on using:

    - by: An exact merge will be attempted using these discrete variables.
    - asof [optional]: The model can peform an "asof" join for continuous variables, which finds the nearest value. This method should not be necessary since synthetic populations are all discrete.

    - method [optional]: Optional join method can be "soft", default is None. For cases where a full inner join is not possible, a Naive Bayes clustering method is fast but discretely constrained method. The proto-population is treated as the "training data" to match the synthetic population value to the best possible proto-population candidate. The Some refinement may be necessary to make this procedure work.

  * annotate_proto_tables [optional] - Annotation configurations if users which to modify the proto-population beyond basic generation in the YAML.
  * DESTINATION_SAMPLE_SIZE - The *destination* sample size (0 = all zones), e.g., the number of destination zone alternatives sampled for calculating the destination logsum. Decimal values < 1 will be interpreted as a percentage, e.g., 0.5 = 50% sample.
  * ORIGIN_SAMPLE_SIZE - The *origin* sample size (0 = all zones), e.g., the number of origins where logsum is calculated. Origins without a logsum will draw from the nearest zone with a logsum. This parameter is useful for systems with a large number of zones with similar accessibility. Decimal values < 1 will be interpreted as a percentage, e.g., 0.5 = 50% sample.
  * ORIGIN_SAMPLE_METHOD - The method in which origins are sampled. Population weighted sampling can be TAZ-based or "TAZ-agnostic" using KMeans clustering. The potential advantage of KMeans is to provide a more geographically even spread of MAZs sampled that do not rely on TAZ hierarchies. Unweighted sampling is also possible using 'uniform' and 'uniform-taz'.

    - None [Default] - Sample zones weighted by population, ensuring at least one TAZ is sampled per MAZ. If n-samples > n-tazs then sample 1 MAZ from each TAZ until n-remaining-samples < n-tazs, then sample n-remaining-samples TAZs and sample an MAZ within each of those TAZs. If n-samples < n-tazs, then it proceeds to the above 'then' condition.

    - "kmeans" - K-Means clustering is performed on the zone centroids (must be provided as maz_centroids.csv), weighted by population. The clustering yields k XY coordinates weighted by zone population for n-samples = k-clusters specified. Once k new cluster centroids are found, these are then approximated into the nearest available zone centroid and used to calculate accessibilities on. By default, the k-means method is run on 10 different initial cluster seeds (n_init) using using [k-means++ seeding algorithm](https://en.wikipedia.org/wiki/K-means%2B%2B). The k-means method runs for max_iter iterations (default=300).

    - "uniform" - Unweighted sample of N zones independent of each other.

    - "uniform-taz" - Unweighted sample of 1 zone per taz up to the N samples specified.


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
