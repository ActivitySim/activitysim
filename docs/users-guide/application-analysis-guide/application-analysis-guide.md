# ActivitySim Application and Analysis Guide

This guide is to demonstrate to modelers how to apply ActivitySim for the analysis of various projects. It is intended to both demonstrate exactly how to change the inputs to test a particular project or policy, process the outputs to best answer the question that was asked, and provide a general understanding of what ActivitySim can and can't do.

There are presently three example scenarios using the [SANDAG ABM3 Example model](https://github.com/activitysim/sandag-abm3-example), though more may be added in the future. Before running all of them, it is recommended to run the SANDAG example, as that will download the full data and provide a baseline run to compare each scenario to. For each example scenario, a step-by-step guide for changing the inputs along with notebooks demonstrating how to calculate key metrics from the model outputs.

## General Configuration

Every ActivitySim model component has a settings file, that is typically hardcoded in the model code as being the component name followed by .yaml. For example, the workplace location model is configured with the `workplace_location.yaml` file. Within these files there are several key settings for each model component, which can include file definitions. Further filenames are defined in this file with the following settings:
- `SPEC`: The spec file defines how the utilities are calculated for each model alternative.
- `COEFFICIENTS`: The coefficients file defines coefficient files that are references in the spec file.
- `COEFFICIENTS TEMPLATE`: For models that are segmented (such as how trip mode choice is segmented by purpose), this defines how to name each coefficient for each segment. As an example from the SANDAG ABM3 Example, the trip mode choice spec file refers to [`coef_ivt`](https://github.com/ActivitySim/sandag-abm3-example/tree/main/configs/resident) in multiple places, but if one were to look at the coefficient file they would see a different value of `coef_ivt` [for each purpose](https://github.com/ActivitySim/sandag-abm3-example/blob/main/configs/resident/trip_mode_choice_coefficients.csv#L17-L22). The association between `coef_ivt` and `coef_ivt_[purpose]` is defined in the [coefficient template file](https://github.com/ActivitySim/sandag-abm3-example/blob/main/configs/resident/trip_mode_choice_coefficients_template.csv#L15).

The user can also define preprocessor to add fields to a choosers table before a component is run or annotators to add that afterward. The fields added in a preprocessor can then be used in utility calculations to simplify the calculations and reduce the total amount of memory usage. They can be added by specifying `preprocessor` or `annotator` in the settings (there are other options as well), with the following additional options:
- `SPEC`: The name of the file (.csv extension does not need to be included)
- `DF`: The table in memory that the fields will be added to. This can be set to `choosers` to be the same model that the component will be run on.
- `TABLES`: Tables to include in memory in case data from other tables are to be added to `DF`.

The files that define these are CSV files with three fields:
- `Description`: A description of the field being added. This is only for the users and is not read by ActivitySim.
- `Target`: The name of the field to be added.
- `Expression`: An expression defining how to calculate the field.
In the case where an output variable name is hardcoded (such as the outputs of the [trip mode choice](https://github.com/ActivitySim/activitysim/blob/main/activitysim/abm/models/trip_mode_choice.py#L81) model), one can use an annotator to get around that by creating a variable on the same table with a custom name by just setting that field to be equivalent to the hardcoded field via the expression. For example, after running trip mode choice, if one were to create a field called `modeTrip` that was a copy of the hardcoded `trip_mode` field, they would need to have a line in the annotator that's run after trip mode choice as follows:
| Description                  | Target   | Expression |
| ---------------------------- | -------- | ---------- |
| New name for trip mode field | modeTrip | trip_mode  |

There are further settings regarding the structure of any logit model that's used. One setting, `LOGIT_TYPE` will be `MNL` for multinomial logit and `NL` for nested logit. If a nested logit structure is used, then there needs to be a setting called `NESTS` that define how each alternative nests. Further, there is an additional setting called `CONSTANTS` that contains constants that can be refered to in the model component's preprocessor and spec files. If a variable is used in multiple components, then it should be specified in the global settings file `constants.yaml`, which contains constants that can be accessed by all model components.

## Example scenarios
[Land Use Change](land-use-change\land_use_change.md)
[Network Change](network-change\network_change.md)
[Telecommuting Change](telecommute-change\telecommute_change.md)