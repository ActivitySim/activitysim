(component-shadow-pricing)=
# Shadow Pricing

```{eval-rst}
.. currentmodule:: activitysim.abm.tables.shadow_pricing
```

The shadow pricing calculator used by work and school location choice.

## Structure
- *Configuration File*: `shadow_pricing.yaml`

### Turning on and saving shadow prices

Shadow pricing is activated by setting the `use_shadow_pricing` to True in the settings.yaml file.
Once this setting has been activated, ActivitySim will search for shadow pricing configuration in
the shadow_pricing.yaml file. When shadow pricing is activated, the shadow pricing outputs will be
exported by the tracing engine. As a result, the shadow pricing output files will be prepended with
`trace` followed by the iteration number the results represent. For example, the shadow pricing
outputs for iteration 3 of the school location model will be called
`trace.shadow_price_school_shadow_prices_3.csv`.

In total, ActivitySim generates three types of output files for each model with shadow pricing:

- `trace.shadow_price_<model>_desired_size.csv` The size terms by zone that the ctramp and daysim
  methods are attempting to target. These equal the size term columns in the land use data
  multiplied by size term coefficients.

- `trace.shadow_price_<model>_modeled_size_<iteration>.csv` These are the modeled size terms after
  the iteration of shadow pricing identified by the <iteration> number. In other words, these are
  the predicted choices by zone and segment for the model after the iteration completes. (Not
  applicable for ``simulation`` option.)

- `trace.shadow_price_<model>_shadow_prices_<iteration>.csv` The actual shadow price for each zone
  and segment after the <iteration> of shadow pricing. This is the file that can be used to warm
  start the shadow pricing mechanism in ActivitySim. (Not applicable for `simulation` option.)

There are three shadow pricing methods in activitysim: `ctramp`, `daysim`, and `simulation`.
The first two methods try to match model output with workplace/school location model size terms,
while the last method matches model output with actual employment/enrollmment data.

The simulation approach operates the following steps.  First, every worker / student will be
assigned without shadow prices applied. The modeled share and the target share for each zone are
compared. If the zone is overassigned, a sample of people from the over-assigned zones will be
selected for re-simulation.  Shadow prices are set to -999 for the next iteration for overassigned
zones which removes the zone from the set of alternatives in the next iteration. The sampled people
will then be forced to choose from one of the under-assigned zones that still have the initial
shadow price of 0. (In this approach, the shadow price variable is really just a switch turning that
zone on or off for selection in the subsequent iterations. For this reason, warm-start functionality
for this approach is not applicable.)  This process repeats until the overall convergence criteria
is met or the maximum number of allowed iterations is reached.

Because the simulation approach only re-simulates workers / students who were over-assigned in the
previous iteration, run time is significantly less (~90%) than the CTRAMP or DaySim approaches which
re-simulate all workers and students at each iteration.

## Configuration

```{eval-rst}
.. autopydantic_model:: ShadowPriceSettings
    :inherited-members: BaseModel, PydanticReadable
    :show-inheritance:
```

### Examples

- [Prototype MTC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mtc/configs/free_parking.yaml)
- [Prototype ARC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_arc/configs/free_parking.yaml)

## Implementation

```{eval-rst}
.. autofunction:: ShadowPriceCalculator
.. autofunction:: buffers_for_shadow_pricing
.. autofunction:: buffers_for_shadow_pricing_choice
.. autofunction:: shadow_price_data_from_buffers_choice
.. autofunction:: shadow_price_data_from_buffers
.. autofunction:: load_shadow_price_calculator
.. autofunction:: add_size_tables
.. autofunction:: get_shadow_pricing_info
.. autofunction:: get_shadow_pricing_choice_info

```
