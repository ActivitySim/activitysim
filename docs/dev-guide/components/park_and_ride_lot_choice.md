(component-park-and-ride-lot-choice)=
# Park-and-Ride Lot Choice

```{eval-rst}
.. currentmodule:: activitysim.abm.models.park_and_ride_lot_choice
```

The park-and-ride lot choice model selects a candidate parking zone for a tour,
so that mode choice can evaluate driving to that lot and taking transit to the
primary destination. Optional iteration with tour mode choice redistributes
excess demand at lots with limited capacity. This functionality was added in
[PR #1001](https://github.com/ActivitySim/activitysim/pull/1001).

A selected lot does not mean that the tour uses park-and-ride: tour mode choice
still selects the mode. Use both `tour_mode` and `pnr_zone_id` to identify actual
park-and-ride demand. Lot choice is distinct from
{ref}`component-parking-location-choice`, which selects parking for individual
trips near their destinations.

## Structure

- *Configuration File*: `park_and_ride_lot_choice.yaml`
- *Core Table*: `tours`, with chooser attributes from `tours_merged`
- *Alternatives*: `land_use` zones with positive park-and-ride spaces
- *Result Field*: `pnr_zone_id` on `tours`; `-1` indicates no lot choice
- *Chooser Keys*: `home_zone_id`, `destination`, `start`, `end`

The model evaluates a utility specification for each eligible chooser and lot
using interaction simulation. Lots are represented at the land-use zone level;
spaces at multiple facilities in the same zone must be aggregated. Lot zone IDs
must be compatible with the default skim dictionary.

## Configuring an Implementation

PR #1001 provides the model code and unit tests, but does not add a configured
agency example. The fragments below illustrate the configuration structure;
the land-use fields, skim names, utility expressions, coefficients, and mode
names must be supplied for the implementing model.

1. Add a spaces column to `land_use`, with positive values for lot zones and
   zero for zones without lots. Provide auto access and transit skims between
   the origin, lot, and destination.
2. Add `park_and_ride_lot_choice` to the `models` list in `settings.yaml` after
   the relevant tour destinations and schedules are available and before
   `tour_mode_choice_simulate`. Keep this step in the model list when enabling
   capacity iteration; multiprocessing also uses its presence to allocate
   shared capacity buffers.
3. Create `park_and_ride_lot_choice.yaml` and its specification and coefficient
   files. For example:

   ```yaml
   SPEC: park_and_ride_lot_choice.csv
   COEFFICIENTS: park_and_ride_lot_choice_coefficients.csv
   LANDUSE_PNR_SPACES_COLUMN: pnr_spaces
   LANDUSE_COL_FOR_PNR_ELIGIBLE_DEST: pnr_eligible_destination
   ITERATE_WITH_TOUR_MODE_CHOICE: false
   ```

   Here, `pnr_eligible_destination` is a boolean column in `land_use`. The model
   first filters destinations for transit eligibility, then runs the chooser
   `preprocessor`, applies the optional `CHOOSER_FILTER_EXPR` as a pandas query,
   and runs `alts_preprocessor` on lot alternatives.
4. Update the tour mode choice specification and preprocessors to use the
   selected lot and its access/transit skims. Make park-and-ride modes
   unavailable when `pnr_zone_id == -1`, and guard skim expressions against
   that sentinel. The lot-choice step does not impose mode availability.
5. Configure trip mode choice and matrix output consistently with the tour
   mode and selected lot. See {ref}`component-write-trip-matrices` for assigning
   the drive and transit portions to different origin/destination pairs.

Instead of a land-use eligibility column, set `TRANSIT_SKIMS_FOR_ELIGIBILITY`
to a list of skim names. A destination is eligible if at least one lot has a
positive value to it in any listed skim. Time-dependent skim names use the
form `TRANSIT_TIME__AM`. If both eligibility settings are supplied, the land-use
column takes precedence; if neither is supplied, all destinations are eligible.
This is a destination screen, so the utility specification must still handle
unavailable paths for individual lot/destination pairs. Filtered-out tours
receive `pnr_zone_id = -1`.

### Skims and Utility Expressions

The following wrappers are available in lot choice and, when `pnr_zone_id` is
present, primary tour mode choice. Here, origin is `home_zone_id`, lot is
`pnr_zone_id`, and destination is the tour's primary destination.

| Wrapper | From / to | Time period |
| --- | --- | --- |
| `olt_skims` | Origin to lot | Outbound (`start`) |
| `ldt_skims` | Lot to destination | Outbound (`start`) |
| `dlt_skims` | Destination to lot | Inbound (`end`) |
| `lot_skims` | Lot to origin | Inbound (`end`) |
| `ol_skims` | Origin to lot | Time independent |
| `ld_skims` | Lot to destination | Time independent |

For example, a lot-choice CSV specification can include
`@olt_skims['DRIVE_TIME']` and `@ldt_skims['TRANSIT_TIME']`, as well as the
return-leg equivalents, with the implementing model's skim names and
coefficients. The alternative attributes include `pnr_zone_id` and
`pnr_lot_full`. When iteration is disabled, `pnr_lot_full` is zero.

## Capacity Iteration

To enable capacity feedback, add the following to
`park_and_ride_lot_choice.yaml`, replacing `DRIVE_TRANSIT` with the applicable
mode names from the tour mode choice specification:

```yaml
ITERATE_WITH_TOUR_MODE_CHOICE: true
MAX_ITERATIONS: 5
PARK_AND_RIDE_MODES: [DRIVE_TRANSIT]
ACCEPTED_TOLERANCE: 0.95
RESAMPLE_STRATEGY: latest
TRACE_PNR_CAPACITIES_PER_ITERATION: true
```

The first mode-choice pass includes all primary tours. Capacity accounting then
counts only tours choosing a mode in `PARK_AND_RIDE_MODES`. Excess tours are
selected for a new lot choice and mode choice; other tours keep their results.
The final lot and mode choices are written back to `tours`.

Capacity is scaled to the simulated population as
`ceil(spaces * sample_rate)`, using the modal value of `households.sample_rate`.
Occupancy counts tours, not persons or time-dependent parking stays. The
implementation does not release spaces at tour end times. `latest` uses the
tour `start` to select excess tours in descending start-time order, with tour
ID breaking ties. `random` selects occupants with probability
`(occupancy - capacity) / occupancy`, so the number resampled equals the excess
in expectation.

`ACCEPTED_TOLERANCE` (default `0.95`) flags a lot as full when occupancy reaches
`ceil(scaled_capacity * ACCEPTED_TOLERANCE)`. Resampling addresses demand above
the full scaled capacity, not above the tolerance threshold. Flagged lots remain
in the alternatives with `pnr_lot_full = 1`: **the lot-choice specification must
use this flag to discourage or exclude them**. They are not automatically
removed. If all alternatives are flagged full, lot choice returns `-1` and the
mode-choice specification must allow a non-park-and-ride alternative.

`MAX_ITERATIONS` (default `5`) includes the initial mode-choice pass; use a
value greater than one to allow resampling. Iteration can stop earlier when no
choosers need resimulation. Capacity iteration is disabled during estimation.
Reaching the iteration limit does not guarantee that every lot meets capacity.

When `TRACE_PNR_CAPACITIES_PER_ITERATION` is true (the default), capacity checks
write `pnr_capacity_snapshot_i<N>` traces containing scaled capacity,
occupancy, percent utilized, excess demand, and the capacitated flag. Checks
occur before resampling, not after the final allowed mode-choice pass. Inspect
final tour choices as well as these traces when assessing remaining excess
demand.

## Logsums and At-Work Subtours

Set `include_pnr_for_logsums: true` in the tour mode settings used for logsum
calculations to select a lot for each logsum chooser OD pair and expose the
lot skims in the logsum utility calculations. This also supports disaggregate
accessibility calculations. These lot choices are uncapacitated and can add
runtime. Lot-choice preprocessors used here must tolerate tables such as
`tours` not existing yet; guard expressions that depend on those tables.

Set `run_atwork_pnr_lot_choice: true` in the settings used by
{ref}`component-atwork-subtour-mode-choice` to select lots for at-work subtours.
This uses `workplace_zone_id` as the origin and does not apply the primary-tour
capacity iteration. Both settings default to `false`.

## Configuration

```{eval-rst}
.. autopydantic_model:: ParkAndRideLotChoiceSettings
    :inherited-members: BaseModel, PydanticReadable
    :show-inheritance:
    :class-doc-from: init
```

## Implementation

```{eval-rst}
.. autofunction:: park_and_ride_lot_choice
.. autofunction:: run_park_and_ride_lot_choice
```
