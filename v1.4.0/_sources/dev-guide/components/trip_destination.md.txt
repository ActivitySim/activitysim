(component-trip-destination)=
# Trip Destination

```{eval-rst}
.. currentmodule:: activitysim.abm.models.trip_destination
```

The trip location choice model predicts the location of trips (or stops) made
along the tour other than the primary destination. Final trips already have a
destination (the primary tour destination for outbound trips, and home for
inbound trips) so no choice is needed in those cases.

## Structure

- *Configuration File*: `trip_destination.yaml`
- *Core Table*: `trips`
- *Result Field*: `destination`

This model is structured as
a multinomial logit model using a zone attraction size variable
and route deviation measure as impedance. The alternatives are sampled
from the full set of zones, subject to availability of a zonal attraction size
term. The sampling mechanism is also based on accessibility between tour origin
and primary destination, and is subject to certain rules based on tour mode.

All destinations are available for auto tour modes, so long as there is a positive
size term for the zone. Intermediate stops on walk tours must be within X miles of both the tour
origin and primary destination zones. Intermediate stops on bike tours must be within X miles of both
the tour origin and primary destination zones. Intermediate stops on walk-transit tours must either be
within X miles walking distance of both the tour origin and primary destination, or have transit access to
both the tour origin and primary destination. Additionally, only short and long walk zones are
available destinations on walk-transit tours.

The intermediate stop location choice model works by cycling through stops on tours. The level-of-service
variables (including mode choice logsums) are calculated as the additional utility between the
last location and the next known location on the tour. For example, the LOS variable for the first stop
on the outbound direction of the tour is based on additional impedance between the tour origin and the
tour primary destination. The LOS variable for the next outbound stop is based on the additional
impedance between the previous stop and the tour primary destination. Stops on return tour legs work
similarly, except that the location of the first stop is a function of the additional impedance between the
tour primary destination and the tour origin. The next stop location is based on the additional
impedance between the first stop on the return leg and the tour origin, and so on.

Trip location choice for [two- and three-zone](multiple_zone_systems) models
uses [presampling](presampling) by default.

The main interface to the trip destination choice model is the
[trip_destination](activitysim.abm.models.trip_destination.trip_destination) function.
This function is registered as an Inject step in the example Pipeline.
See [writing_logsums](writing_logsums) for how to write logsums for estimation.

```{note}
Trip purpose and trip destination choice can be run iteratively together via
[trip_purpose_and_destination](trip_purpose_and_destination_model).
```

## Configuration

```{eval-rst}
.. autopydantic_model:: TripDestinationSettings
```

### Examples

- [Prototype MTC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mtc/configs/trip_destination.yaml)
- [Prototype ARC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_arc/configs/trip_destination.yaml)

## Implementation

```{eval-rst}
.. autofunction:: trip_destination
```

## Context Variables

The following variables are made available for use in this component's utility
specifications:

- `network_los` ([Network_LOS](activitysim.core.los.Network_LOS)): A reference
    to the main network_los object for this model.
- `size_terms` ([DataFrameMatrix](activitysim.core.skim_dictionary.DataFrameMatrix)):
    This DataFrameMatrix offers a specialized two-dimension lookup wrapper
    on an array, to access the size terms by alternative zone and trip
    purpose by label simultaneously.  When sharrow is enabled for this model,
    this value also embeds a special linked array accessible as
    `size_terms['sizearray']`, which automatically compiles to point to the
    alternative destination zone and the 'purpose_index_num' (an integer
    offset value instead of a label, which should be set in a preprocessor).
- `size_terms_array` (numpy.ndarray):  A numpy array of size term values,
    with rows for zones and columns for trip purposes.  This is just the
    raw data underneath the `size_terms` value above.
- `timeframe` (str): Contains the constant value "trip".
- `odt_skims` ([Skim3dWrapper](activitysim.core.skim_dictionary.Skim3dWrapper)
    or [DatasetWrapper](activitysim.core.skim_dataset.DatasetWrapper)): Skims
    giving the LOS characteristics from the fixed trip origin to each
    alternative destination, by time period.
- `dot_skims` ([Skim3dWrapper](activitysim.core.skim_dictionary.Skim3dWrapper)
    or [DatasetWrapper](activitysim.core.skim_dataset.DatasetWrapper)): Skims giving the LOS
    characteristics from each alternative destination backwards to the
    fixed trip origin, by time period.
- `dpt_skims` ([Skim3dWrapper](activitysim.core.skim_dictionary.Skim3dWrapper)
    or [DatasetWrapper](activitysim.core.skim_dataset.DatasetWrapper)): Skims giving the LOS
    characteristics from each alternative destination to the tour final
    destination, by time period.
- `pdt_skims` ([Skim3dWrapper](activitysim.core.skim_dictionary.Skim3dWrapper)
    or [DatasetWrapper](activitysim.core.skim_dataset.DatasetWrapper)): Skims giving the LOS
    characteristics backwards to each alternative destination from the tour
    final destination, by time period.
- `od_skims` ([SkimWrapper](activitysim.core.skim_dictionary.SkimWrapper)
    or [DatasetWrapper](activitysim.core.skim_dataset.DatasetWrapper)): Skims giving the LOS
    characteristics from the fixed trip origin to each alternative
    destination.
- `dpt_skims` ([SkimWrapper](activitysim.core.skim_dictionary.SkimWrapper)
    or [DatasetWrapper](activitysim.core.skim_dataset.DatasetWrapper)): Skims giving the LOS
    characteristics from each alternative destination to the tour final
    destination.

The following TransitVirtualPathLogsumWrapper values are also available,
only for 3-zone models:

- `tvpb_logsum_odt`
- `tvpb_logsum_dot`
- `tvpb_logsum_dpt`
- `tvpb_logsum_pdt`

## Additional Related Functions

```{eval-rst}
.. autofunction:: trip_destination_sample
.. autofunction:: compute_logsums
.. autofunction:: compute_ood_logsums
.. autofunction:: trip_destination_simulate
```
