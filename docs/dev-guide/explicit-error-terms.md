(explicit-error-terms-dev)=
# Explicit Error Terms

Explicit Error Terms (EET) is an alternative way to simulate choices from ActivitySim's
logit models. It keeps the same systematic utilities and the same random-utility
interpretation as the standard method, but changes how the final simulated choice is
drawn. For details, see
[this ATRF paper](https://australasiantransportresearchforum.org.au/frozen-randomness-at-the-individual-utility-level/).

For user-facing guidance on when to use EET, see {ref}`explicit_error_terms_ways_to_run`.

## Enabling EET

Enable EET globally in `settings.yaml`:

```yaml
use_explicit_error_terms: True
```

The top-level switch is defined in
`activitysim.core.configuration.top.SimulationSettings.use_explicit_error_terms`.
Choice simulation code reads that setting through the model compute settings and routes
supported logit simulations through the EET path.

## Default Draw Versus EET

Under the default ActivitySim simulation path, choice drawing works like this:

1. Compute systematic utilities.
2. Convert those utilities into analytical probabilities.
3. Draw one uniform random number per chooser.
4. Select the alternative whose cumulative probability interval contains that draw.

With EET enabled, the final draw step changes:

1. Compute systematic utilities.
2. Draw error terms for each chooser-alternative pair.
3. Add those error terms to the systematic utilities.
4. Choose the alternative with the highest total utility.

For multinomial logit, the error term distribution is i.i.d. Gumbel and draws are generated
by inverting the cumulative density function. For nested logit, this method is not available
due to correlations between error terms. Instead, ActivitySim makes use of recent advances
regarding the [representation of nested logit models](https://doi.org/10.1017/S026646662000047X)
and combines this with
[exact numerical sampling methods](https://doi.org/10.1007/978-3-030-52915-4)
to draw error terms of all fundamental alternatives (leafs).

## Practical Effects

### Comparisons and Simulation Noise

For EET to reduce simulation noise, it is important that alternatives of a choice situation
keep the same unobserved error term in different scenario runs. This is intimately tied
to how random numbers are generated; see {ref}`random_in_detail` for the underlying
random-number stream design and the `activitysim.core.random` API.
Because unchanged alternatives can keep the same unobserved draws, changes to choices between
scenarios can only happen when the observed utility of an alternative increases. This is not
the case for the Monte Carlo simulation method, where the draws are based on probabilities,
which necessarily change for all alternatives if any observed utility changes.

This also means that it is advisable to use the same setting in all runs. Comparing a baseline
run with EET to a scenario run without EET mixes two simulation methods and can make differences
harder to interpret. Aggregate choice patterns should remain statistically the same
as for the default probability-based method. The project test suite includes parity tests for
MNL, NL, and interaction-based simulations.

### Numerical and Debugging Behavior

EET changes the final simulation step, not the utility calculation itself. Utility
expressions, availability logic, nesting structure, and utility validation still matter in
the same way as in the default method.

In practice, EET can make some comparisons easier to interpret because the selected
alternative is the one with the highest total utility after adding the explicit error term,
rather than the one reached by a cumulative-probability threshold. That can reduce
sensitivity to small differences in the final CDF draw when comparing nearby scenarios.
It does not eliminate the need to inspect invalid or unavailable alternatives, and it does
not guarantee identical results across different RNG seeds or different model
configurations.

For shadow-priced location choice, ActivitySim resets RNG offsets between iterations when
EET is enabled so each shadow-pricing iteration uses the same sequence of random numbers.
That keeps the comparison across iterations focused on the shadow price updates instead of
changing random draws between iterations.

### Runtime

Runtime differs between the methods. EET generates one EV1 error term per chooser-alternative
pair, while the default Monte Carlo path draws only one uniform random number per chooser after
probabilities are computed. EET, however, does not need to compute probabilities to make choices.

Exact runtimes depend on the number of alternatives, nesting structure, interaction size, and
sampling configuration. With default settings, current full-scale demand model runs with EET
are about 100% higher than the default MC method. While the relative runtime increase
of nested logit models is large, these typically contribute only a very small fraction to the
overall runtime and virtually all of the increase is due to sampling in location choice. To
avoid this penalty, it is possible to use MC for sampling only by adding the following to each
model setting where sampling is used (currently all location and destination choice models as
well as disaggregate accessibilities):

```yaml
compute_settings:
  use_explicit_error_terms:
    sample: false
```

With this setting, model runtimes should be roughly equal. The influence of this change on
sampling noise is under investigation.

(explicit_error_terms_zone_encoding)=
#### Zone ID encoding and runtime

For location choice models, encoding zone IDs as a 0-based contiguous index reduces EET runtime
and memory use during sampling.

The current implementation draws error terms into a dense 1-D array of length `max_zone_id + 1`
per chooser (see `AltsContext.n_alts_to_cover_max_id` in `activitysim.core.logit`). Each sampled
alternative is then looked up by direct offset into that array, so the same zone always receives
the same error term regardless of which alternatives are in the sampled choice set — a property
needed for consistent scenario comparisons.

When zone IDs are a contiguous 0-based sequence, the dense array has exactly as many entries as
there are zones and every draw is used. When zone IDs contain gaps or start from a large value,
the array must still cover `max_zone_id + 1` entries, so the draws for the missing IDs are
generated but never used. For zone systems with large or sparse IDs, this waste can be substantial.

An alternative would be to draw only as many error terms as there are sampled alternatives and
retrieve the relevant term for each zone via a lookup. That would avoid unused draws but adds an
index-mapping step for every chooser-sample in the interaction frame, trading one form of overhead
for another. The current design favours the dense approach because the direct-offset indexing is
simpler and because the ``recode_columns`` setting can encode zone IDs as ``zero-based`` in
the input table list; see the
[Zero-based Recoding of Zones](using-sharrow.md#zero-based-recoding-of-zones) section for details.

(explicit_error_terms_memory)=
### Memory usage

When running EET with MC for location sampling as described in the Runtime section above,
there should be only a small increase in memory usage for location choice models compared to full
MC simulation.

However, when EET is run with its current default location sampling settings, an array of size
(number of choosers, number of alternatives, number of samples) is allocated for all random error
terms. This can quickly become unwieldy for machines with limited memory, and
[chunking](../users-guide/performance/chunking.md) will likely be needed.

When chunking is needed and [explicit chunking](../users-guide/performance/chunking.md#explicit-chunking)
is used, using fractional values for the chunk size rather than absolute numbers of choosers is
often a better fit. This is because the individual steps of location choice models
(location sampling, location logsums, and location choice from the sampled choice set) all have
very different chooser characteristics, but the chunk size currently can only be set at the model
level. Using absolute values for the explicit chunk size would lead to a large number of chunks
for the logsum calculations, which is relatively slow.


## Implementation Details and Adding New Models

The core simulation is implemented in `activitysim.core.logit.make_choices_utility_based`. Most
calls to this function are wrapped in one of the following methods:

- `activitysim.core.simulate`
- `activitysim.core.interaction_simulate`
- `activitysim.core.interaction_sample`
- `activitysim.core.interaction_sample_simulate`

These wrappers all implement EET consistently, so any model using them will automatically support
EET. Some models call the underlying choice simulation method
`activitysim.core.logit.make_choices` directly. For EET to work in that case, the developer must
add a corresponding call to `logit.make_choices_utility_based`; see for example
`activitysim.abm.models.utils.cdap.household_activity_choices`. Models that draw directly
from probability distributions, such as `activitysim.abm.models.utils.cdap.extra_hh_member_choices`,
do not have a corresponding EET implementation because there are no utilities to work with.


### Unavailable choices utility convention

For EET, only utility differences matter, and therefore the outcome for two utilities that are
very small, say -10000 and -10001, is identical to the outcome for 0 and 1. For MC, utilities
have to be exponentiated and therefore floating point precision dictates the smallest and largest
utility that can be used in practice. ActivitySim models historically often use a utility of
-999 to make alternatives practically unavailable. That value is below the utility threshold
used in the probability-based path, which is about -691 because ActivitySim clips
exponentiated utilities at 1e-300. To keep behavior consistent, EET treats alternatives with
utilities at or below that threshold as unavailable; see `activitysim.core.logit.validate_utils`.

### Scale of the distribution
Error terms are drawn from standard Gumbel distributions, i.e., the scale of the error term is
fixed to one.
