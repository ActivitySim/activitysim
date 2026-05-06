(explicit-error-terms-dev)=
# Explicit Error Terms

Explicit Error Terms (EET) is an alternative way to simulate choices from ActivitySim's
logit models. It keeps the same systematic utilities and the same random-utility
interpretation as the standard method, but changes how the final simulated choice is
drawn. For details, see
[this ATRF paper](https://australasiantransportresearchforum.org.au/frozen-randomness-at-the-individual-utility-level/).

For user-facing guidance, see {ref}`explicit_error_terms_ways_to_run`.

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
in the [representation of nested logit models](https://doi.org/10.1017/S026646662000047X)
and combines this with
[exact numerical sampling methods](https://doi.org/10.1007/978-3-030-52915-4)
to draw error terms of all fundamental alternatives.

## Practical Effects

### Comparisons and Simulation Noise

For EET to reduce simulation noise, it is important that alternatives of a choice situation
keep the same unobserved error term in different scenario runs. This is intimately tied
to how random numbers are generated; see {ref}`random_in_detail` for the underlying
random-number stream design and the `activitysim.core.random` API. In essence, keeping the
global random number generator seed constant for comparison runs is essential. This also means
that it is advisable to use the same setting in all runs. Comparing a baseline
run with EET to a scenario run without EET mixes two simulation methods and can make differences
harder to interpret. Aggregate choice patterns should remain statistically the same
as for the default probability-based method.

Because unchanged alternatives can keep the same unobserved draws, changes to choices between
scenarios can only happen when the observed utility of an alternative increases. This is not
the case for the Monte Carlo simulation method, where the draws are based on probabilities,
which necessarily change for all alternatives if any observed utility changes. This combined
with sensitivity to small differences in the final CDF draw when comparing nearby scenarios
means that EET is a good candidate to remove noise from scenario comparisons.


#### EET as a variance reduction method
TODO: expand on this here.

Common random numbers. Stronger correlations for exptectation values of differences -> less
variance in the estimator. So we need less model runs to be representative.


### Runtime

Runtime differs between the methods. EET generates one EV1 error term per chooser-alternative
pair, while the default Monte Carlo path draws only one uniform random number per chooser after
probabilities are computed. EET, however, does not need to compute probabilities to make choices.

Exact runtimes depend on the number of alternatives, nesting structure, interaction size, and
sampling configuration. With default settings, current full-scale demand model runs with EET
are about
%%% TODO: REVIEW THIS AND UPDATE
100% higher than the default MC method. Most of this is due to sampling in location choice.
%%% END TODO: REVIEW THIS AND UPDATE

Memory usage should be comparable albeit slightly higher when running with EET.

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
retrieve the relevant term for each zone via a lookup. That would avoid unused draws but it does
not fit naturally with with ActivitySim's current random number generation machinery, trading
one form of overhead for another. The current design favours the dense approach because
benchmarking suggested it was quicker and because ActivitySim has a ``recode_columns`` setting
that optionally encodes zone IDs as ``zero-based`` in the input table list; see the
[Zero-based Recoding of Zones](using-sharrow.md#zero-based-recoding-of-zones) section for details.
We recommend using this option when running with EET.

#### Sampling method considerations for model components with sampled choice sets
ActivitySim uses sampling of alternatives to reduce runtime of location choice methods.

TODO: Add details here once sampling method discussion have been resolved. Make clear this is
independent of overall simulation strategy. Maybe rename EET sampling so there is no confusion?

To use MC sampling with EET simulation, add the following lines to the settings of all models
where location choice sampling is used (currently all location and destination choice models as
well as disaggregate accessibilities):

.. code-block:: yaml

  compute_settings:
    use_explicit_error_terms:
      sample: false



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
MNL error terms are drawn from standard Gumbel distributions, i.e., the scale of the error term is
fixed to one.
