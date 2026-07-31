(explicit-error-terms-dev)=
# Explicit Error Terms

Explicit Error Terms (EET) is an alternative way to simulate choices from ActivitySim's
logit models. It keeps the same systematic utilities and the same random-utility
interpretation as the standard method, but changes how the final simulated choice is
drawn. For details, see
[this ATRF paper](https://australasiantransportresearchforum.org.au/frozen-randomness-at-the-individual-utility-level/).

<!-- For user-facing guidance, see {ref}`explicit_error_terms_ways_to_run`. -->

## Enabling EET

Enable EET globally in `settings.yaml`:

```yaml
use_explicit_error_terms: True
```

The top-level switch is defined in
`activitysim.core.configuration.top.SimulationSettings.use_explicit_error_terms`.
Choice simulation code reads that setting through the supported logit wrappers and routes
final choice simulation through the EET path. For interaction-sample-specific sampling
configuration, see {doc}`/dev-guide/sampling-methods`.

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
keep the same unobserved error term in different scenario runs. If unchanged alternatives
keep the same unobserved draws, changes to choices between scenarios can only happen when
the observed utility of an alternative increases. This is not the case for the Monte Carlo
simulation method, where the draws are based on probabilities, which necessarily change for
all alternatives if any observed utility changes. This combined with sensitivity to small
differences in the final CDF draw when comparing nearby scenarios means that EET removes
noise from scenario comparisons.

Note that the both MC and EET are simulating the same model, so individual runs with identical
inputs but varying global seed will lead to the same statistical results for individual
output metrics. EET's properties become apparent when comparing two model runs with different
inputs. Because error terms are aligned, the variance of the estimator of the indicator, e.g.,
mode choice shift or VMT difference, is reduced. In other words, difference metrics are more
precise estimators under EET.

In mathematical terms, for any two metrics $X$ (baseline) and $Y$ (scenario), the variance
of the difference $X - Y$ is

$$\text{Var}(X - Y) = \text{Var}(X) + \text{Var}(Y) - 2\,\text{Cov}(X, Y)$$

EET deliberately drives $\text{Cov}(X, Y)$ up by aligning error terms, so $\text{Var}(X-Y)$
collapses even though $\text{Var}(X)$ and $\text{Var}(Y)$ individually are unchanged.

In practice, models are often run once for each scenario. EET is still usefull because the
lower the noise of the estimator, the higher the chance that a single run is representative.
In other words, the noise level of comparison metrics is lower. Additionally, under MC small
but real benefits can show up as negative in a single run. Under EET, the sign of the effect
is far more trustworthy.

Independent of any statistical argument, under EET, choice changes between two runs are
attributable to utility changes which can be helpful for model development, sensitivity
testing, and presenting results to stakeholders.

### Aligning error terms

Aligning error terms between runs is essential. This is intimately tied
to how random numbers are generated; see {ref}`random_in_detail` for the underlying
random-number stream design and the `activitysim.core.random` API. It boils down to
each chooser needing to have the same ID between scenarios, and all alternatives being
reproduciably ordered.

For chooser alignment, it is necessary that person and household IDs are stable between runs.
When running a scenario with population changes, it is important to only change the IDs of
those households and persons that have changed, e.g., new households.

For alternative alignment, it is important to know the universal choice set, i.e., all possible
alternatives, for each model. For example, when running scenarios where a new mode is introduced,
this new mode should also be in the specification of the run where it is not available, with
its utility specification such that it is never chosen. In case the model is nested logit, the
nesting structure also needs to be held constant across scenarios.
For location choice models, all alternatives need to be listed in the land use table and the
zone IDs need to be stable between scenarios. Additionally, for computational efficiency it
is recommended to have zone IDs that are a contiguous 0-based sequence because ActivitySim aligns
random draws to positions in the full zone universe and generates draws for all zone IDs up to the
maximum. For models where this is not the case, ActivitySim can automatically perform the
conversion for internal calculations. The `recode_columns` option creates contiguous zero-based IDs
where needed; see the
[Zero-based Recoding of Zones](using-sharrow.md#zero-based-recoding-of-zones) section for details.

For models that use sub-sampling of alternatives, it is important to keep the sampling scheme
identical between scenarios, otherwise the error terms for the choice from the sampled set are
not guaranteed to be aligned. When running with EET, the default sampling method is ``poisson``,
which balances runtime performance and noise reduction. For more details on sampling methods,
see {ref}`sampling-methods-dev`.

Finally, it also important to keep the global random number generator seed constant for two
individual comparison runs.


### Runtime and memory usage
EET draws one error term per chooser and alternative, which requires many more random numbers
than MC's one per chooser. For models with many alternatives, this can lead to a large amount
of random numbers being calculated. The implementation of EET avoids materialization of large
chooser-alternative arrays of error terms in memory so that the memory usage is in line with MC
simulation.
Regarding runtimes, EET with default settings currently carries a runtime penalty of about 3-10%
per demand model run. However, when run in combination with an assignment model the overall
system can converge faster and this can reduce the overall model runtime penalty.

<!-- For location choice models, keeping error terms aligned to zone IDs also affects runtime and
memory usage. To keep the same unobserved error term attached to the same zone across runs,
ActivitySim indexes EET draws by zone ID over the full universal choice set rather than only the
alternatives that happen to appear in a given calculation.

When zone IDs are a contiguous 0-based sequence, this indexing is efficient because the dense
draw array has one entry per zone. When zone IDs contain gaps or start from a large value, the
implementation must still allocate draws up to the maximum zone ID, so additional random numbers
are generated for missing IDs and never used. Encoding zone IDs as a contiguous 0-based index can
therefore reduce both runtime and memory use for location choice models with EET; see
{ref}`explicit_error_terms_zone_encoding` for how to set this up.
-->


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

### Normalization
For MNL, the error term scale is normalized to 1 by using the standard Gumbel distribution. For
nested logit, ActivitySim uses the normalized formulation in which the root nest coefficient is
fixed at 1; the EET implementation relies on that convention.
