(sampling-methods-dev)=
# Sampling Methods for Interaction Sample

`activitysim.core.interaction_sample` supports multiple alternative-sampling methods.
These methods are independent of the global final-choice switch controlled by
`use_explicit_error_terms`, although the global switch determines the default when no
sampling-method override is provided.

For user-facing configuration guidance, see {ref}`sampling_methods_ways_to_run`.

## Why sample alternatives?

`interaction_sample` is mainly used in destination and location choice models, where the full
utility can be expensive to evaluate for every chooser-alternative pair. The most common example
is mode choice logsums: computing a logsum for every chooser and every possible destination can be
much more expensive than the final destination-choice simulation itself.

ActivitySim handles this by splitting the problem into two stages:

1. Build a sampled choice set using a cheaper approximate utility.
2. Compute the expensive terms only for the sampled alternatives and make the final choice from
  that sampled set.

In the example models, the sampling utility usually replaces `mode_choice_logsum` with cheaper
proxies such as distance skims. For example,
`activitysim/examples/prototype_arc/configs/school_location_sample.csv` and
`activitysim/examples/prototype_mtc/configs/workplace_location_sample.csv` use distance-based
sampling utilities, while the corresponding final-choice specs in
`activitysim/examples/prototype_arc/configs/school_location.csv` and
`activitysim/examples/prototype_mtc/configs/workplace_location.csv` add the full
`mode_choice_logsum` and a sampling correction term.

## Available Methods

- `monte_carlo`: importance sampling with replacement using probabilities and uniform draws
- `eet`: importance sampling with replacement using explicit error-term draws
- `poisson`: importance sampling via independent Poisson inclusion sampling

## Defaults and Overrides

At the top level, `sample_method` may be set in `settings.yaml`.
When it is omitted, ActivitySim preserves the intended default behavior:

- if `use_explicit_error_terms` is `False`, `interaction_sample` defaults to `monte_carlo`
- if `use_explicit_error_terms` is `True`, `interaction_sample` defaults to `poisson`

A model may override this default through:

```yaml
compute_settings:
  sample_method: eet
```

This override affects only `activitysim.core.interaction_sample`.
It does not change the final-choice simulation method used by
`simulate`, `interaction_simulate`, or `interaction_sample_simulate`.

## Workflow

At a high level, the sampled-choice workflow is:

1. Evaluate a simplified sampling utility for the full active alternative set.
2. Draw a sample of alternatives using one of the methods described below.
3. Return a sampled-alternative table with one row per chooser-sampled-alternative pair.
4. Compute expensive terms, such as `mode_choice_logsum`, only for that sampled table.
5. Add the sampling correction term to the final utility and choose from the sampled set.

This is the standard sample-of-alternatives pattern: the sampling stage can use an approximation,
as long as the final stage includes the appropriate correction term.

## Returned Sample Table

`interaction_sample` returns a dataframe indexed by chooser id with columns including:

- the sampled alternative id column named by `alt_col_name`
- `prob`
- `pick_count`

For `monte_carlo`, the implementation also creates a `rand` column during sampling, but that
column is only kept for tracing and is dropped from the returned dataframe in normal operation.

The meanings of `prob` and `pick_count` are important:

- For `monte_carlo` and `eet`, `pick_count` is the number of times the alternative was selected in
  the repeated with-replacement draws.
- For `poisson`, `pick_count` is always `1`, because an alternative is either included or not
  included.
- For all methods, `prob` is the quantity used in the correction term, but it does not mean the
  same thing for every method.

## Sampling Correction

ActivitySim's final sampled-choice specs typically include the term:

```python
np.log(df.pick_count/df.prob)
```

This is the sample-of-alternatives correction factor used in the final choice model.

For `monte_carlo` and `eet`, `prob` is the one-draw sampling probability implied by the
approximate sampling utility, and `pick_count` is the number of times that alternative appeared in
the repeated sample. In textbook notation, the correction for repeated with-replacement sampling is
proportional to `pick_count / (sample_size * prob)`. ActivitySim omits the common `sample_size`
factor because it is the same for every sampled alternative for that chooser and therefore only
adds a constant to utility for all alternatives, which does not change probabilities.

For `poisson`, `prob` is the inclusion probability of the alternative in the sampled set, not the
one-draw choice probability. Specifically, if the original approximate choice probability is $p$
and the configured sample size is $s$, then the returned `prob` is:

$$
1 - (1 - p)^s
$$

Since `pick_count` is always `1` for `poisson`, the correction becomes exactly
$\log(1 / \text{prob})$.

This means that all sampling methods can be used interchangeably as long as the correction factor
is specified as `np.log(df.pick_count/df.prob)`.

## Methods in Detail

### Monte Carlo and EET-with-replacement

The `monte_carlo` and `eet` sampling methods both draw sampled alternatives with replacement.
As a result, duplicates are possible within a chooser's sampled set, and the resulting sampled
shares track repeated-draw MNL behavior closely.

The difference between them is how each draw is made:

- `monte_carlo` draws from analytical probabilities using uniform random numbers
- `eet` draws explicit EV1 error terms and chooses the utility-plus-error argmax

For both methods, the returned `prob` column is the one-draw sampling probability of the selected
alternative under the approximate sampling utility. If an alternative is drawn multiple times,
those duplicate rows are collapsed and the total multiplicity is stored in `pick_count`.

In practice, these methods are useful when a modeler wants the sampled set to behave like repeated
draws from the approximate choice model. `eet` preserves that with-replacement behavior while also
freezing the unobserved draws in a way that can greatly reduce scenario-to-scenario sampling noise.

### Poisson Sampling

`poisson` does not perform repeated draws with replacement. Instead, each chooser-alternative
pair is sampled independently with inclusion probability
$1 - (1 - p)^s$, where $p$ is the original choice probability and $s$ is the configured
sample size.

Because sampled alternatives appear at most once per chooser, raw sampled shares can differ
noticeably from repeated-draw MNL shares in highly peaked cases. This is structural behavior,
not numerical noise. The interaction-sample tests document this explicitly.

There can be cases where a chooser does not receive any alternatives with Poisson sampling:
each alternative is subjected to an independent inclusion test with one random draw. However, for
the models that require sampling in ActivitySim, this is rare. If it happens, the sampler retries
that chooser row up to 10 times and then falls back to a simple without-replacement random sample.
That retry-and-fallback path is important in practice. It makes the method robust, but it also
means that Poisson sampling can have rare edge cases where two nearby scenarios consume different
random numbers because one scenario needed retries or fallback and the other did not.

## Runtime and Simulation Noise

Runtime and noise characteristics differ significantly across methods.

- `monte_carlo` is the cheapest method runtime wise. It draws one uniform random number per
  repeated sample, but it also has the most simulation noise because small changes in approximate
  probabilities can change the sampled set substantially.
- `poisson` is also relatively inexpensive. It draws one Bernoulli inclusion test per
  chooser-alternative pair, with possible retries for chooser rows that initially sample no
  alternatives. With stable alternative alignment it is much less noisy than Monte Carlo,
  but it can still show structural sample-set differences in highly peaked cases and rare retry
  edge cases.
- `eet` is the most expensive sampling method. It draws one EV1 error term per chooser,
  alternative, and repeated sample draw. In return, it produces the most stable sampled sets across
  nearby scenarios because unchanged alternatives can keep the same unobserved error draws.

For location choice models, this often leads to a practical ranking of:

- runtime: `monte_carlo` and `poisson` low, `eet` high
- simulation noise: `monte_carlo` high, `poisson` low, `eet` lowest

`eet` does not remove the dependence on the approximate sampling utility itself: if that utility
changes, the sampled set can still change. What it removes is the extra Monte Carlo noise from the
sampling draw. `poisson` also benefits from stable alignment, but unlike `eet` it still has some
edge cases because it uses probabilites for sampling and these depend on the utility of all
alternatives, as well as the retry/fallback edge case described above. 

(explicit_error_terms_zone_encoding)=
(sampling_methods_zone_encoding)=
### Runtime and Zone Encoding
For location choice models, ActivitySim can align random draws to positions in the full zone
universe rather than only to the alternatives active in the current sampled set. This keeps the
same zone attached to the same random draws regardless of which alternatives are present in a
particular chooser's calculation.

Both aligned `eet` and aligned `poisson` sampling use this stable mapping. For `eet`, each chooser
receives `sample_size` sets of Gumbel draws over the full encoded zone universe, and the active
alternatives are selected from those draws by their stable zone positions. For `poisson`, each
chooser receives one aligned uniform draw per encoded zone, and those draws are used for the
Bernoulli inclusion tests.

When zone IDs are a contiguous 0-based sequence, the aligned draw universe has exactly as many
positions as there are zones and every position is potentially useful. When zone IDs contain gaps
or start from a large value, the implementation must still cover the full encoded range, so draws
for missing IDs are generated but never used. This increases runtime and memory use, especially
for `eet`, where the aligned draw cost also scales with `sample_size`.

ActivitySim's `recode_columns` option can create contiguous zero-based IDs where needed; see the
[Zero-based Recoding of Zones](using-sharrow.md#zero-based-recoding-of-zones) section for details.

## References

- Kenneth Train, *Discrete Choice Methods with Simulation*, 2nd edition, Cambridge University
  Press, 2009. Chapter 3.7 treats sampled choice sets and choice-model correction terms from
  an estimation perspective.
- Carl-Erik Sarndal, Bengt Swensson, and Jan Wretman, *Model Assisted Survey Sampling*, Springer,
  1992. This is a standard reference for Poisson sampling as independent inclusion sampling.
