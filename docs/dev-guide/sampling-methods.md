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

- `inverse_cdf`: importance sampling with replacement using probabilities and uniform draws
- `eet`: importance sampling with replacement using explicit error-term draws
- `poisson`: importance sampling via independent Poisson inclusion sampling based on probabilities

## Defaults and Overrides

At the top level, `sample_method` may be set in `settings.yaml`.
When it is omitted, ActivitySim preserves the intended default behavior:

- if `use_explicit_error_terms` is `False`, `interaction_sample` defaults to `inverse_cdf`
- if `use_explicit_error_terms` is `True`, `interaction_sample` defaults to `poisson`

Individual models may override this default through:

```yaml
compute_settings:
  sample_method: eet
```

## Workflow

The sampled-choice workflow is:

1. Evaluate a simplified sampling utility for the full active alternative set
2. Draw a sample of alternatives using one of the three methods
3. Return a sampled-alternative table with one row per chooser-sampled-alternative pair and information about the sampling probabilities
4. Compute expensive terms, such as `mode_choice_logsum`, only for that sampled table
5. Add the sampling correction term to the final utility and choose from the sampled set

This is the standard sample-of-alternatives pattern: the sampling stage uses an approximation,
and the final stage corrects for it.

### Inverse-CDF and EET-with-replacement

The `inverse_cdf` and `eet` sampling methods both draw alternatives with replacement. As a result,
duplicates are possible within a chooser's sampled set, and sampled shares track repeated-draw MNL
behavior closely.

The difference between them is how each draw is made:

- `inverse_cdf` draws from analytical probabilities using uniform random numbers against the
  cumulative distribution
- `eet` draws explicit EV1 error terms and chooses the utility-plus-error argmax

`eet` freezes the error terms for each chooser-alternative pair across repeated draws, so that
unchanged alternatives can keep the same unobserved draws, which can greatly reduce
scenario-to-scenario sampling noise compared to `inverse_cdf`. However, `eet` is more expensive to
run because it requires many more random draws and more complex logic to avoid materializing large
chooser-alternative arrays of error terms in memory.

### Poisson Sampling

`poisson` does not perform repeated draws with replacement. Instead, each chooser-alternative
pair is sampled independently with inclusion probability $1 - (1 - p)^s$, where $p$ is the original
choice probability and $s$ is the configured sample size.
A single inclusion draw is made for each alternative. This is much cheaper than repeated draws for
`eet`, and unlike ``inverse_cdf``, it can still benefit from stable alignment of random draws to
alternatives, so it can provide improved noise reduction compared to `inverse_cdf` without the full
cost of `eet` and therefore it is the default when running with explicit error terms, see
{ref}`explicit-error-terms-dev`.

Because sampled alternatives appear at most once per chooser, raw sampled shares can differ
noticeably from repeated-draw MNL shares in highly peaked cases. This is structural behavior, not
numerical noise. The interaction-sample tests document this explicitly.

Under `poisson`, the configured sample size $s$ is a rate parameter rather than a count of draws,
and it is deliberately not clamped to the number of alternatives (`inverse_cdf` and `eet` clamp it,
which is statistically harmless for with-replacement draws because the omitted $\log s$ term in the
correction is constant per chooser). When a chooser has fewer available alternatives than $s$, its
probability mass is concentrated and the inclusion probabilities saturate towards 1: the chooser
receives essentially its whole availability set, each alternative with a correction term near
$\log(1/1) = 0$, and the final choice approaches exact MNL over the true availability set. No
special-casing is needed for such choosers; their expected sample size $\sum_i q_i$ is simply
smaller than $s$.

A chooser can occasionally receive no sampled alternatives under Poisson sampling, because each
alternative is tested independently. The probability of this happening for a given chooser is

$$
P_0 = \prod_j (1 - p_j)^s
$$

Because the probabilities sum to one and $1 - p \le e^{-p}$, this is bounded above by $e^{-s}$
regardless of how the probabilities are distributed. It is therefore negligible at the sample sizes
these models use (at most $10^{-13}$ for a sample size of 30), but not negligible at small sample
sizes, or for a chooser whose probability mass is spread very thinly. If it happens, that chooser
falls back to its $\min(s, n)$ highest-probability *available* alternatives, where $n$ is the
number of alternatives with non-zero probability. Zero-probability alternatives are never included,
so an unavailable alternative cannot enter the choice set through either branch.

The fallback is deliberately deterministic and draws no random numbers, so every chooser advances
its random number channel by exactly the same amount whether or not the fallback fires. A retry or
redraw scheme cannot do this: the number of retries is data-dependent, so two nearby scenarios would
consume different numbers of randoms for the same chooser and desynchronise every draw after it,
which is undesired when running in explicit error term simulation mode. Determinism also keeps the
reported `prob` exact, see below.

Taking the $s$ most likely alternatives is only sound as a rare repair, not as a sampling method in
its own right. Used on its own it would give every selected alternative an inclusion probability of
1 and every other alternative an inclusion probability of 0, so the correction term would be the
same constant for all selected alternatives and would cancel out of the choice entirely. The result
is a plain MNL over the top $s$ alternatives, and the choice mass on all remaining alternatives is
lost. Because the sampling utility is a deliberately cheap approximation, the alternatives it ranks
poorly are not the same ones the final utility ranks poorly, so this is a systematic bias rather
than sampling noise. The problem is the deterministic *exclusion*, not the determinism itself: an
inclusion probability of 1 is perfectly valid, but one of 0 cannot be corrected for by any
weighting. Here the Bernoulli draw keeps every alternative's inclusion probability strictly
positive, and the fallback can only add inclusion mass on top of that, never remove it.


### Sampling Correction

`interaction_sample` returns a dataframe indexed by chooser id with columns including:

- the sampled alternative id column
- `prob`
- `pick_count`

For `inverse_cdf` and `eet`, `pick_count` is the number of times the alternative was selected in
the repeated with-replacement draws. For `poisson`, `pick_count` is always `1`, because an
alternative is either included or not included. For all methods, `prob` is the quantity used in
the correction term, but it means different things for different methods. ActivitySim's final
sampled-choice specs typically include the term:

```python
np.log(df.pick_count/df.prob)
```

This is the sample-of-alternatives correction factor used in the final choice model.

For `inverse_cdf` and `eet`, `prob` is the one-draw sampling probability implied by the
approximate sampling utility, and `pick_count` is the number of times that alternative appeared in
the repeated sample. McFadden's utility correction term for repeated with-replacement sampling is
`log(pick_count / (sample_size * prob)) = log(pick_count / prob) - log(sample_size)`. ActivitySim
omits the common `sample_size` term because it is the same for every sampled alternative for that
chooser and therefore does not affect choice probabilities.

For `poisson`, `prob` is the inclusion probability of the alternative in the sampled set, not the
one-draw choice probability. Specifically, if the original approximate choice probability is $p$
and the configured sample size is $s$, then the inclusion probability of the Bernoulli trial is

$$
q_i = 1 - (1 - p_i)^s
$$

An alternative ends up in the returned choice set either because its inclusion draw succeeded,
or because the chooser drew nothing at all and the alternative is in the fallback set. Something
that was drawn cannot also have been part of an empty draw, so these two events are disjoint, and
because the fallback set is deterministic rather than random the returned `prob` is

$$
\text{prob}_i = q_i + P_0 \cdot 1\{i \in \text{fallback set}\}
$$

Note this is the *unconditional* probability, not either branch on its own. Conditional on the draw
being non-empty the inclusion probability is $q_i / (1 - P_0)$, and conditional on it being empty it
is $1\{i \in \text{fallback set}\}$; mixing those with weights $1 - P_0$ and $P_0$ recovers the
expression above. The unconditional form is the one the correction needs, and it is also what makes
the reported `prob` independent of which branch a given chooser happened to take.

The conditional form $q_i / (1 - P_0)$ is what a design that retried until the draw was non-empty
would have to report. Both designs are valid samplers; this one has an exact closed form that does
not depend on how many times a given chooser was redrawn.

Ranking the probabilities to find the fallback set costs about as much as the Bernoulli draw itself,
so the implementation evaluates the fallback term only for choosers whose $P_0$ exceeds
`POISSON_EMPTY_SAMPLE_TOLERANCE`, which is set to $10^{-12}$, plus every chooser that actually drew
nothing. Since $P_0 \le e^{-s}$, this branch is never evaluated above a sample size of 27. Dropping
the term understates `prob` by $P_0$, so the relative error on the correction is $P_0 / q_i$, which
is only large for an alternative whose own inclusion probability is far below $P_0$. But such an
alternative can only be affected if it is sampled, which happens with probability $q_i$, the same
small quantity. That coupling keeps the expected number of materially wrong corrections far below
one for any model size.

Since `pick_count` is always `1` for `poisson`, the correction becomes $\log(1 / \text{prob})$.

This means that all three methods use the same correction expression,
`np.log(df.pick_count/df.prob)`, even though `prob` has a different interpretation for `poisson`
than for the with-replacement methods.

<!-- TODO: Add section on disaggregate accessibilities here once decision is made in engineering meeting -->

## Runtime and Simulation Noise

Runtime and noise characteristics differ across methods.

- `inverse_cdf` is the fastest method. It draws one uniform random number per repeated sample for
  each chooser, but it also has the most simulation noise because small changes in approximate
  probabilities can change the sampled set substantially.
- `poisson` is also relatively inexpensive. It draws one uniform random number per
  chooser-alternative pair (with stable alternative alignment, one per chooser and
  stable-universe alternative, so inactive alternatives also consume draws). With stable
  alternative alignment it is much less noisy than inverse-CDF sampling.
- `eet` is the slowest sampling method. It draws one EV1 error term per chooser, alternative, and
  repeated sample draw. In return, it produces the most stable sampled sets across scenarios
  because unchanged alternatives keep the same unobserved error draws and only observed utility
  changes can change the sampled set.

Note that `eet` does not remove the dependence on the approximate sampling utility itself: if that
utility changes, the sampled set can still change. What it removes is the extra noise from the
probability-space sampling draw. `poisson` also benefits from stable alignment per alternative, but
unlike `eet` it still depends on probability-based inclusion tests. The practical effect on
scenario comparisons is expected to be negligible, and empirical tests with an increase in
employment in some zones for the SANDAG example model confirm this. `poisson` is therefore the
default sampling method when running in explicit error term simulation mode.


## References

- Kenneth Train, *Discrete Choice Methods with Simulation*, 2nd edition, Cambridge University
  Press, 2009. Chapter 3.7 treats sampled choice sets and choice-model correction terms from
  an estimation perspective.
- Carl-Erik Sarndal, Bengt Swensson, and Jan Wretman, *Model Assisted Survey Sampling*, Springer,
  1992. This is a standard reference for Poisson sampling as independent inclusion sampling.
