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
- `poisson`: importance sampling via independent Poisson inclusion sampling based on probabilities

## Defaults and Overrides

At the top level, `sample_method` may be set in `settings.yaml`.
When it is omitted, ActivitySim preserves the intended default behavior:

- if `use_explicit_error_terms` is `False`, `interaction_sample` defaults to `monte_carlo`
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

### Monte Carlo and EET-with-replacement

The `monte_carlo` and `eet` sampling methods both draw alternatives with replacement. As a result,
duplicates are possible within a chooser's sampled set, and sampled shares track repeated-draw MNL
behavior closely.

The difference between them is how each draw is made:

- `monte_carlo` draws from analytical probabilities using uniform random numbers
- `eet` draws explicit EV1 error terms and chooses the utility-plus-error argmax

`eet` freezes the error terms for each chooser-alternative pair across repeated draws, so that
unchanged alternatives can keep the same unobserved draws, which can greatly reduce
scenario-to-scenario sampling noise compared to `monte_carlo`. However, `eet` is more expensive to
run because it requires many more random draws and more complex logic to avoid materializing large
chooser-alternative arrays of error terms in memory.

### Poisson Sampling

`poisson` does not perform repeated draws with replacement. Instead, each chooser-alternative
pair is sampled independently with inclusion probability $1 - (1 - p)^s$, where $p$ is the original
choice probability and $s$ is the configured sample size.
A single inclusion draw is made for each alternative. This is much cheaper than repeated draws for
`eet`, and unlike ``monte_carlo``, it can still benefit from stable alignment of random draws to
alternatives, so it can provide improved noise reduction compared to `monte_carlo` without the full
cost of `eet` and therefore it is the default when running with explicit error terms, see
{ref}`explicit-error-terms-dev`.

<!-- Because sampled alternatives appear at most once per chooser, raw sampled shares can differ
noticeably from repeated-draw MNL shares in highly peaked cases. This is structural behavior, not
numerical noise. The interaction-sample tests document this explicitly. -->

A chooser can occasionally receive no sampled alternatives under Poisson sampling, because each
alternative is tested independently. In the models that use sampling in ActivitySim, this should be
rare. If it happens, the sampler retries that chooser row up to 10 times and then falls back to a
simple without-replacement random sample.
<!-- This makes the method robust, but it also creates rare edge cases where two nearby scenarios
consume different random numbers because one scenario needed retries or fallback and the other did
not. -->


### Sampling Correction

`interaction_sample` returns a dataframe indexed by chooser id with columns including:

- the sampled alternative id column
- `prob`
- `pick_count`

For `monte_carlo` and `eet`, `pick_count` is the number of times the alternative was selected in
the repeated with-replacement draws. For `poisson`, `pick_count` is always `1`, because an
alternative is either included or not included. For all methods, `prob` is the quantity used in
the correction term, but it means different things for different methods. ActivitySim's final
sampled-choice specs typically include the term:

```python
np.log(df.pick_count/df.prob)
```

This is the sample-of-alternatives correction factor used in the final choice model.

For `monte_carlo` and `eet`, `prob` is the one-draw sampling probability implied by the
approximate sampling utility, and `pick_count` is the number of times that alternative appeared in
the repeated sample. McFadden's utility correction term for repeated with-replacement sampling is
`log(pick_count / (sample_size * prob)) = log(pick_count / prob) - log(sample_size)`. ActivitySim
omits the common `sample_size` term because it is the same for every sampled alternative for that
chooser and therefore does not affect choice probabilities.

For `poisson`, `prob` is the inclusion probability of the alternative in the sampled set, not the
one-draw choice probability. Specifically, if the original approximate choice probability is $p$
and the configured sample size is $s$, then the returned `prob` is:

$$
1 - (1 - p)^s
$$

Since `pick_count` is always `1` for `poisson`, the correction becomes $\log(1 / \text{prob})$.

This means that all three methods use the same correction expression,
`np.log(df.pick_count/df.prob)`, even though `prob` has a different interpretation for `poisson`
than for the with-replacement methods.

<<!-- TODO: Add section on disaggregate accessibilities here once decision is made in engineering meeting -->>

## Runtime and Simulation Noise

Runtime and noise characteristics differ across methods.

- `monte_carlo` is the fastest method. It draws one uniform random number per repeated sample for
  each chooser, but it also has the most simulation noise because small changes in approximate
  probabilities can change the sampled set substantially.
- `poisson` is also relatively inexpensive. It draws one uniform random number per
  chooser-alternative pair, with possible retries for chooser rows that initially sample no
  alternatives. With stable alternative alignment it is much less noisy than Monte Carlo.
- `eet` is the slowest sampling method. It draws one EV1 error term per chooser, alternative, and
  repeated sample draw. In return, it produces the most stable sampled sets across scenarios
  because unchanged alternatives keep the same unobserved error draws and only observed utility
  changes can change the sampled set.

Note that `eet` does not remove the dependence on the approximate sampling utility itself: if that
utility changes, the sampled set can still change. What it removes is the extra Monte Carlo noise
from the sampling draw. `poisson` also benefits from stable alignment per alternative, but unlike
`eet` it still depends on probability-based inclusion tests. The practical effect on scenario
comparisons is ultimately empirical.


## References

- Kenneth Train, *Discrete Choice Methods with Simulation*, 2nd edition, Cambridge University
  Press, 2009. Chapter 3.7 treats sampled choice sets and choice-model correction terms from
  an estimation perspective.
- Carl-Erik Sarndal, Bengt Swensson, and Jan Wretman, *Model Assisted Survey Sampling*, Springer,
  1992. This is a standard reference for Poisson sampling as independent inclusion sampling.
