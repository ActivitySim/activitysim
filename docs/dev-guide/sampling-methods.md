(sampling-methods-dev)=
# Sampling Methods for Interaction Sample

`activitysim.core.interaction_sample` supports multiple alternative-sampling methods.
These methods are independent of the global final-choice switch controlled by
`use_explicit_error_terms`, although the global switch determines the default when no
sampling-method override is provided.

For user-facing configuration guidance, see {ref}`sampling_methods_ways_to_run`.

## Available Methods

- `monte_carlo`: importance sampling with replacement using probabilities and uniform draws
- `eet`: importance sampling with replacement using explicit error-term draws
- `poisson`: independent Poisson inclusion sampling

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

## Behavioral Differences

### Monte Carlo and EET-with-replacement

The `monte_carlo` and `eet` sampling methods both draw sampled alternatives with replacement.
As a result, duplicates are possible within a chooser's sampled set, and the resulting sampled
shares track repeated-draw MNL behavior closely.

The difference between them is how each draw is made:

- `monte_carlo` draws from analytical probabilities using uniform random numbers
- `eet` draws explicit EV1 error terms and chooses the utility-plus-error argmax

### Poisson Sampling

`poisson` does not perform repeated draws with replacement. Instead, each chooser-alternative
pair is sampled independently with inclusion probability
$1 - (1 - p)^s$, where $p$ is the original choice probability and $s$ is the configured
sample size.

Because sampled alternatives appear at most once per chooser, raw sampled shares can differ
substantially from repeated-draw MNL shares in highly peaked cases. This is structural behavior,
not numerical noise. The interaction-sample tests document this explicitly.

## Runtime and Zone Encoding

Sampling runtime differs significantly between methods.

- `monte_carlo` draws one uniform random number per repeated sample
- `eet` draws one EV1 error term per chooser-alternative-sample combination
- `poisson` draws one Bernoulli inclusion test per chooser-alternative pair and may retry rows
  that sample no alternatives

For location choice models, encoding zone IDs as a 0-based contiguous index can reduce runtime
and memory use for the `eet` sampling method.

(explicit_error_terms_zone_encoding)=
(sampling_methods_zone_encoding)=
### Zone ID encoding and runtime

For location choice models, encoding zone IDs as a 0-based contiguous index reduces EET runtime
and memory use during sampling.

The current `eet` sampling implementation draws error terms into a dense 1-D array of length
`max_zone_id + 1` per chooser (see `AltsContext.n_alts_to_cover_max_id` in
`activitysim.core.logit`). Each sampled alternative is then looked up by direct offset into that
array, so the same zone always receives the same error term regardless of which alternatives are
in the sampled choice set.

When zone IDs are a contiguous 0-based sequence, the dense array has exactly as many entries as
there are zones and every draw is used. When zone IDs contain gaps or start from a large value,
the array must still cover `max_zone_id + 1` entries, so draws for missing IDs are generated but
never used.

ActivitySim's `recode_columns` option can create contiguous zero-based IDs where needed; see the
[Zero-based Recoding of Zones](using-sharrow.md#zero-based-recoding-of-zones) section for details.
