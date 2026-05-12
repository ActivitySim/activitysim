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
and memory use for the aligned `eet` and `poisson` sampling methods.

(explicit_error_terms_zone_encoding)=
(sampling_methods_zone_encoding)=
### Zone ID encoding and runtime

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
