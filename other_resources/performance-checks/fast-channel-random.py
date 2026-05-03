"""
Random Number Generation Performance Benchmark
===============================================
Compares the "fast" channel (new implementation) vs. the "slow" channel
(legacy implementation) for various random generation functions in
activitysim.core.random.

Run this script from the activitysim repo root with:
    python other_resources/scripts/random-performance.py
"""

from __future__ import annotations

import sys
import textwrap
import timeit

import numpy as np
import pandas as pd

from activitysim.core.random import Random

# Windows consoles default to a locale-specific encoding (e.g. cp1252) that
# cannot represent the box-drawing characters used below.  Reconfiguring stdout
# and stderr to UTF-8 here means the script works correctly regardless of the
# active code page, without requiring the caller to set PYTHONUTF8=1.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


# ── helpers ────────────────────────────────────────────────────────────────────

SEP_WIDE = "=" * 72
SEP_THIN = "-" * 72
INDENT = "  "


def section(title):
    print()
    print(SEP_WIDE)
    print(f"  {title}")
    print(SEP_WIDE)


def subsection(title):
    print()
    print(f"{INDENT}{SEP_THIN[2:]}")
    print(f"{INDENT}{title}")
    print(f"{INDENT}{SEP_THIN[2:]}")


def note(text, indent=2):
    prefix = " " * indent + "│  "
    wrapped = textwrap.fill(
        text, width=68, initial_indent=prefix, subsequent_indent=prefix
    )
    print(wrapped)


def result(label, elapsed_ms, n=None):
    if n is not None:
        avg_ms = elapsed_ms / n
        print(
            f"{INDENT}  {label:<40s}  {avg_ms:8.3f} ms/call  "
            f"(total {elapsed_ms:.1f} ms × {n})"
        )
    else:
        print(f"{INDENT}  {label:<40s}  {elapsed_ms:8.3f} ms")


def speedup(fast_ms, slow_ms, fast_label="fast", slow_label="slow"):
    if fast_ms > 0:
        ratio = slow_ms / fast_ms
        print(f"{INDENT}  --> {slow_label} is {ratio:.1f}x slower than {fast_label}")


def bench(fn, number=10, repeat=3):
    """Return best average time in milliseconds."""
    times = timeit.repeat(fn, number=number, repeat=repeat)
    best_total = min(times)  # best total over `number` calls
    return best_total / number * 1_000  # convert to ms per call


# ── setup ──────────────────────────────────────────────────────────────────────

section("Setup")

note(
    "Building a household index of 250 000 unique random integers drawn "
    "from [0, 1 000 000 000).  This mimics a realistic, sparse household "
    "ID space as seen in large-scale travel-demand models."
)

prng = np.random.default_rng(seed=12345)
idxs = pd.Index(np.sort(prng.integers(1_000_000_000, size=250_000))).unique()
print(f"\n{INDENT}  Household IDs generated: {idxs.size:,}")

hh = pd.DataFrame(index=idxs, columns=[])
hh.index.name = "household_id"
hh["dummy"] = 1

# fast channel dataframe
shh = hh.copy()
shh.index.name = "slow_hhid"

# small slices for later
hh3 = hh.iloc[2:5]
shh3 = shh.iloc[2:5]

note(
    "Creating the Random object and registering one FAST channel "
    "(new implementation) and one SLOW channel (legacy implementation).",
    indent=2,
)

r = Random(channel_type="faster")
r.set_base_seed(42)
r.add_channel("households", hh, fast=True)
r.add_channel("slow_households", shh, fast=False)

print(f"\n{INDENT}  Random object created.")
print(f"{INDENT}  Base seed : 42")
print(f"{INDENT}  Fast channel index name  : '{hh.index.name}'")
print(f"{INDENT}  Slow channel index name  : '{shh.index.name}'")

# ── benchmark: first-call costs ────────────────────────────────────────────────

section("First-call costs (Numba JIT compilation + reseeding)")

note(
    "The very first call to random_for_df on a fast channel within an "
    "ActivitySim session triggers Numba ahead-of-time compilation.  "
    "This one-time cost disappears for all subsequent calls in the same "
    "session.  The slow channel does NOT use Numba."
)

r.begin_step("peekaboo")

t0 = timeit.default_timer()
r.random_for_df(hh)
first_fast_ms = (timeit.default_timer() - t0) * 1_000

t0 = timeit.default_timer()
r.random_for_df(shh)
first_slow_ms = (timeit.default_timer() - t0) * 1_000

print()
result("Fast channel – first call (w/ JIT + reseed)", first_fast_ms)
result("Slow channel – first call (w/ reseed)      ", first_slow_ms)
note(
    "The fast channel first-call cost includes Numba compilation.  "
    "This cost is paid only once per process, not once per step."
)

# ── benchmark: random_for_df (subsequent calls) ────────────────────────────────

section("random_for_df — subsequent calls within the same step")

note(
    "After the first call, the fast channel has already compiled its Numba "
    "kernels and cached the reseeded state.  Subsequent calls within the "
    "same step are pure number generation — very fast.  "
    "The slow channel must reseed from scratch on every single call, "
    "making it consistently slow regardless of call order."
)

fast_rand_ms = bench(lambda: r.random_for_df(hh), number=10, repeat=3)
slow_rand_ms = bench(lambda: r.random_for_df(shh), number=3, repeat=5)

print()
result("Fast channel  random_for_df  (n=250k)", fast_rand_ms)
result("Slow channel  random_for_df  (n=250k)", slow_rand_ms)
speedup(fast_rand_ms, slow_rand_ms)

# ── benchmark: normal_for_df ───────────────────────────────────────────────────

section("normal_for_df — uniform-to-normal transform (μ=3.0, σ=1.5)")

note(
    "Generating normally-distributed draws adds a Box–Muller / inverse-CDF "
    "transform on top of the uniform draws.  The fast channel benefits from "
    "the same Numba-compiled, pre-seeded state as before.  "
    "The slow channel pays the full reseeding cost every call."
)

# Trigger any remaining first-call compilation for normal_for_df
r.normal_for_df(hh, mu=3.0, sigma=1.5)

fast_norm_ms = bench(
    lambda: r.normal_for_df(hh, mu=3.0, sigma=1.5), number=10, repeat=3
)
slow_norm_ms = bench(
    lambda: r.normal_for_df(shh, mu=3.0, sigma=1.5), number=3, repeat=5
)

print()
result("Fast channel  normal_for_df  (n=250k)", fast_norm_ms)
result("Slow channel  normal_for_df  (n=250k)", slow_norm_ms)
speedup(fast_norm_ms, slow_norm_ms)

# ── benchmark: choice_for_df ───────────────────────────────────────────────────

section("choice_for_df — sampling without replacement (size=5 from [1..7])")

note(
    "choice_for_df is built atop random_for_df, so by this point the fast "
    "channel's seed state is already warm.  Drawing 5 choices per row "
    "without replacement means each row independently samples from [1,2,3,4,5,6,7]. "
    "The slow channel is again penalised by per-call reseeding."
)

fast_choice_ms = bench(
    lambda: r.choice_for_df(hh, a=[1, 2, 3, 4, 5, 6, 7], size=5, replace=False),
    number=5,
    repeat=3,
)
slow_choice_ms = bench(
    lambda: r.choice_for_df(shh, a=[1, 2, 3, 4, 5, 6, 7], size=5, replace=False),
    number=3,
    repeat=3,
)

print()
result("Fast channel  choice_for_df  (n=250k, size=5)", fast_choice_ms)
result("Slow channel  choice_for_df  (n=250k, size=5)", slow_choice_ms)
speedup(fast_choice_ms, slow_choice_ms)

r.end_step("peekaboo")

# ── benchmark: cross-step reproducibility & reseed cost ───────────────────────

section("Cross-step reproducibility: reseed cost at start of a new step")

note(
    "When a new step begins (or the same step name is restarted), the random "
    "state must be reseeded.  For the fast channel this happens exactly once "
    "— on the first draw after begin_step().  For the slow channel the cost "
    "is paid on every single draw, every step."
)

r.begin_step("peekaboo")

t0 = timeit.default_timer()
r.normal_for_df(hh, mu=3.0, sigma=1.5)
fast_reseed_ms = (timeit.default_timer() - t0) * 1_000

t0 = timeit.default_timer()
r.normal_for_df(shh, mu=3.0, sigma=1.5)
slow_reseed_ms = (timeit.default_timer() - t0) * 1_000

print()
result("Fast channel – first draw after new step (reseed + generate)", fast_reseed_ms)
result("Slow channel – first draw after new step (reseed + generate)", slow_reseed_ms)
note(
    "The slow channel's cost here is representative of EVERY subsequent "
    "call too, since it reseeds every time.  For the fast channel this "
    "cost is only paid once per step."
)

r.end_step("peekaboo")

# ── benchmark: same step name ⇒ same numbers ──────────────────────────────────

section("Reproducibility: re-running the same step name")

note(
    "Restarting a step with the same name must yield identical random draws. "
    "Here we verify this and measure the cost of the reseed-on-first-draw "
    "pattern again."
)

r.begin_step("peekaboo")

arr1_fast = r.normal_for_df(hh, mu=3.0, sigma=1.5)
arr1_slow = r.normal_for_df(shh, mu=3.0, sigma=1.5)

r.end_step("peekaboo")
r.begin_step("peekaboo")

arr2_fast = r.normal_for_df(hh, mu=3.0, sigma=1.5)
arr2_slow = r.normal_for_df(shh, mu=3.0, sigma=1.5)

fast_repro = np.allclose(arr1_fast, arr2_fast)
slow_repro = np.allclose(arr1_slow, arr2_slow)

print()
print(f"{INDENT}  Fast channel reproducible across step restarts: {fast_repro}")
print(f"{INDENT}  Slow channel reproducible across step restarts: {slow_repro}")
note(
    "Both channels produce identical draws when the same step name is reused, "
    "guaranteeing reproducibility across model runs."
)

r.end_step("peekaboo")

# ── benchmark: small subsets ───────────────────────────────────────────────────

section("Performance on small subsets (3 rows)")

note(
    "Even when only a handful of rows need random numbers, the slow channel "
    "must still reseed the entire household table before slicing.  "
    "The fast channel uses the pre-computed entropy to seed only the rows "
    "requested, making it proportionally cheaper on small subsets too."
)

r.begin_step("peekaboo")
# warm up fast channel reseed for this step
r.normal_for_df(hh, mu=3.0, sigma=1.5)

fast_small_ms = bench(
    lambda: r.normal_for_df(hh3, mu=3.0, sigma=1.5), number=3, repeat=5
)
slow_small_ms = bench(
    lambda: r.normal_for_df(shh3, mu=3.0, sigma=1.5), number=3, repeat=5
)

print()
result("Fast channel  normal_for_df  (n=3)", fast_small_ms)
result("Slow channel  normal_for_df  (n=3)", slow_small_ms)
speedup(fast_small_ms, slow_small_ms, fast_label="fast (warm)", slow_label="slow")
note(
    "Even for a 3-row draw the fast channel wins because the seeding work "
    "was already done at the first draw of this step."
)

r.end_step("peekaboo")

# ── summary ────────────────────────────────────────────────────────────────────

section("Summary")

rows = [
    ("random_for_df  250k rows", fast_rand_ms, slow_rand_ms),
    ("normal_for_df  250k rows", fast_norm_ms, slow_norm_ms),
    ("choice_for_df  250k rows x5", fast_choice_ms, slow_choice_ms),
    ("normal_for_df  3 rows (warm)", fast_small_ms, slow_small_ms),
]

col_w = 38
print()
print(
    f"{INDENT}  {'Operation':<{col_w}}  {'Fast (ms)':>10}  {'Slow (ms)':>10}  {'Speedup':>8}"
)
print(f"{INDENT}  {'-'*col_w}  {'-'*10}  {'-'*10}  {'-'*8}")
for label, f, s in rows:
    ratio = s / f if f > 0 else float("inf")
    print(f"{INDENT}  {label:<{col_w}}  {f:10.3f}  {s:10.3f}  {ratio:7.1f}x")

print()
note(
    "The fast channel should be faster in every scenario.  The gap should be largest "
    "on repeated within-step calls because the slow channel pays the full "
    "reseeding cost each time, while the fast channel pays it only once per "
    "step.  Both channels are fully reproducible when the same step name "
    "is reused."
)

# ── sanity check ───────────────────────────────────────────────────────────────

failures = [label for label, f, s in rows if s <= f]
if failures:
    print()
    print(f"{INDENT}  !! PERFORMANCE REGRESSION DETECTED !!")
    for label in failures:
        print(
            f"{INDENT}     · '{label}': slow channel was NOT slower than fast channel"
        )
    note(
        "A 'slow' result that is faster than (or equal to) the corresponding "
        "'fast' result indicates a regression in the fast-channel implementation "
        "or an anomaly in the benchmark environment.  Investigate before merging."
    )
    print()
    print(SEP_WIDE)
    print()
    sys.exit(1)

print()
print(SEP_WIDE)
print()
