"""Benchmark ActivitySim's reproducible random-channel implementations.

The benchmark separates one-time process compilation, per-step state
initialization, and warm generation. Run it from the repository root with::

    python other_resources/performance-checks/fast-channel-random.py

Use ``--rows`` for a shorter exploratory run. Timings are descriptive rather
than pass/fail thresholds because process startup and hardware vary widely.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
import timeit
import tracemalloc
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from activitysim.core.random import Random

CHANNEL_TYPES = ("simple", "fast", "faster")
INDENT = "  "
SEP = "=" * 88


@dataclass
class Timing:
    """One timing result for the summary table."""

    phase: str
    operation: str
    channel_type: str
    milliseconds: float


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        type=int,
        default=250_000,
        help="number of channel rows to generate (default: 250000)",
    )
    parser.add_argument(
        "--number",
        type=int,
        default=2,
        help="calls in each warm timing sample (default: 2)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="timing samples; the best is reported (default: 3)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="spawned workers used for aggregate startup timing; 0 disables it",
    )
    return parser.parse_args()


def section(title):
    print()
    print(SEP)
    print(title)
    print(SEP)


def make_households(row_count):
    """Build a reproducible channel with sparse, unique household IDs."""
    prng = np.random.default_rng(seed=12345)
    # Oversampling makes it very likely that enough unique IDs remain without
    # constructing a one-billion-element population array.
    candidates = prng.integers(1_000_000_000, size=row_count + row_count // 100 + 1)
    index = pd.Index(np.unique(candidates)[:row_count], name="household_id")
    if len(index) != row_count:
        raise RuntimeError("failed to generate the requested number of unique IDs")
    return pd.DataFrame({"dummy": 1}, index=index)


def make_manager(channel_type, households):
    """Create a manager with one channel but no active pipeline step."""
    rng = Random(channel_type=channel_type)
    rng.set_base_seed(42)
    rng.add_channel("households", households)
    return rng


def elapsed_ms(fn):
    start = time.perf_counter()
    value = fn()
    return (time.perf_counter() - start) * 1_000, value


def best_ms(fn, number, repeat):
    """Return the best average call time in milliseconds."""
    samples = timeit.repeat(fn, number=number, repeat=repeat)
    return min(samples) * 1_000 / number


def first_step_draw(channel_type, households, requested, step_name):
    """Time begin_step plus its first draw, including full-domain state setup."""
    rng = make_manager(channel_type, households)

    def draw():
        rng.begin_step(step_name)
        return rng.normal_for_df(requested, mu=3.0, sigma=1.5)

    milliseconds, values = elapsed_ms(draw)
    rng.end_step(step_name)
    return milliseconds, values


def worker_first_call(task):
    """Build a channel and measure its first draw in a fresh spawned worker."""
    channel_type, row_count = task
    households = make_households(row_count)
    milliseconds, _ = first_step_draw(
        channel_type,
        households,
        households,
        step_name="worker_first_call",
    )
    return milliseconds


def spawned_worker_timing(channel_type, row_count, worker_count):
    """Measure wall time and individual first calls for fresh worker processes."""
    context = mp.get_context("spawn")
    tasks = [(channel_type, row_count)] * worker_count
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
        worker_milliseconds = list(executor.map(worker_first_call, tasks))
    wall_milliseconds = (time.perf_counter() - start) * 1_000
    return wall_milliseconds, worker_milliseconds


def warm_compiled_kernels(households):
    """Compile all fast kernels before measuring per-step initialization."""
    tiny = households.iloc[:8]
    utilities = pd.DataFrame(
        np.tile(np.linspace(-1.0, 1.0, 4), (len(tiny), 1)), index=tiny.index
    )
    for channel_type in ("fast", "faster"):
        rng = make_manager(channel_type, tiny)
        rng.begin_step("jit_warmup")
        rng.random_for_df(tiny, n=2)
        rng.normal_for_df(tiny)
        rng.gumbel_for_df(tiny, n=4)
        rng.gumbel_choice_positions_for_df(utilities)
        rng.choice_for_df(tiny, a=np.arange(10), size=3, replace=False)
        rng.end_step("jit_warmup")


def state_megabytes(rng, households):
    """Return memory held by a channel's persistent per-row RNG state."""
    channel = rng.get_channel_for_df(households)
    if hasattr(channel, "_state_array"):
        return channel._state_array.nbytes / (1024**2)
    return channel.row_states.memory_usage(index=True, deep=True).sum() / (1024**2)


def tracked_peak_megabytes(fn):
    """Measure peak Python/NumPy allocation visible to tracemalloc."""
    tracemalloc.start()
    try:
        value = fn()
        _, peak = tracemalloc.get_traced_memory()
        # Keep the result alive until after the peak is sampled.
        del value
    finally:
        tracemalloc.stop()
    return peak / (1024**2)


def reproducibility_sequence(channel_type, households):
    """Run a mixed sequence used to verify fresh-manager reproducibility."""
    requested = households.iloc[[2, 0, 1]]
    rng = make_manager(channel_type, households)
    rng.begin_step("reproducibility")
    sequence = (
        rng.random_for_df(requested, n=3),
        rng.normal_for_df(requested),
        rng.choice_for_df(requested, a=np.arange(20), size=5, replace=False),
    )
    rng.end_step("reproducibility")
    return sequence


def print_timing_table(timings):
    section("Timing summary")
    print(
        f"{INDENT}{'Phase':<22} {'Operation':<34} " f"{'Channel':<10} {'Best ms':>12}"
    )
    print(f"{INDENT}{'-' * 22} {'-' * 34} {'-' * 10} {'-' * 12}")
    for item in timings:
        print(
            f"{INDENT}{item.phase:<22} {item.operation:<34} "
            f"{item.channel_type:<10} {item.milliseconds:12.3f}"
        )


def main():
    args = parse_args()
    if args.rows < 8 or args.number < 1 or args.repeat < 1 or args.workers < 0:
        raise ValueError(
            "--rows must be at least 8; --number and --repeat must be positive; "
            "--workers cannot be negative"
        )

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    households = make_households(args.rows)
    subset = households.iloc[:3]
    timings = []

    section("Setup")
    print(f"{INDENT}Rows: {len(households):,}")
    print(f"{INDENT}Warm timing calls/sample: {args.number}")
    print(f"{INDENT}Warm timing samples: {args.repeat}")
    print(f"{INDENT}Spawned workers: {args.workers or 'disabled'}")
    print(f"{INDENT}Channel types: {', '.join(CHANNEL_TYPES)}")

    section("Observed first call in this process")
    print(
        f"{INDENT}These measurements include any Numba compilation encountered in "
        "the displayed order. They are intentionally separated from repeatable "
        "per-step initialization timings."
    )
    for channel_type in CHANNEL_TYPES:
        milliseconds, _ = first_step_draw(
            channel_type,
            households,
            households,
            step_name="process_first_call",
        )
        timings.append(
            Timing(
                "process first call", "normal_for_df, full", channel_type, milliseconds
            )
        )
        print(f"{INDENT}{channel_type:<10} {milliseconds:12.3f} ms")

    if args.workers:
        section("Aggregate spawned-worker startup")
        print(
            f"{INDENT}Each spawned worker imports ActivitySim, creates its channel, "
            "and performs one full-domain normal draw. Wall time includes process "
            "startup and input/channel construction; worker times cover begin_step "
            "and the first draw."
        )
        for channel_type in CHANNEL_TYPES:
            try:
                wall, worker_times = spawned_worker_timing(
                    channel_type, len(households), args.workers
                )
            except OSError as err:
                print(
                    f"{INDENT}Spawned-worker timing is unavailable in this "
                    f"environment: {err}"
                )
                break
            timings.append(
                Timing(
                    "spawned workers",
                    f"{args.workers} workers, aggregate wall",
                    channel_type,
                    wall,
                )
            )
            print(
                f"{INDENT}{channel_type:<10} wall {wall:12.3f} ms; "
                f"worker first draws {min(worker_times):.3f}–"
                f"{max(worker_times):.3f} ms"
            )

    warm_compiled_kernels(households)

    section("Cold per-step initialization")
    print(
        f"{INDENT}Cold means compiled kernels are warm but a new step has no row "
        "state. Both current channel implementations initialize the full registered "
        "domain even when the first request contains only three rows."
    )
    for channel_type in CHANNEL_TYPES:
        for label, requested in (("full", households), ("3-row subset", subset)):
            samples = []
            for sample_number in range(args.repeat):
                milliseconds, _ = first_step_draw(
                    channel_type,
                    households,
                    requested,
                    step_name=f"cold_{label}_{sample_number}",
                )
                samples.append(milliseconds)
            best = min(samples)
            timings.append(
                Timing("cold step", f"normal_for_df, {label}", channel_type, best)
            )
            print(f"{INDENT}{channel_type:<10} {label:<14} {best:12.3f} ms")

    section("Warm generation")
    print(
        f"{INDENT}Each manager is initialized once before timing. These measurements "
        "therefore isolate repeated generation from step reseeding."
    )
    utilities = pd.DataFrame(
        np.tile(np.linspace(-1.0, 1.0, 8), (len(households), 1)),
        index=households.index,
    )
    managers = {}
    operations: dict[str, dict[str, Callable]] = {}
    for channel_type in CHANNEL_TYPES:
        rng = make_manager(channel_type, households)
        rng.begin_step("warm_generation")
        rng.random_for_df(households)
        managers[channel_type] = rng
        operations[channel_type] = {
            "random_for_df, full": lambda rng=rng: rng.random_for_df(households),
            "random_for_df, 3-row subset": lambda rng=rng: rng.random_for_df(subset),
            "normal_for_df, full": lambda rng=rng: rng.normal_for_df(households),
            "gumbel_for_df, 8 draws": lambda rng=rng: rng.gumbel_for_df(
                households, n=8
            ),
            "gumbel choice, 8 alternatives": lambda rng=rng: (
                rng.gumbel_choice_positions_for_df(utilities)
            ),
            "choice 5 of 20, no replacement": lambda rng=rng: rng.choice_for_df(
                households, a=np.arange(20), size=5, replace=False
            ),
        }

    for operation_name in operations["simple"]:
        for channel_type in CHANNEL_TYPES:
            milliseconds = best_ms(
                operations[channel_type][operation_name],
                number=args.number,
                repeat=args.repeat,
            )
            timings.append(
                Timing("warm generation", operation_name, channel_type, milliseconds)
            )
            print(
                f"{INDENT}{operation_name:<34} {channel_type:<10} "
                f"{milliseconds:12.3f} ms"
            )

    section("Memory")
    print(
        f"{INDENT}Persistent state is measured after initialization. Peak temporary "
        "memory is the allocation visible to tracemalloc during one warm eight-"
        "alternative Gumbel choice; native allocations invisible to tracemalloc "
        "are not included."
    )
    print(
        f"{INDENT}{'Channel':<10} {'Persistent state MiB':>22} "
        f"{'Tracked peak MiB':>20}"
    )
    for channel_type in CHANNEL_TYPES:
        persistent = state_megabytes(managers[channel_type], households)
        peak = tracked_peak_megabytes(
            operations[channel_type]["gumbel choice, 8 alternatives"]
        )
        print(f"{INDENT}{channel_type:<10} {persistent:22.3f} {peak:20.3f}")

    section("Reproducibility")
    for channel_type in CHANNEL_TYPES:
        first = reproducibility_sequence(channel_type, households)
        second = reproducibility_sequence(channel_type, households)
        reproducible = all(np.array_equal(a, b) for a, b in zip(first, second))
        print(
            f"{INDENT}{channel_type:<10} fresh-manager mixed sequence: {reproducible}"
        )
        if not reproducible:
            raise AssertionError(f"{channel_type} failed the reproducibility check")

    for rng in managers.values():
        rng.end_step("warm_generation")

    print_timing_table(timings)
    print()
    print(
        f"{INDENT}Interpret cold and warm timings separately. In particular, the "
        "three-row cold result includes full-domain state initialization and must "
        "not be described as lazy per-row seeding."
    )


if __name__ == "__main__":
    main()
