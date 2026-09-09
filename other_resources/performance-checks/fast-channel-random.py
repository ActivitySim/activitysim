"""Benchmark ActivitySim random-channel lifecycle and generation costs.

The benchmark separates order-dependent process compilation, fresh-worker
startup, per-step state initialization, and warm generation. It also validates
the stream invariants on which chunked and filtered model execution relies.

Run a quick comparison from the repository root with::

    python other_resources/performance-checks/fast-channel-random.py

Use ``--profile full`` for the larger CI/manual comparison. Timings are
descriptive rather than pass/fail thresholds because startup and hardware vary.
CSV, JSON, and Markdown artifacts are written below ``output/`` by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import platform
import subprocess
import sys
import time
import timeit
import tracemalloc
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

from activitysim.core.random import Random

CHANNEL_TYPES = ("simple", "fast", "faster")
INDENT = "  "
SEP = "=" * 100
PROFILE_DEFAULTS = {
    "quick": {"rows": 25_000, "number": 1, "repeat": 3, "workers": 1},
    "full": {"rows": 250_000, "number": 2, "repeat": 5, "workers": 2},
}


@dataclass
class Timing:
    """Aggregate timing and its underlying per-call samples."""

    phase: str
    operation: str
    channel_type: str
    rows: int
    number: int
    samples_ms: list[float]

    def summary(self):
        samples = np.asarray(self.samples_ms, dtype=np.float64)
        return {
            "phase": self.phase,
            "operation": self.operation,
            "channel_type": self.channel_type,
            "rows": self.rows,
            "number": self.number,
            "repeat": len(self.samples_ms),
            "mean_ms": float(samples.mean()),
            "median_ms": float(np.median(samples)),
            "std_ms": float(samples.std()),
            "min_ms": float(samples.min()),
            "max_ms": float(samples.max()),
        }


@dataclass
class MemoryResult:
    """Persistent channel state and tracemalloc-visible temporary allocation."""

    channel_type: str
    rows: int
    persistent_mib: float
    tracked_peak_mib: float


@dataclass
class InvarianceCheck:
    """One deterministic-stream property checked by the benchmark."""

    channel_type: str
    check: str
    passed: bool


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILE_DEFAULTS),
        default="quick",
        help="benchmark size preset; explicit sizing options override it",
    )
    parser.add_argument("--rows", type=int, help="number of channel rows")
    parser.add_argument("--number", type=int, help="calls in each warm timing sample")
    parser.add_argument("--repeat", type=int, help="number of timing samples")
    parser.add_argument(
        "--workers",
        type=int,
        help="fresh spawned workers per channel; 0 disables worker timing",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/performance-checks/fast-channel-random"),
        help="directory for CSV, JSON, and Markdown artifacts",
    )
    args = parser.parse_args()
    defaults = PROFILE_DEFAULTS[args.profile]
    for name, default in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, default)
    return args


def section(title):
    print()
    print(SEP)
    print(title)
    print(SEP)


def make_households(row_count):
    """Build a reproducible channel with sparse, unique household IDs."""
    prng = np.random.default_rng(seed=12345)
    # Oversampling avoids constructing a one-billion-element population solely
    # to produce stable, non-consecutive chooser identifiers.
    candidates = prng.integers(1_000_000_000, size=row_count + row_count // 100 + 1)
    index = pd.Index(np.unique(candidates)[:row_count], name="household_id")
    if len(index) != row_count:
        raise RuntimeError("failed to generate the requested number of unique IDs")
    return pd.DataFrame({"dummy": 1}, index=index)


def make_manager(channel_type, households):
    """Create a manager with one registered channel and a fixed base seed."""
    rng = Random(channel_type=channel_type)
    rng.set_base_seed(42)
    rng.add_channel("households", households)
    return rng


def elapsed_ms(fn):
    start = time.perf_counter()
    value = fn()
    return (time.perf_counter() - start) * 1_000, value


def timing_samples_ms(fn, number, repeat):
    """Return every per-call time instead of selecting a favorable minimum."""
    samples = timeit.repeat(fn, number=number, repeat=repeat)
    return [sample * 1_000 / number for sample in samples]


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
    """Measure wall time and first calls in fresh, independently compiled workers."""
    context = mp.get_context("spawn")
    tasks = [(channel_type, row_count)] * worker_count
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
        worker_milliseconds = list(executor.map(worker_first_call, tasks))
    wall_milliseconds = (time.perf_counter() - start) * 1_000
    return wall_milliseconds, worker_milliseconds


def warm_compiled_kernels(households):
    """Compile fast kernels before measuring repeatable per-step costs."""
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
    """Measure persistent RNG state plus the index used to address it."""
    channel = rng.get_channel_for_df(households)
    if hasattr(channel, "_state_array"):
        state_bytes = channel._state_array.nbytes
        index_bytes = channel.domain_index.memory_usage(deep=True)
        return (state_bytes + index_bytes) / (1024**2)
    return channel.row_states.memory_usage(index=True, deep=True).sum() / (1024**2)


def tracked_peak_megabytes(fn):
    """Measure peak Python/NumPy allocation visible to tracemalloc."""
    tracemalloc.start()
    try:
        value = fn()
        _, peak = tracemalloc.get_traced_memory()
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


def stable_draw(channel_type, households, requested, positions, step_name):
    """Draw active alternatives from the same stable universe in a fresh manager."""
    active = pd.DataFrame(index=requested.index, columns=np.arange(len(positions)))
    rng = make_manager(channel_type, households)
    rng.begin_step(step_name)
    values = rng.random_for_df_stable_alt_positions(
        active, np.asarray(positions), n_total_alts=32
    )
    rng.end_step(step_name)
    return pd.DataFrame(values, index=requested.index, columns=positions)


def invariance_checks(channel_type, households):
    """Validate replay, chooser ordering, subset, alternative, and reset invariance."""
    tiny = households.iloc[:8]
    checks = []

    first = reproducibility_sequence(channel_type, tiny)
    second = reproducibility_sequence(channel_type, tiny)
    checks.append(
        InvarianceCheck(
            channel_type,
            "fresh-manager mixed sequence",
            all(np.array_equal(a, b) for a, b in zip(first, second, strict=True)),
        )
    )

    positions = [1, 3, 7, 12]
    ordered = stable_draw(channel_type, tiny, tiny, positions, "row_order")
    reversed_rows = stable_draw(
        channel_type, tiny, tiny.iloc[::-1], positions, "row_order"
    )
    checks.append(
        InvarianceCheck(
            channel_type,
            "chooser order",
            np.array_equal(
                ordered.to_numpy(), reversed_rows.loc[tiny.index].to_numpy()
            ),
        )
    )

    subset = stable_draw(channel_type, tiny, tiny.iloc[[1, 4, 6]], positions, "subset")
    full = stable_draw(channel_type, tiny, tiny, positions, "subset")
    checks.append(
        InvarianceCheck(
            channel_type,
            "chooser subset",
            np.array_equal(subset.to_numpy(), full.loc[subset.index].to_numpy()),
        )
    )

    alternatives_a = stable_draw(
        channel_type, tiny, tiny, [1, 3, 7, 12], "alternative_subset"
    )
    alternatives_b = stable_draw(
        channel_type, tiny, tiny, [0, 3, 7, 21], "alternative_subset"
    )
    checks.append(
        InvarianceCheck(
            channel_type,
            "overlapping stable alternatives",
            np.array_equal(
                alternatives_a[[3, 7]].to_numpy(),
                alternatives_b[[3, 7]].to_numpy(),
            ),
        )
    )

    rng = make_manager(channel_type, tiny)
    rng.begin_step("offset_reset")
    initial = rng.random_for_df(tiny, n=5)
    rng.random_for_df(tiny, n=2)
    rng.reset_offsets_for_step("offset_reset")
    replay = rng.random_for_df(tiny, n=5)
    rng.end_step("offset_reset")
    checks.append(
        InvarianceCheck(channel_type, "offset reset", np.array_equal(initial, replay))
    )
    return checks


def package_version(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return "not installed"


def git_revision():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def environment_metadata():
    """Capture enough context to compare artifacts across CI runners."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python": platform.python_version(),
        "packages": {
            name: package_version(name)
            for name in ("activitysim", "numpy", "pandas", "numba")
        },
    }


def print_timing_table(timings):
    section("Timing summary")
    print(
        f"{INDENT}{'Phase':<23} {'Operation':<38} {'Channel':<10} "
        f"{'Mean ± std ms':>22} {'Min ms':>12}"
    )
    print(f"{INDENT}{'-' * 23} {'-' * 38} {'-' * 10} " f"{'-' * 22} {'-' * 12}")
    for item in timings:
        summary = item.summary()
        mean_std = f"{summary['mean_ms']:.3f} ± {summary['std_ms']:.3f}"
        print(
            f"{INDENT}{item.phase:<23} {item.operation:<38} "
            f"{item.channel_type:<10} {mean_std:>22} {summary['min_ms']:12.3f}"
        )


def write_artifacts(output_dir, metadata, config, timings, memory, checks):
    """Write analysis-friendly data and a compact human-readable report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timing_rows = [item.summary() for item in timings]
    csv_path = output_dir / "timings.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(timing_rows[0]))
        writer.writeheader()
        writer.writerows(timing_rows)

    payload = {
        "metadata": metadata,
        "config": config,
        "timings": [asdict(item) | item.summary() for item in timings],
        "memory": [asdict(item) for item in memory],
        "invariance_checks": [asdict(item) for item in checks],
    }
    json_path = output_dir / "results.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# ActivitySim random-channel lifecycle benchmark",
        "",
        f"- Git revision: `{metadata['git_revision']}`",
        f"- Platform: {metadata['platform']}",
        f"- Python: {metadata['python']}",
        f"- Profile: {config['profile']} ({config['rows']:,} rows)",
        "",
        "## Timings",
        "",
        "| Phase | Operation | Channel | Mean ms | Std ms | Min ms |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in timing_rows:
        lines.append(
            f"| {row['phase']} | {row['operation']} | {row['channel_type']} | "
            f"{row['mean_ms']:.3f} | {row['std_ms']:.3f} | {row['min_ms']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Memory",
            "",
            "| Channel | Persistent MiB | Tracemalloc peak MiB |",
            "|---|---:|---:|",
        ]
    )
    for item in memory:
        lines.append(
            f"| {item.channel_type} | {item.persistent_mib:.3f} | "
            f"{item.tracked_peak_mib:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Stream invariants",
            "",
            "| Channel | Check | Result |",
            "|---|---|---:|",
        ]
    )
    for item in checks:
        lines.append(
            f"| {item.channel_type} | {item.check} | "
            f"{'PASS' if item.passed else 'FAIL'} |"
        )
    markdown_path = output_dir / "summary.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, json_path, markdown_path


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
    print(f"{INDENT}Profile: {args.profile}")
    print(f"{INDENT}Rows: {len(households):,}")
    print(f"{INDENT}Warm timing calls/sample: {args.number}")
    print(f"{INDENT}Timing samples: {args.repeat}")
    print(f"{INDENT}Spawned workers/channel: {args.workers or 'disabled'}")
    print(f"{INDENT}Channel types: {', '.join(CHANNEL_TYPES)}")

    section("Observed first call in this process")
    print(
        f"{INDENT}This diagnostic includes Numba compilation encountered in the "
        "displayed order. Use fresh-worker results for cross-channel startup "
        "comparisons."
    )
    for channel_type in CHANNEL_TYPES:
        milliseconds, _ = first_step_draw(
            channel_type, households, households, step_name="process_first_call"
        )
        timings.append(
            Timing(
                "process first call",
                "normal_for_df, full (ordered diagnostic)",
                channel_type,
                len(households),
                1,
                [milliseconds],
            )
        )
        print(f"{INDENT}{channel_type:<10} {milliseconds:12.3f} ms")

    if args.workers:
        section("Fresh spawned-worker startup")
        print(
            f"{INDENT}Each worker imports ActivitySim, creates a channel, and performs "
            "one full-domain normal draw. Worker draw times are independent samples; "
            "wall time also includes spawn, imports, and data construction."
        )
        for channel_type in CHANNEL_TYPES:
            try:
                wall, worker_times = spawned_worker_timing(
                    channel_type, len(households), args.workers
                )
            except OSError as err:
                print(f"{INDENT}Spawned-worker timing is unavailable: {err}")
                break
            timings.extend(
                [
                    Timing(
                        "fresh worker draw",
                        "begin_step + normal_for_df, full",
                        channel_type,
                        len(households),
                        1,
                        worker_times,
                    ),
                    Timing(
                        "spawned worker wall",
                        f"{args.workers} worker(s), aggregate wall",
                        channel_type,
                        len(households),
                        1,
                        [wall],
                    ),
                ]
            )
            print(
                f"{INDENT}{channel_type:<10} wall {wall:12.3f} ms; "
                f"draw mean {np.mean(worker_times):.3f} ms"
            )

    warm_compiled_kernels(households)

    section("Cold per-step initialization")
    print(
        f"{INDENT}Compiled kernels are warm, but every sample creates a new manager "
        "and step. Full registered state is initialized even for three requested rows."
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
            timing = Timing(
                "cold step",
                f"normal_for_df, {label}",
                channel_type,
                len(requested),
                1,
                samples,
            )
            timings.append(timing)
            summary = timing.summary()
            print(
                f"{INDENT}{channel_type:<10} {label:<14} "
                f"{summary['mean_ms']:12.3f} ± {summary['std_ms']:.3f} ms"
            )

    section("Warm generation")
    print(
        f"{INDENT}Each manager is initialized once. gumbel_for_df uses the native "
        "generator in vectorized modes; the baseline transforms one uniform batch."
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
            "uniform + Gumbel transform, 8 draws": lambda rng=rng: -np.log(
                -np.log(rng.random_for_df(households, n=8))
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
            timing = Timing(
                "warm generation",
                operation_name,
                channel_type,
                len(households),
                args.number,
                timing_samples_ms(
                    operations[channel_type][operation_name], args.number, args.repeat
                ),
            )
            timings.append(timing)
            summary = timing.summary()
            print(
                f"{INDENT}{operation_name:<38} {channel_type:<10} "
                f"{summary['mean_ms']:12.3f} ± {summary['std_ms']:.3f} ms"
            )

    section("Memory")
    print(
        f"{INDENT}Persistent memory includes RNG state and its addressing index. "
        "Temporary peak is one warm eight-alternative Gumbel choice as visible to "
        "tracemalloc; untracked native allocations are excluded."
    )
    print(
        f"{INDENT}{'Channel':<10} {'Persistent state MiB':>22} "
        f"{'Tracked peak MiB':>20}"
    )
    memory = []
    for channel_type in CHANNEL_TYPES:
        item = MemoryResult(
            channel_type,
            len(households),
            state_megabytes(managers[channel_type], households),
            tracked_peak_megabytes(
                operations[channel_type]["gumbel choice, 8 alternatives"]
            ),
        )
        memory.append(item)
        print(
            f"{INDENT}{channel_type:<10} {item.persistent_mib:22.3f} "
            f"{item.tracked_peak_mib:20.3f}"
        )

    section("Stream invariants")
    checks = []
    for channel_type in CHANNEL_TYPES:
        channel_checks = invariance_checks(channel_type, households)
        checks.extend(channel_checks)
        for item in channel_checks:
            print(
                f"{INDENT}{channel_type:<10} {item.check:<36} "
                f"{'PASS' if item.passed else 'FAIL'}"
            )
    failed = [item for item in checks if not item.passed]
    if failed:
        raise AssertionError(
            "stream invariance checks failed: "
            + ", ".join(f"{item.channel_type}/{item.check}" for item in failed)
        )

    for rng in managers.values():
        rng.end_step("warm_generation")

    print_timing_table(timings)
    config = {
        "profile": args.profile,
        "rows": args.rows,
        "number": args.number,
        "repeat": args.repeat,
        "workers": args.workers,
    }
    paths = write_artifacts(
        args.output_dir,
        environment_metadata(),
        config,
        timings,
        memory,
        checks,
    )
    section("Artifacts")
    for path in paths:
        print(f"{INDENT}{path}")
    print(
        f"\n{INDENT}Interpret cold and warm timings separately. The three-row cold "
        "result still includes full-domain state initialization."
    )


if __name__ == "__main__":
    main()
