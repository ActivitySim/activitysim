"""Benchmark production RNG paths used by stable-alternative and EET choices.

Unlike a fixed-width microbenchmark, this script sweeps chooser count, stable
alternative universe size, and repeated-choice sample size. The reported waste
factor makes explicit when reproducibility requires generating more shocks than
the active utility columns consume.

Run the bounded local profile from the repository root with::

    python other_resources/performance-checks/eet-random-scaling.py

Use --profile full for a broader manual comparison. Results include raw timing
samples, CSV/JSON/Markdown summaries, invariance checks, and plots when
Matplotlib is available.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

from activitysim.core.random import Random

CHANNEL_TYPES = ("simple", "fast", "faster")
OPERATIONS = (
    "stable uniforms",
    "repeated Gumbel max",
    "mapped Gumbel choice",
)
INDENT = "  "
SEP = "=" * 112


@dataclass(frozen=True)
class ScenarioSpec:
    """One production-shaped RNG workload."""

    sweep: str
    name: str
    choosers: int
    active_alternatives: int
    stable_alternatives: int
    sample_size: int
    prior_draws: int = 0


@dataclass
class BenchmarkResult:
    """Timing, work-volume, throughput, and temporary-memory metrics."""

    sweep: str
    scenario: str
    operation: str
    channel_type: str
    choosers: int
    active_alternatives: int
    stable_alternatives: int
    sample_size: int
    prior_draws: int
    generated_shocks: int
    useful_shocks: int
    waste_factor: float
    dense_shock_volume_mib: float
    number: int
    samples_ms: list[float]
    mean_ms: float
    median_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    useful_shocks_per_second: float
    generated_shocks_per_second: float
    tracked_peak_mib: float

    def csv_row(self):
        row = asdict(self)
        del row["samples_ms"]
        return row


@dataclass
class InvarianceCheck:
    """One deterministic-stream property exercised through production APIs."""

    channel_type: str
    check: str
    passed: bool


def profile_scenarios(profile):
    """Return bounded sweeps that isolate the three main scaling dimensions."""
    if profile == "quick":
        scenarios = [
            ScenarioSpec("sparsity", "stable-32", 128, 32, 32, 4),
            ScenarioSpec("sparsity", "stable-128", 128, 32, 128, 4),
            ScenarioSpec("sparsity", "stable-512", 128, 32, 512, 4),
            ScenarioSpec("sparsity", "stable-2048", 128, 32, 2_048, 4),
            ScenarioSpec("choosers", "choosers-32", 32, 32, 256, 4),
            ScenarioSpec("choosers", "choosers-256", 256, 32, 256, 4),
            ScenarioSpec("choosers", "choosers-1024", 1_024, 32, 256, 4),
            ScenarioSpec("samples", "samples-1", 128, 32, 256, 1),
            ScenarioSpec("samples", "samples-8", 128, 32, 256, 8),
            ScenarioSpec("offset", "prior-0", 128, 32, 256, 4, prior_draws=0),
            ScenarioSpec("offset", "prior-256", 128, 32, 256, 4, prior_draws=256),
            ScenarioSpec("offset", "prior-2048", 128, 32, 256, 4, prior_draws=2_048),
        ]
        return scenarios, 3, 1

    scenarios = [
        ScenarioSpec("sparsity", "stable-64", 250, 64, 64, 4),
        ScenarioSpec("sparsity", "stable-256", 250, 64, 256, 4),
        ScenarioSpec("sparsity", "stable-1024", 250, 64, 1_024, 4),
        ScenarioSpec("sparsity", "stable-4096", 250, 64, 4_096, 4),
        ScenarioSpec("sparsity", "stable-8192", 250, 64, 8_192, 4),
        ScenarioSpec("choosers", "choosers-250", 250, 64, 512, 4),
        ScenarioSpec("choosers", "choosers-1000", 1_000, 64, 512, 4),
        ScenarioSpec("choosers", "choosers-5000", 5_000, 64, 512, 4),
        ScenarioSpec("samples", "samples-1", 500, 64, 512, 1),
        ScenarioSpec("samples", "samples-8", 500, 64, 512, 8),
        ScenarioSpec("samples", "samples-16", 500, 64, 512, 16),
        ScenarioSpec("offset", "prior-0", 250, 64, 512, 4, prior_draws=0),
        ScenarioSpec("offset", "prior-512", 250, 64, 512, 4, prior_draws=512),
        ScenarioSpec("offset", "prior-4096", 250, 64, 512, 4, prior_draws=4_096),
        ScenarioSpec("offset", "prior-16384", 250, 64, 512, 4, prior_draws=16_384),
    ]
    return scenarios, 5, 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=("quick", "full"), default="quick", help="workload preset"
    )
    parser.add_argument("--repeat", type=int, help="timing samples per result")
    parser.add_argument("--number", type=int, help="calls per timing sample")
    parser.add_argument(
        "--channels",
        default=",".join(CHANNEL_TYPES),
        help="comma-separated subset of simple,fast,faster",
    )
    parser.add_argument(
        "--max-scenarios",
        type=int,
        help="run only the first N scenarios (useful for smoke tests)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/performance-checks/eet-random-scaling"),
        help="directory for benchmark artifacts",
    )
    parser.add_argument(
        "--skip-plots", action="store_true", help="do not create Matplotlib plots"
    )
    args = parser.parse_args()
    scenarios, default_repeat, default_number = profile_scenarios(args.profile)
    if args.repeat is None:
        args.repeat = default_repeat
    if args.number is None:
        args.number = default_number
    if args.max_scenarios is not None and args.max_scenarios < 1:
        parser.error("--max-scenarios must be positive")
    args.scenarios = scenarios[: args.max_scenarios]
    args.channels = tuple(
        item.strip() for item in args.channels.split(",") if item.strip()
    )
    unknown = set(args.channels) - set(CHANNEL_TYPES)
    if unknown:
        parser.error(f"unknown channel type(s): {sorted(unknown)}")
    if not args.channels:
        parser.error("--channels must select at least one channel type")
    return args


def section(title):
    print()
    print(SEP)
    print(title)
    print(SEP)


def make_choosers(row_count):
    """Create sparse, deterministic chooser identifiers."""
    index = pd.Index(
        np.arange(1, row_count + 1, dtype=np.int64) * 7_919,
        name="household_id",
    )
    return pd.DataFrame({"dummy": 1}, index=index)


def make_inputs(scenario):
    """Build aligned utilities and stable-position mappings for a scenario."""
    choosers = make_choosers(scenario.choosers)
    positions = np.linspace(
        0,
        scenario.stable_alternatives - 1,
        scenario.active_alternatives,
        dtype=np.int64,
    )
    utility_row = np.linspace(-2.0, 2.0, scenario.active_alternatives)
    utilities = pd.DataFrame(
        np.tile(utility_row, (scenario.choosers, 1)),
        index=choosers.index,
    )
    alt_nrs = pd.DataFrame(
        np.tile(positions, (scenario.choosers, 1)),
        index=choosers.index,
        columns=utilities.columns,
    )
    return choosers, utilities, alt_nrs, positions


def make_manager(channel_type, choosers, step_name):
    """Initialize one production Random manager before the timed operation."""
    rng = Random(channel_type=channel_type)
    rng.set_base_seed(42)
    rng.add_channel("households", choosers)
    rng.begin_step(step_name)
    return rng


def operation_callable(operation, rng, scenario, utilities, alt_nrs, positions):
    """Bind one of the production APIs without including setup in its timing."""
    if operation == "stable uniforms":
        return lambda: rng.random_for_df_stable_alt_positions(
            utilities,
            stable_alt_positions=positions,
            n_total_alts=scenario.stable_alternatives,
        )
    if operation == "repeated Gumbel max":
        return lambda: rng.gumbel_max_positions_for_df(
            utilities,
            sample_size=scenario.sample_size,
            stable_alt_positions=positions,
            n_total_alts=scenario.stable_alternatives,
        )
    if operation == "mapped Gumbel choice":
        return lambda: rng.gumbel_choice_positions_for_df(
            utilities,
            alt_nrs_df=alt_nrs,
            n_rands=scenario.stable_alternatives,
        )
    raise ValueError(f"unknown operation {operation!r}")


def shock_counts(operation, scenario):
    """Return generated and consumed shock counts for a single call."""
    multiplier = scenario.sample_size if operation == "repeated Gumbel max" else 1
    generated = scenario.choosers * scenario.stable_alternatives * multiplier
    useful = scenario.choosers * scenario.active_alternatives * multiplier
    return generated, useful


def prepare_stream_position(rng, step_name, choosers, prior_draws):
    """Reset a stream and advance it outside the measured interval."""
    rng.reset_offsets_for_step(step_name)
    # Draw and reset one value to materialize lazily seeded vectorized state at
    # offset zero. This keeps robust/quick per-step initialization outside the
    # warm-operation measurement even when the controlled offset is zero.
    initialized_state = rng.random_for_df(choosers, n=1)
    del initialized_state
    rng.reset_offsets_for_df(choosers)
    if prior_draws:
        prior_values = rng.random_for_df(choosers, n=prior_draws)
        del prior_values


def tracked_peak_megabytes(rng, step_name, choosers, prior_draws, fn):
    """Measure allocations visible to tracemalloc for a reset warm call."""
    prepare_stream_position(rng, step_name, choosers, prior_draws)
    tracemalloc.start()
    try:
        value = fn()
        _, peak = tracemalloc.get_traced_memory()
        del value
    finally:
        tracemalloc.stop()
    return peak / (1024**2)


def benchmark_operation(channel_type, scenario, operation, repeat, number):
    """Measure one warm production operation with identical offsets per sample."""
    choosers, utilities, alt_nrs, positions = make_inputs(scenario)
    step_name = "eet_scaling"
    rng = make_manager(channel_type, choosers, step_name)
    fn = operation_callable(operation, rng, scenario, utilities, alt_nrs, positions)

    # This untimed call compiles a fast kernel if needed. Each measured sample
    # starts at the same controlled offset. SimpleChannel must replay prior draws
    # inside its timed operation, while vectorized channels retain advanced state.
    prepare_stream_position(rng, step_name, choosers, scenario.prior_draws)
    fn()
    samples = []
    for _ in range(repeat):
        prepare_stream_position(rng, step_name, choosers, scenario.prior_draws)
        start = time.perf_counter()
        for _ in range(number):
            fn()
        samples.append((time.perf_counter() - start) * 1_000 / number)

    peak_mib = tracked_peak_megabytes(
        rng, step_name, choosers, scenario.prior_draws, fn
    )
    rng.end_step(step_name)
    samples_array = np.asarray(samples)
    generated, useful = shock_counts(operation, scenario)
    mean_ms = float(samples_array.mean())
    elapsed_seconds = mean_ms / 1_000
    return BenchmarkResult(
        sweep=scenario.sweep,
        scenario=scenario.name,
        operation=operation,
        channel_type=channel_type,
        choosers=scenario.choosers,
        active_alternatives=scenario.active_alternatives,
        stable_alternatives=scenario.stable_alternatives,
        sample_size=scenario.sample_size,
        prior_draws=scenario.prior_draws,
        generated_shocks=generated,
        useful_shocks=useful,
        waste_factor=generated / useful,
        dense_shock_volume_mib=generated * np.dtype(np.float64).itemsize / (1024**2),
        number=number,
        samples_ms=samples,
        mean_ms=mean_ms,
        median_ms=float(np.median(samples_array)),
        std_ms=float(samples_array.std()),
        min_ms=float(samples_array.min()),
        max_ms=float(samples_array.max()),
        useful_shocks_per_second=useful / elapsed_seconds,
        generated_shocks_per_second=generated / elapsed_seconds,
        tracked_peak_mib=peak_mib,
    )


def warm_compiled_kernels():
    """Compile every production fast-channel path with a tiny workload."""
    scenario = ScenarioSpec("warmup", "warmup", 8, 4, 16, 2)
    for channel_type in ("fast", "faster"):
        choosers, utilities, alt_nrs, positions = make_inputs(scenario)
        rng = make_manager(channel_type, choosers, "eet_warmup")
        for operation in OPERATIONS:
            operation_callable(
                operation, rng, scenario, utilities, alt_nrs, positions
            )()
        rng.end_step("eet_warmup")


def production_sequence(channel_type, choosers, requested, step_name):
    """Exercise all three APIs in sequence for replay and row-order checks."""
    positions = np.array([1, 4, 7, 11], dtype=np.int64)
    utilities = pd.DataFrame(
        np.tile(np.linspace(-1.0, 1.0, len(positions)), (len(requested), 1)),
        index=requested.index,
    )
    alt_nrs = pd.DataFrame(
        np.tile(positions, (len(requested), 1)),
        index=requested.index,
        columns=utilities.columns,
    )
    rng = make_manager(channel_type, choosers, step_name)
    values = (
        rng.random_for_df_stable_alt_positions(utilities, positions, 16),
        rng.gumbel_max_positions_for_df(
            utilities, 3, stable_alt_positions=positions, n_total_alts=16
        ),
        rng.gumbel_choice_positions_for_df(utilities, alt_nrs, 16),
    )
    rng.end_step(step_name)
    return values


def invariance_checks(channel_type):
    """Check production EET replay, chooser order/subset, and offset reset."""
    choosers = make_choosers(8)
    first = production_sequence(channel_type, choosers, choosers, "eet_replay")
    second = production_sequence(channel_type, choosers, choosers, "eet_replay")
    checks = [
        InvarianceCheck(
            channel_type,
            "production EET replay",
            all(np.array_equal(a, b) for a, b in zip(first, second, strict=True)),
        )
    ]

    reversed_values = production_sequence(
        channel_type, choosers, choosers.iloc[::-1], "eet_order"
    )
    ordered_values = production_sequence(channel_type, choosers, choosers, "eet_order")
    reverse_index = np.arange(len(choosers) - 1, -1, -1)
    checks.append(
        InvarianceCheck(
            channel_type,
            "production EET chooser order",
            all(
                np.array_equal(ordered, reversed_value[reverse_index])
                for ordered, reversed_value in zip(
                    ordered_values, reversed_values, strict=True
                )
            ),
        )
    )

    subset_rows = choosers.iloc[[1, 4, 6]]
    subset_values = production_sequence(
        channel_type, choosers, subset_rows, "eet_subset"
    )
    full_values = production_sequence(channel_type, choosers, choosers, "eet_subset")
    subset_positions = choosers.index.get_indexer(subset_rows.index)
    checks.append(
        InvarianceCheck(
            channel_type,
            "production EET chooser subset",
            all(
                np.array_equal(subset, full[subset_positions])
                for subset, full in zip(subset_values, full_values, strict=True)
            ),
        )
    )

    utilities = pd.DataFrame(0.0, index=choosers.index, columns=np.arange(4))
    rng = make_manager(channel_type, choosers, "eet_reset")
    initial = rng.random_for_df_stable_alt_positions(
        utilities, np.array([1, 3, 7, 12]), 16
    )
    rng.gumbel_max_positions_for_df(
        utilities,
        2,
        stable_alt_positions=np.array([1, 3, 7, 12]),
        n_total_alts=16,
    )
    rng.reset_offsets_for_step("eet_reset")
    replay = rng.random_for_df_stable_alt_positions(
        utilities, np.array([1, 3, 7, 12]), 16
    )
    rng.end_step("eet_reset")
    checks.append(
        InvarianceCheck(
            channel_type, "production EET offset reset", np.array_equal(initial, replay)
        )
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
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python": platform.python_version(),
        "packages": {
            name: package_version(name)
            for name in ("activitysim", "numpy", "pandas", "numba", "matplotlib")
        },
    }


def write_csv(path, results):
    rows = [result.csv_row() for result in results]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, metadata, config, results, checks):
    """Create a reviewable summary that retains every scenario result."""
    lines = [
        "# ActivitySim EET and stable-alternative RNG scaling",
        "",
        f"- Git revision: {metadata['git_revision']}",
        f"- Platform: {metadata['platform']}",
        f"- Python: {metadata['python']}",
        f"- Profile: {config['profile']}",
        "",
        "The waste factor is generated shocks divided by shocks attached to active",
        "utility columns. Dense shock volume is the equivalent float64 volume;",
        "row-wise implementations need not allocate it all at once.",
        "",
        "## Results",
        "",
        "| Sweep | Scenario | Operation | Channel | Prior draws | Waste | Mean ms | Std ms | Useful M shocks/s | Peak MiB |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.sweep} | {result.scenario} | {result.operation} | "
            f"{result.channel_type} | {result.prior_draws:,} | "
            f"{result.waste_factor:.1f}× | "
            f"{result.mean_ms:.3f} | {result.std_ms:.3f} | "
            f"{result.useful_shocks_per_second / 1_000_000:.3f} | "
            f"{result.tracked_peak_mib:.3f} |"
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
    for check in checks:
        lines.append(
            f"| {check.channel_type} | {check.check} | "
            f"{'PASS' if check.passed else 'FAIL'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_results(output_dir, results):
    """Plot time and useful throughput against stable-universe size."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"{INDENT}Matplotlib is unavailable; skipping plots")
        return []

    subset = [result for result in results if result.sweep == "sparsity"]
    paths = []
    metrics = (
        ("mean_ms", "Mean time (ms)", "stable-universe-time.png"),
        (
            "useful_shocks_per_second",
            "Useful shocks / second",
            "stable-universe-useful-throughput.png",
        ),
    )
    for metric, ylabel, filename in metrics:
        figure, axes = plt.subplots(1, len(OPERATIONS), figsize=(15, 4.5), sharex=True)
        for axis, operation in zip(axes, OPERATIONS, strict=True):
            for channel_type in CHANNEL_TYPES:
                series = sorted(
                    (
                        item
                        for item in subset
                        if item.operation == operation
                        and item.channel_type == channel_type
                    ),
                    key=lambda item: item.stable_alternatives,
                )
                if not series:
                    continue
                axis.plot(
                    [item.stable_alternatives for item in series],
                    [getattr(item, metric) for item in series],
                    marker="o",
                    label=channel_type,
                )
            axis.set_title(operation)
            axis.set_xscale("log", base=2)
            axis.set_yscale("log")
            axis.set_xlabel("Stable alternatives")
            axis.grid(True, which="both", alpha=0.3)
        axes[0].set_ylabel(ylabel)
        axes[-1].legend()
        figure.suptitle("RNG scaling as the stable alternative universe grows")
        figure.tight_layout()
        path = output_dir / filename
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(path)

    offset_results = [result for result in results if result.sweep == "offset"]
    figure, axes = plt.subplots(1, len(OPERATIONS), figsize=(15, 4.5), sharex=True)
    for axis, operation in zip(axes, OPERATIONS, strict=True):
        for channel_type in CHANNEL_TYPES:
            series = sorted(
                (
                    item
                    for item in offset_results
                    if item.operation == operation and item.channel_type == channel_type
                ),
                key=lambda item: item.prior_draws,
            )
            if not series:
                continue
            axis.plot(
                [item.prior_draws for item in series],
                [item.mean_ms for item in series],
                marker="o",
                label=channel_type,
            )
        axis.set_title(operation)
        axis.set_xscale("symlog", base=2, linthresh=1)
        axis.set_yscale("log")
        axis.set_xlabel("Prior draws per chooser")
        axis.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("Mean time (ms)")
    axes[-1].legend()
    figure.suptitle("RNG cost after advancing each chooser stream")
    figure.tight_layout()
    path = output_dir / "prior-offset-time.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    paths.append(path)
    return paths


def write_artifacts(output_dir, metadata, config, results, checks, skip_plots):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "results.csv"
    json_path = output_dir / "results.json"
    markdown_path = output_dir / "summary.md"
    write_csv(csv_path, results)
    json_path.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "config": config,
                "results": [asdict(result) for result in results],
                "invariance_checks": [asdict(check) for check in checks],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_markdown(markdown_path, metadata, config, results, checks)
    paths = [csv_path, json_path, markdown_path]
    if not skip_plots:
        paths.extend(plot_results(output_dir, results))
    return paths


def main():
    args = parse_args()
    if args.repeat < 1 or args.number < 1:
        raise ValueError("--repeat and --number must be positive")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    section("Setup")
    print(f"{INDENT}Profile: {args.profile}")
    print(f"{INDENT}Scenarios: {len(args.scenarios)}")
    print(f"{INDENT}Channels: {', '.join(args.channels)}")
    print(f"{INDENT}Timing samples: {args.repeat}; calls/sample: {args.number}")
    print(
        f"{INDENT}Operations use ActivitySim Random directly; setup, state "
        "initialization, and offset resets are outside measured intervals."
    )

    warm_compiled_kernels()
    results = []
    for scenario in args.scenarios:
        section(
            f"{scenario.sweep}: {scenario.name} — {scenario.choosers:,} choosers, "
            f"{scenario.active_alternatives} active / "
            f"{scenario.stable_alternatives:,} stable, sample {scenario.sample_size}"
            f", prior draws {scenario.prior_draws:,}"
        )
        for operation in OPERATIONS:
            for channel_type in args.channels:
                result = benchmark_operation(
                    channel_type, scenario, operation, args.repeat, args.number
                )
                results.append(result)
                print(
                    f"{INDENT}{operation:<24} {channel_type:<8} "
                    f"{result.mean_ms:10.3f} ± {result.std_ms:8.3f} ms; "
                    f"waste {result.waste_factor:6.1f}×; "
                    f"useful {result.useful_shocks_per_second / 1_000_000:8.3f} M/s"
                )

    section("Production stream invariants")
    checks = []
    for channel_type in args.channels:
        channel_checks = invariance_checks(channel_type)
        checks.extend(channel_checks)
        for check in channel_checks:
            print(
                f"{INDENT}{channel_type:<8} {check.check:<36} "
                f"{'PASS' if check.passed else 'FAIL'}"
            )
    failed = [check for check in checks if not check.passed]
    if failed:
        raise AssertionError(
            "stream invariance checks failed: "
            + ", ".join(f"{item.channel_type}/{item.check}" for item in failed)
        )

    config = {
        "profile": args.profile,
        "repeat": args.repeat,
        "number": args.number,
        "channels": args.channels,
        "scenarios": [asdict(scenario) for scenario in args.scenarios],
    }
    paths = write_artifacts(
        args.output_dir,
        environment_metadata(),
        config,
        results,
        checks,
        args.skip_plots,
    )
    section("Artifacts")
    for path in paths:
        print(f"{INDENT}{path}")


if __name__ == "__main__":
    main()
