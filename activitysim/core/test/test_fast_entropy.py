"""Statistical regression tests for ActivitySim's quick entropy initialization."""

from __future__ import annotations

import numpy as np
import pandas as pd

from activitysim.core.fast_random._entropy import fast_entropy_SFC64
from activitysim.core.fast_random._fast_random import FastGenerator

_BASE_SEEDS = [11, 22, 33]


def test_quick_entropy_state_uniqueness_and_diffusion():
    """Sequential row keys should produce unique, well-diffused SFC64 states."""
    state_count = 100_000
    index = pd.Index(np.arange(state_count, dtype=np.uint64))
    states = fast_entropy_SFC64(_BASE_SEEDS, index)

    # State collisions would make two agents share a random stream.
    assert np.unique(states, axis=0).shape[0] == state_count

    comparison = fast_entropy_SFC64(_BASE_SEEDS, index + np.uint64(1))
    differing_bits = np.unpackbits(
        np.bitwise_xor(states, comparison).view(np.uint8), axis=1
    ).sum(axis=1)

    # SFC64's counter word is initialized identically for every stream, leaving
    # 192 independently mixed bits and an expected Hamming distance of 96.
    assert 93.0 < differing_bits.mean() < 99.0


def test_quick_entropy_interstream_uniformity_and_correlation():
    """Quickly initialized streams should remain uniform and uncorrelated."""
    stream_count = 4_096
    draws_per_stream = 256
    index = pd.Index(np.arange(stream_count, dtype=np.uint64))
    states = fast_entropy_SFC64(_BASE_SEEDS, index)
    generator = FastGenerator(bit_gen="SFC64")
    values = generator.vector_random_standard_uniform(states, shape=draws_per_stream)

    # A 256-bin Pearson statistic provides a deterministic aggregate uniformity
    # regression check. The bounds conservatively bracket the expected value of
    # 255 while still catching obvious initialization bias.
    counts = np.histogram(values, bins=256, range=(0.0, 1.0))[0]
    expected_count = values.size / 256
    chi_square = np.sum((counts - expected_count) ** 2 / expected_count)
    assert 160.0 < chi_square < 350.0

    # Pair disjoint halves of the keyed streams and calculate a Pearson
    # correlation for each pair. This checks cross-stream dependence instead of
    # only the distribution obtained by pooling all streams.
    first, second = np.split(values, 2)
    first = first - first.mean(axis=1, keepdims=True)
    second = second - second.mean(axis=1, keepdims=True)
    correlations = np.sum(first * second, axis=1) / np.sqrt(
        np.sum(first * first, axis=1) * np.sum(second * second, axis=1)
    )
    assert abs(correlations.mean()) < 0.005
    assert np.abs(correlations).mean() < 0.055
    assert np.abs(correlations).max() < 0.25
