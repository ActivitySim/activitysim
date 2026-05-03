from __future__ import annotations

from typing import Literal

import numba as nb
import numpy as np
import pandas as pd

from ._fast_random import FastGenerator


@nb.njit
def _fold_in_256(key_arr: np.ndarray, data: np.uint64) -> np.ndarray:
    """
    Numba kernel: mixes a uint64 `data` value into each of the four 64-bit
    words of `key_arr` using a MurmurHash3-style finalizer, then applies a
    four-step XOR diffusion pass so that every input bit influences all four
    output words.
    """
    GOLDEN = np.uint64(0x9E3779B97F4A7C15)
    C1 = np.uint64(0xFF51AFD7ED558CCD)
    C2 = np.uint64(0xC4CEB9FE1A85EC53)
    SHIFT = np.uint64(33)

    result = np.empty(4, dtype=np.uint64)
    for i in range(4):
        # Salt each word position uniquely before mixing
        salt = data + np.uint64(i) * GOLDEN
        h = key_arr[i] ^ salt
        # MurmurHash3 64-bit finalizer
        h ^= h >> SHIFT
        h *= C1
        h ^= h >> SHIFT
        h *= C2
        h ^= h >> SHIFT
        result[i] = h

    # Cross-word diffusion: ensure all 256 input bits affect all 256 output bits.
    # Four XOR steps form a simple diffusion network at negligible cost.
    result[1] ^= result[0]
    result[2] ^= result[1]
    result[3] ^= result[2]
    result[0] ^= result[3]

    return result


@nb.njit
def _fold_in_256_batch(key_arr: np.ndarray, data_arr: np.ndarray) -> np.ndarray:
    """
    Batch seeding.

    Generates `len(data_arr)` independent 256-bit child states from a single
    parent `key_arr` state in one sweep.

    Parameters
    ----------
    key_arr : np.ndarray
        A 4-element uint64 array representing the parent 256-bit key.
    data_arr : np.ndarray
        A 1D array of uint64 values to fold into the key. Each value
        will produce one child key.

    Returns
    -------
    results : np.ndarray
        A (N, 4) uint64 array representing the child keys.
    """
    n = data_arr.shape[0]
    results = np.empty((n, 4), dtype=np.uint64)
    for j in range(n):
        results[j] = _fold_in_256(key_arr, data_arr[j])
    return results


def _fast_entropy_raw(base_seeds: int | list[int], index_keys: pd.Index) -> np.ndarray:

    if isinstance(base_seeds, int):
        base_seeds = [base_seeds]
    base_seed_sequence = np.random.SeedSequence(base_seeds)
    base_state = base_seed_sequence.generate_state(4, dtype=np.uint64)
    return _fold_in_256_batch(base_state, np.asarray(index_keys).astype(np.uint64))


_FG_PCG64 = FastGenerator(42, "PCG64")
_FG_SFC64 = FastGenerator(42, "SFC64")


def fast_entropy_PCG64(base_seeds: int | list[int], index_keys: pd.Index) -> np.ndarray:
    generated_states = _fast_entropy_raw(base_seeds, index_keys)
    generated_states[:, -1] |= 1  # Ensure the last word is odd for PCG64
    # make a couple draws to properly mix the state and avoid any initial correlation with the input keys
    _FG_PCG64.vector_random_standard_uniform(generated_states, shape=2)
    return generated_states


def fast_entropy_SFC64(base_seeds: int | list[int], index_keys: pd.Index) -> np.ndarray:
    generated_states = _fast_entropy_raw(base_seeds, index_keys)
    generated_states[:, -1] = 1  # the last word is a counter, start it at 1
    # SFC initialization typically makes a dozen draws to properly mix the state
    _FG_SFC64.vector_random_standard_uniform(generated_states, shape=12)
    return generated_states


def fast_entropy(
    base_seeds: int | list[int],
    index_keys: pd.Index,
    bit_gen: Literal["PCG64", "SFC64"],
) -> np.ndarray:
    if bit_gen == "PCG64":
        return fast_entropy_PCG64(base_seeds, index_keys)
    elif bit_gen == "SFC64":
        return fast_entropy_SFC64(base_seeds, index_keys)
    else:
        raise ValueError("Unsupported bit gen %s" % bit_gen)
