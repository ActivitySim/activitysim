import numpy as np
import pandas as pd
from numba import njit


@njit
def choice_maker(pr, rn, out=None):
    if out is None:
        out = np.empty(pr.shape[0], dtype=np.int32)
    n_alts = pr.shape[1]
    for row in range(pr.shape[0]):
        z = rn[row]
        for col in range(n_alts):
            z = z - pr[row, col]
            if z <= 0:
                out[row] = col
                break
        else:
            # rare condition, only if a random point is greater than 1 (a bug)
            # or if the sum of probabilities is less than 1 and a random point
            # is greater than that sum, which due to the limits of numerical
            # precision can technically happen
            max_pr = 0.0
            for col in range(n_alts):
                if pr[row, col] > max_pr:
                    out[row] = col
                    max_pr = pr[row, col]
    return out


@njit
def sample_choices_maker(
    prob_array,
    random_array,
    alts_array,
    out_choices=None,
    out_choice_probs=None,
):
    """
    Random sample of alternatives.

    Parameters
    ----------
    prob_array : array of float, shape (n_choosers, n_alts)
    random_array : array of float, shape (n_choosers, n_samples)
    alts_array : array of int, shape (n_alts)
    out_choices : array of int, shape (n_samples, n_choosers), optional
    out_choice_probs : array of float, shape (n_samples, n_choosers), optional

    Returns
    -------
    out_choices, out_choice_probs
    """
    n_choosers = random_array.shape[0]
    sample_size = random_array.shape[1]
    n_alts = prob_array.shape[1]
    if out_choices is None:
        out_choices = np.empty((sample_size, n_choosers), dtype=np.int32)
    if out_choice_probs is None:
        out_choice_probs = np.empty((sample_size, n_choosers), dtype=np.float32)

    for c in range(n_choosers):
        random_points = np.sort(random_array[c, :])
        a = 0
        s = 0
        z = 0.0
        for a in range(n_alts):
            z += prob_array[c, a]
            while s < sample_size and z > random_points[s]:
                out_choices[s, c] = alts_array[a]
                out_choice_probs[s, c] = prob_array[c, a]
                s += 1
            if s >= sample_size:
                break
        if s < sample_size:
            # rare condition, only if a random point is greater than 1 (a bug)
            # or if the sum of probabilities is less than 1 and a random point
            # is greater than that sum, which due to the limits of numerical
            # precision can technically happen
            a = n_alts - 1
            while prob_array[c, a] < 1e-30 and a > 0:
                # slip back to the last choice with non-trivial prob
                a -= 1
            while s < sample_size:
                out_choices[s, c] = alts_array[a]
                out_choice_probs[s, c] = prob_array[c, a]
                s += 1

    return out_choices, out_choice_probs


@njit
def _sample_choices_maker_preserve_ordering(
    prob_array,
    random_array,
    alts_array,
    out_choices=None,
    out_choice_probs=None,
):
    """
    Random sample of alternatives.

    Preserves the ordering of the random array, consistent with the legacy
    ActivitySim implementation.

    Parameters
    ----------
    prob_array : array of float, shape (n_choosers, n_alts)
    random_array : array of float, shape (n_choosers, n_samples)
    alts_array : array of int, shape (n_alts)
    out_choices : array of int, shape (n_samples, n_choosers), optional
    out_choice_probs : array of float, shape (n_samples, n_choosers), optional

    Returns
    -------
    out_choices, out_choice_probs
    """
    n_choosers = random_array.shape[0]
    sample_size = random_array.shape[1]
    n_alts = prob_array.shape[1]
    if out_choices is None:
        out_choices = np.empty((sample_size, n_choosers), dtype=np.int32)
    if out_choice_probs is None:
        out_choice_probs = np.empty((sample_size, n_choosers), dtype=np.float32)

    for c in range(n_choosers):
        sorted_random = np.argsort(random_array[c, :])
        a = 0
        s = 0
        z = 0.0
        for a in range(n_alts):
            z += prob_array[c, a]
            while s < sample_size and z > random_array[c, sorted_random[s]]:
                out_choices[sorted_random[s], c] = alts_array[a]
                out_choice_probs[sorted_random[s], c] = prob_array[c, a]
                s += 1
            if s >= sample_size:
                break
        if s < sample_size:
            # rare condition, only if a random point is greater than 1 (a bug)
            # or if the sum of probabilities is less than 1 and a random point
            # is greater than that sum, which due to the limits of numerical
            # precision can technically happen
            a = n_alts - 1
            while prob_array[c, a] < 1e-30 and a > 0:
                # slip back to the last choice with non-trivial prob
                a -= 1
            while s < sample_size:
                out_choices[sorted_random[s], c] = alts_array[a]
                out_choice_probs[sorted_random[s], c] = prob_array[c, a]
                s += 1

    return out_choices, out_choice_probs


def sample_choices_maker_preserve_ordering(
    prob_array,
    random_array,
    alts_array,
    out_choices=None,
    out_choice_probs=None,
):
    """
    Make sample choices.

    Parameters
    ----------
    prob_array : array[float]
    random_array : array[float]
    alts_array : array
    out_choices : array[alts_array.dtype], optional
    out_choice_probs : array[float], optional

    Returns
    -------
    out_choices : array[alts_array.dtype]
    out_choice_probs : array[float]
    """
    if alts_array.dtype.kind != "i":
        # when the alternatives array is not integers (which is unusual, but
        # does happen in the OD choice model) we need to choose integers in
        # numba and then convert those back to whatever dtype the alternatives are
        out_choices_, out_choice_probs = _sample_choices_maker_preserve_ordering(
            prob_array,
            random_array,
            np.arange(alts_array.size),
            out_choices=None,
            out_choice_probs=out_choice_probs,
        )
        if out_choices is not None:
            out_choices[:] = alts_array[out_choices_]
        else:
            out_choices = alts_array[out_choices_]
        return out_choices, out_choice_probs

    return _sample_choices_maker_preserve_ordering(
        prob_array,
        random_array,
        alts_array,
        out_choices=out_choices,
        out_choice_probs=out_choice_probs,
    )
