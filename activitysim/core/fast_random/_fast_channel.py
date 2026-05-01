from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from cffi import FFI

from ._fast_random import FastGenerator

# one more than 0xFFFFFFFF so we can wrap using: int64 % _MAX_SEED
_MAX_SEED = 1 << 32
_SEED_MASK = 0xFFFFFFFF

_FFI = FFI()


def hash32(s):
    """

    Parameters
    ----------
    s: str

    Returns
    -------
        32 bit unsigned hash
    """
    s = s.encode("utf8")
    h = hashlib.md5(s).hexdigest()
    return int(h, base=16) & _SEED_MASK


class FastChannel:
    def __init__(
        self,
        channel_name: str,
        base_seed: int,
        domain_df: pd.DataFrame,
        step_name: str = "",
    ) -> None:
        """
        Create a new FastChannel for vectorised PCG64-based random number generation.

        Each row in *domain_df* gets its own independent PCG64 bit-generator whose
        initial state is derived from the combination of *base_seed*,
        *channel_name*, the current step name, and the row's index value.  This
        guarantees reproducibility across runs while keeping every row's stream
        independent.

        Parameters
        ----------
        channel_name : str
            Unique name for this channel (e.g. ``"households"``, ``"persons"``).
            Hashed into the per-row seed so that different channels produce
            distinct streams even when their domain index values overlap.
        base_seed : int
            Global base seed added to every per-row seed sequence.  Change this
            value to shift all random streams without breaking their relative
            independence.
        domain_df : pandas.DataFrame
            DataFrame whose index defines the set of agents (rows) managed by
            this channel.  The index is copied and stored; the DataFrame columns
            are ignored.
        step_name : str, optional
            If non-empty, ``begin_step(step_name)`` is called immediately after
            construction so the channel is ready to generate numbers straight
            away.  Defaults to ``""`` (no step started).
        """
        self.base_seed = base_seed
        self.channel_name = channel_name
        self.channel_seed = hash32(self.channel_name)
        self.domain_index = domain_df.index[:0].copy()
        self.step_name = None
        self.step_seed = None
        self._fast_generator = FastGenerator()
        self._state_array = None
        self.extend_domain(domain_df)
        if step_name:
            self.begin_step(step_name)

    def extend_domain(self, domain_df: pd.DataFrame) -> None:
        """
        Extend the channel's domain by adding new rows from *domain_df*.

        If a step is currently active, the per-row PCG64 state for the new rows
        is initialised immediately (using the current ``step_seed``) and
        appended to ``self._state_array`` so that random draws can be made for
        the extended rows within the same step.

        The index values of *domain_df* must be disjoint from the channel's
        existing ``domain_index`` so there is no ambiguity / collision between
        rows.

        Parameters
        ----------
        domain_df : pandas.DataFrame
            DataFrame whose index defines the new agents (rows) to add to the
            channel.  Columns are ignored.

        Raises
        ------
        AssertionError
            If any index value in *domain_df* already exists in the channel's
            domain.
        """
        new_index = domain_df.index

        if new_index.empty:
            return

        # new rows must be disjoint from existing domain
        assert (
            len(self.domain_index.intersection(new_index)) == 0
        ), "extend_domain: new domain_df index overlaps existing domain"

        if self.step_name is not None:
            # generate state for the new rows and append to existing state array
            new_state = np.empty(shape=[len(new_index), 4], dtype=np.uint64)
            for n, i in enumerate(new_index):
                ss = np.random.SeedSequence(
                    [self.base_seed, self.channel_seed, self.step_seed, i]
                )
                new_state[n, :] = self._fast_generator.get_state_array(ss)

            if self._state_array is None:
                self._state_array = new_state
            else:
                self._state_array = np.concatenate(
                    [self._state_array, new_state], axis=0
                )

        if len(self.domain_index) == 0:
            self.domain_index = new_index.copy()
        else:
            self.domain_index = self.domain_index.append(new_index)

    def begin_step(self, step_name: str) -> None:
        """
        Initialise (or re-initialise) the per-row PCG64 states for a new step.

        Must be called before any random-number methods are used within a step.
        The method seeds every row's bit-generator from the four-integer sequence
        ``[base_seed, channel_seed, step_seed, row_index]`` via
        :class:`numpy.random.SeedSequence`, ensuring that:

        * the same step always produces the same stream (reproducibility), and
        * different steps produce independent streams (no cross-step correlation).

        Parameters
        ----------
        step_name : str
            Name of the pipeline step being started (e.g. ``"auto_ownership"``).
            Hashed into the seed so that different steps yield distinct streams.

        Raises
        ------
        AssertionError
            If a step is already active (``end_step`` was not called first).
        """

        assert self.step_name is None

        self.step_name = step_name
        self.step_seed = hash32(self.step_name)

        # Seed the bit generators, extracting state along the way
        state_array = np.empty(shape=[len(self.domain_index), 4], dtype=np.uint64)
        for n, i in enumerate(self.domain_index):
            ss = np.random.SeedSequence(
                [self.base_seed, self.channel_seed, self.step_seed, i]
            )
            state_array[n, :] = self._fast_generator.get_state_array(ss)

        self._state_array = state_array

    def end_step(self, step_name: str = "") -> None:
        """
        Tear down the per-row PCG64 states at the end of a step.

        Clears ``step_name``, ``step_seed``, ``_bitgenerator``, and
        ``_state_array`` so that accidental use of stale state after a step
        boundary raises an error rather than silently producing wrong numbers.

        Parameters
        ----------
        step_name : str, optional
            When provided, asserts that the currently active step matches
            *step_name* as a consistency check.  Pass an empty string (the
            default) to skip the check.

        Raises
        ------
        AssertionError
            If *step_name* is provided and does not match the active step name.
        """
        if step_name:
            assert self.step_name == step_name
        self.step_name = None
        self.step_seed = None
        self._state_array = None

    def _check_valid_df(self, df: pd.DataFrame) -> np.ndarray:
        """
        Validate *df* against the channel's domain and return row positions.

        Performs three checks:

        1. *df* has no duplicate index values.
        2. Every index value in *df* exists in the channel's domain index.
        3. A step is currently active (``_state_array`` is not ``None``).

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame whose index is to be validated.  Columns are ignored.

        Returns
        -------
        selected_positions : numpy.ndarray
            1-D integer array of shape ``(len(df),)`` containing the positional
            indices into ``self.domain_index`` (and therefore into
            ``self._state_array``) that correspond to each row of *df*.

        Raises
        ------
        ValueError
            If *df* contains duplicate index values, if any index value is
            absent from the domain, or if no step is currently active.
        """
        # check that df.index has no duplicates
        if len(df.index.unique()) != len(df.index):
            raise ValueError("DataFrame must have unique index")

        selected_positions = self.domain_index.get_indexer(df.index)

        # check that all df.index values were found in self.domain_index
        # (skip the check for empty input – min() on a zero-size array errors)
        if selected_positions.size and selected_positions.min() < 0:
            raise ValueError("DataFrame has index values not found in the domain")

        if self._state_array is None:
            raise ValueError("outside of a defined step")

        return selected_positions

    def normal_for_df(
        self,
        df: pd.DataFrame,
        step_name: str,
        mu: float | np.ndarray = 0,
        sigma: float | np.ndarray = 1,
        lognormal: bool = False,
        size: int | tuple[int, ...] = 1,
    ) -> np.ndarray:
        """
        Draw normal (or lognormal) random variates for each row in *df*.

        Uses the vectorised PCG64 state array to generate standard-normal
        samples, then affinely transforms them with *mu* and *sigma*.
        Successive calls within the same step advance each row's stream
        independently.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame whose index selects which agents receive draws.  Columns
            are ignored.
        step_name : str
            Name of the currently active step; checked for consistency.
        mu : float or array-like, optional
            Mean of the normal distribution.  A scalar is broadcast across all
            rows; an array must have one element per row in *df*.  Defaults to
            ``0``.
        sigma : float or array-like, optional
            Standard deviation of the normal distribution.  Same broadcasting
            rules as *mu*.  Defaults to ``1``.
        lognormal : bool, optional
            When ``True``, return ``exp(normal_sample)`` so that the result
            follows a lognormal distribution with the given underlying-normal
            parameters.  Defaults to ``False``.
        size : int or tuple of int, optional
            Number of draws per agent.  A plain ``int`` *k* yields *k* draws
            per row; a tuple gives the per-row shape.  Defaults to ``1``.

        Returns
        -------
        result : numpy.ndarray
            Array of shape ``(len(df), *size)`` containing the random variates.

        Raises
        ------
        AssertionError
            If *step_name* is ``None`` or does not match the active step.
        ValueError
            If *df* fails the domain validation performed by
            :meth:`_check_valid_df`.
        """
        assert step_name is not None
        assert step_name == self.step_name
        selected_positions = self._check_valid_df(df)
        if size is None:
            size = 1

        mu = np.asarray(mu)
        sigma = np.asarray(sigma)
        result = self._fast_generator.vector_random_standard_normal(
            self._state_array, selected_positions=selected_positions, shape=size
        )
        result = result * sigma + mu
        if lognormal:
            result = np.exp(result)
        return result

    def random_for_df(
        self,
        df: pd.DataFrame,
        step_name: str,
        n: int | tuple[int, ...] = 1,
    ) -> np.ndarray:
        """
        Draw standard-uniform random variates for each row in *df*.

        Uses the vectorised PCG64 state array to generate draws from U[0, 1).
        Successive calls within the same step advance each row's stream
        independently.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame whose index selects which agents receive draws.  Columns
            are ignored.
        step_name : str
            Name of the currently active step; checked for consistency.
        n : int or tuple of int, optional
            Number of draws per agent.  A plain ``int`` *k* yields *k* draws
            per row; a tuple gives the per-row shape.  Defaults to ``1``.

        Returns
        -------
        rands : numpy.ndarray
            Array of shape ``(len(df), *n)`` with values in U[0, 1).

        Raises
        ------
        AssertionError
            If *step_name* is ``None`` or does not match the active step.
        ValueError
            If *df* fails the domain validation performed by
            :meth:`_check_valid_df`.
        """
        assert step_name is not None
        assert step_name == self.step_name
        selected_positions = self._check_valid_df(df)
        return self._fast_generator.vector_random_standard_uniform(
            self._state_array, selected_positions=selected_positions, shape=n
        )

    def choice_for_df(
        self,
        df: pd.DataFrame,
        step_name: str,
        a: int | np.ndarray,
        size: int | tuple[int, ...] = 1,
        replace: bool = False,
    ) -> np.ndarray:
        """
        Apply numpy.random.choice once for each row in df
        using the appropriate random channel for each row.

        Concatenate the the choice arrays for every row into a single 1-D ndarray
        The resulting array will be of length: size * len(df.index)
        This method is designed to support creation of a interaction_dataset

        The columns in df are ignored; the index name and values are used to determine
        which random number sequence to to use.

        Parameters
        ----------
        df : pandas.DataFrame
            df with index name and values corresponding to a registered channel
        step_name : str
            current step name so we can update row_states seed info
        a : 1-D array-like or int
            If an ndarray, a random sample is generated from its elements.
            If an int, the random sample is generated as if a was np.arange(a).
        size : int or tuple of ints
            Output shape (per df row).
        replace : bool
            Whether the sample is with or without replacement.

        Returns
        -------
        choices : 1-D ndarray of length: prod(size) * len(df.index)
            The generated random samples for each row concatenated into a
            single (flat) array.
        """
        assert step_name is not None
        assert step_name == self.step_name
        selected_positions = self._check_valid_df(df)

        # total number of draws required per row
        if isinstance(size, (int, np.integer)):
            total = int(size)
        else:
            total = int(np.prod(size))

        # population to sample from
        if isinstance(a, (int, np.integer)):
            a_arr = np.arange(int(a))
        else:
            a_arr = np.asarray(a)
        n_pop = len(a_arr)

        if replace:
            # draw `total` uniforms per selected row and map to indices in a
            rands = self._fast_generator.vector_random_standard_uniform(
                self._state_array,
                selected_positions=selected_positions,
                shape=total,
            )
            idx = (rands * n_pop).astype(np.int64)
            # guard against the (vanishingly rare) case rands == 1.0 - epsilon edge
            np.minimum(idx, n_pop - 1, out=idx)
            sample = a_arr[idx].reshape(-1)
        else:
            if total > n_pop:
                raise ValueError(
                    "Cannot take a larger sample than population when 'replace=False'"
                )
            # draw n_pop uniforms per selected row; argsort produces a random
            # permutation of [0, n_pop), and we take the first `total` entries.
            rands = self._fast_generator.vector_random_standard_uniform(
                self._state_array,
                selected_positions=selected_positions,
                shape=n_pop,
            )
            order = np.argsort(rands, axis=1)[:, :total]
            sample = a_arr[order].reshape(-1)

        return sample
