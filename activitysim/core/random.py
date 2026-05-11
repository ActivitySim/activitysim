# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import hashlib
import logging
from builtins import object, range

import numpy as np
import pandas as pd

from activitysim.core.exceptions import DuplicateLoadableObjectError, TableIndexError
from activitysim.core.util import reindex

from .tracing import print_elapsed_time

logger = logging.getLogger(__name__)

# one more than 0xFFFFFFFF so we can wrap using: int64 % _MAX_SEED
_MAX_SEED = 1 << 32
_SEED_MASK = 0xFFFFFFFF


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


class SimpleChannel(object):
    """

    We need to ensure that we generate the same random streams (when re-run or even across
    different simulations.) We do this by generating a random seed for each domain_df row
    that is based on the domain_df index (which implies that generated tables like tours
    and trips are also created with stable, predictable, repeatable row indexes.

    Because we need to generate a distinct stream for each step, we can't just use the
    domain_df index - we need a strategy for handling multiple steps without generating
    collisions between streams (i.e. choosing the same seed for more than one stream.)

    The easiest way to do this would be to use an array of integers to seed the generator,
    with a global seed, a channel seed, a row seed, and a step seed. Unfortunately, seeding
    numpy RandomState with arrays is a LOT slower than with a single integer seed, and
    speed matters because we reseed on-the-fly for every call because creating a different
    RandomState object for each row uses too much memory (5K per RandomState object)

    numpy random seeds are unsigned int32 so there are 4,294,967,295 available seeds.
    That is probably just about enough to distribute evenly, for most cities, depending on the
    number of households, persons, tours, trips, and steps.

    So we use (global_seed + channel_seed + step_seed + row_index) % (1 << 32)
    to get an int32 seed rather than a tuple.

    We do read in the whole households and persons tables at start time, so we could note the
    max index values. But we might then want a way to ensure stability between the test, example,
    and full datasets. I am punting on this for now.
    """

    def __init__(self, channel_name, base_seed, domain_df, step_name):
        self.base_seed = base_seed

        # ensure that every channel is different, even for the same df index values and max_steps
        self.channel_name = channel_name
        self.channel_seed = hash32(self.channel_name)

        self.step_name = None
        self.step_seed = None
        self.row_states = None

        # create dataframe to hold state for every df row
        self.extend_domain(domain_df)
        assert self.row_states.shape[0] == domain_df.shape[0]

        if step_name:
            self.begin_step(step_name)

    def init_row_states_for_step(self, row_states):
        """
        initialize row states (in place) for new step

        with stable, predictable, repeatable row_seeds for that domain_df index value

        See notes on the seed generation strategy in class comment above.

        Parameters
        ----------
        row_states
        """

        assert self.step_name

        if self.step_name and not row_states.empty:
            row_states["row_seed"] = (
                self.base_seed + self.channel_seed + self.step_seed + row_states.index
            ) % _MAX_SEED

            # number of rands pulled this step
            row_states["offset"] = 0

        return row_states

    def extend_domain(self, domain_df):
        """
        Extend or create row_state df by adding seed info for each row in domain_df

        If extending, the index values of new tables must be disjoint so
        there will be no ambiguity/collisions between rows

        Parameters
        ----------
        domain_df : pandas.DataFrame
            domain dataframe with index values for which random streams are to be generated
            and well-known index name corresponding to the channel
        """

        if domain_df.empty:
            logger.warning(
                "extend_domain for channel %s for empty domain_df" % self.channel_name
            )

        # dataframe to hold state for every df row
        row_states = pd.DataFrame(columns=["row_seed", "offset"], index=domain_df.index)

        if self.step_name and not row_states.empty:
            self.init_row_states_for_step(row_states)

        if self.row_states is None:
            self.row_states = row_states
        else:
            # row_states already exists, so we are extending
            # if extending, these should be new rows, no intersection with existing row_states
            assert len(self.row_states.index.intersection(domain_df.index)) == 0
            self.row_states = pd.concat([self.row_states, row_states])

    def begin_step(self, step_name):
        """
        Reset channel state for a new state

        Parameters
        ----------
        step_name : str
            pipeline step name for this step
        """

        assert self.step_name is None

        self.step_name = step_name
        self.step_seed = hash32(self.step_name)

        self.init_row_states_for_step(self.row_states)

        # standard constant to use for choice_for_df instead of fast-forwarding rand stream
        self.multi_choice_offset = None

    def end_step(self, step_name):
        assert self.step_name == step_name

        self.step_name = None
        self.step_seed = None
        self.row_states["offset"] = 0
        self.row_states["row_seed"] = 0

    def _generators_for_df(self, df):
        """
        Python generator function for iterating over numpy prngs (nomenclature collision!)
        seeded and fast-forwarded on-the-fly to the appropriate position in the channel's
        random number stream for each row in df.

        WARNING:
            since we are reusing a single underlying randomstate,
            prng must be called when yielded as generated sequence,
            not serialized and called later after iterator finishes

        Parameters
        ----------
        df : pandas.DataFrame
            dataframe with index values for which random streams are to be generated
            and well-known index name corresponding to the channel
        """

        # assert no dupes
        assert len(df.index.unique()) == len(df.index)

        df_row_states = self.row_states.loc[df.index]

        # https://numpy.org/doc/stable/reference/random/generator.html
        # np.random.default_rng()
        prng = np.random.RandomState()
        for row in df_row_states.itertuples():
            prng.seed(row.row_seed)

            if row.offset:
                # consume rands
                prng.rand(row.offset)

            yield prng

    def random_for_df(self, df, step_name, n=1):
        """
        Return n floating point random numbers in range [0, 1) for each row in df
        using the appropriate random channel for each row.

        Subsequent calls (in the same step) will return the next rand for each df row

        The resulting array will be the same length (and order) as df
        This method is designed to support alternative selection from a probability array

        The columns in df are ignored; the index name and values are used to determine
        which random number sequence to to use.

        If "true pseudo random" behavior is desired (i.e. NOT repeatable) the set_base_seed
        method (q.v.) may be used to globally reseed all random streams.

        Parameters
        ----------
        df : pandas.DataFrame
            df with index name and values corresponding to a registered channel

        n : int
            number of rands desired per df row

        Returns
        -------
        rands : 2-D ndarray
            array the same length as df, with n floats in range [0, 1) for each df row
        """

        assert self.step_name
        assert self.step_name == step_name

        # - reminder: prng must be called when yielded as generated sequence, not serialized
        generators = self._generators_for_df(df)

        rands = np.asanyarray([prng.rand(n) for prng in generators])
        # update offset for rows we handled
        self.row_states.loc[df.index, "offset"] += n
        return rands

    def random_for_df_stable_alt_positions(
        self,
        df,
        step_name,
        stable_alt_positions,
        n_total_alts,
    ):
        """
        Return one uniform draw per stable-universe alternative and chooser row,
        then project to the active alternative positions.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with one row per chooser and one column per active alternative.
        stable_alt_positions : 1-D ndarray
            Mapping from active columns in `df` to positions in the larger stable
            alternative universe.
        n_total_alts : int
            Number of alternatives in the larger stable universe.

        Returns
        -------
        rands : 2-D ndarray
            Array with shape `(len(df), df.shape[1])` containing uniforms aligned to
            the active alternatives.
        """

        assert self.step_name
        assert self.step_name == step_name

        n_alts = df.shape[1]
        stable_alt_positions = np.asarray(stable_alt_positions)
        if stable_alt_positions.shape != (n_alts,):
            raise ValueError(
                "stable_alt_positions must be a 1-D array aligned to df columns"
            )
        if stable_alt_positions.min() < 0 or stable_alt_positions.max() >= n_total_alts:
            raise ValueError(
                "stable_alt_positions values must be within [0, n_total_alts)"
            )

        generators = self._generators_for_df(df)
        rands = np.asanyarray(
            [prng.rand(n_total_alts)[stable_alt_positions] for prng in generators]
        )
        self.row_states.loc[df.index, "offset"] += n_total_alts
        return rands

    def gumbel_for_df(self, df, step_name, n=1):
        """
        Return n floating point gumbel-distributed numbers for each row in df
        using the appropriate random channel for each row.

        Subsequent calls (in the same step) will return the next rand for each df row

        The resulting array will be the same length (and order) as df
        This method is designed to support alternative selection from a probability array

        The columns in df are ignored; the index name and values are used to determine
        which random number sequence to to use.

        If "true pseudo random" behavior is desired (i.e. NOT repeatable) the set_base_seed
        method (q.v.) may be used to globally reseed all random streams.

        Parameters
        ----------
        df : pandas.DataFrame
            df with index name and values corresponding to a registered channel

        n : int
            number of rands desired per df row

        Returns
        -------
        rands : 2-D ndarray
            array the same length as df, with n floats in range [0, 1) for each df row
        """

        assert self.step_name
        assert self.step_name == step_name

        # - reminder: prng must be called when yielded as generated sequence, not serialized
        generators = self._generators_for_df(df)

        # rands = np.asanyarray([prng.gumbel(size=n) for prng in generators])
        # this is about 20% faster for large arrays, like for destination choice
        rands = np.asanyarray([-np.log(-np.log(prng.rand(n))) for prng in generators])

        # update offset for rows we handled
        self.row_states.loc[df.index, "offset"] += n
        return rands

    def gumbel_max_positions_for_df(
        self,
        utilities,
        step_name,
        sample_size,
        stable_alt_positions=None,
        n_total_alts=None,
    ):
        """
        Return the winning alternative position for each chooser/sample pair
        without materializing the full chooser-by-alternative-by-sample Gumbel array.

        Parameters
        ----------
        utilities : pandas.DataFrame
            DataFrame with one row per chooser and one column per alternative.
        sample_size : int
            Number of repeated sampled choices to make per chooser.
        stable_alt_positions : 1-D ndarray, optional
            Mapping from active utility columns to positions in a larger stable
            alternative universe.
        n_total_alts : int, optional
            Number of alternatives in the larger stable universe.

        Returns
        -------
        positions : 2-D ndarray of int32
            Array with shape (len(utilities), sample_size) containing the column
            position of the winning alternative for each chooser/sample pair.
        """

        assert self.step_name
        assert self.step_name == step_name

        utility_values = utilities.to_numpy()
        n_rows, n_alts = utility_values.shape
        positions = np.empty((n_rows, sample_size), dtype=np.int32)

        if stable_alt_positions is not None or n_total_alts is not None:
            if stable_alt_positions is None or n_total_alts is None:
                raise ValueError(
                    "stable_alt_positions and n_total_alts must both be provided or omitted together"
                )
            stable_alt_positions = np.asarray(stable_alt_positions)
            if stable_alt_positions.shape != (n_alts,):
                raise ValueError(
                    "stable_alt_positions must be a 1-D array aligned to utilities columns"
                )
            if stable_alt_positions.min() < 0 or stable_alt_positions.max() >= n_total_alts:
                raise ValueError(
                    "stable_alt_positions values must be within [0, n_total_alts)"
                )
            n_gumbels = n_total_alts
        else:
            n_gumbels = n_alts

        generators = self._generators_for_df(utilities)

        # for each chooser, generate the error terms for all samples at once. reshaping this
        # in (default) C order means that the the first n_alts values are the gumbels for the
        # first sample, the next n_alts values are the gumbels for the second sample, etc.
        for row_num, prng in enumerate(generators):
            utility_row = utility_values[row_num]
            row_gumbels = -np.log(-np.log(prng.rand(n_gumbels * sample_size))).reshape(
                (sample_size, n_gumbels)
            )
            if stable_alt_positions is not None:
                row_gumbels = row_gumbels[:, stable_alt_positions]
            positions[row_num, :] = np.argmax(
                row_gumbels + utility_row[np.newaxis, :],
                axis=1,
            )

        self.row_states.loc[utilities.index, "offset"] += n_gumbels * sample_size
        return positions

    def gumbel_choice_positions_for_df(
        self,
        utilities,
        step_name,
        alt_nrs_df=None,
        n_rands=None,
    ):
        """
        Return the winning alternative position for each chooser row without
        materializing the utility-plus-error table.

        Parameters
        ----------
        utilities : pandas.DataFrame
            DataFrame with one row per chooser and one column per available alternative.
        alt_nrs_df : pandas.DataFrame, optional
            DataFrame aligned to `utilities` whose values identify which dense alternative
            each utility column corresponds to. Use -999 for masked or unavailable positions.
        n_rands : int, optional
            Number of EV1 draws to generate per chooser row. Required when `alt_nrs_df`
            is provided and may exceed the visible number of utility columns.

        Returns
        -------
        positions : 1-D ndarray of int32
            Array with shape (len(utilities),) containing the winning column position
            for each chooser row.
        """

        assert self.step_name
        assert self.step_name == step_name

        utility_values = utilities.to_numpy()
        n_rows, n_alts = utility_values.shape
        positions = np.empty(n_rows, dtype=np.int32)

        if alt_nrs_df is not None:
            assert alt_nrs_df.shape == utilities.shape
            if n_rands is None:
                raise ValueError("n_rands is required when alt_nrs_df is provided")
            alt_nr_values = alt_nrs_df.to_numpy()
            masked = alt_nr_values == -999
            safe_alt_nrs = np.where(masked, 0, alt_nr_values)
        else:
            if n_rands is None:
                n_rands = n_alts
            elif n_rands != n_alts:
                raise ValueError("n_rands must equal utilities.shape[1] when alt_nrs_df is omitted")
            alt_nr_values = masked = safe_alt_nrs = None

        generators = self._generators_for_df(utilities)

        for row_num, prng in enumerate(generators):
            utility_row = utility_values[row_num]
            row_gumbels = -np.log(-np.log(prng.rand(n_rands)))

            if alt_nrs_df is None:
                positions[row_num] = np.argmax(row_gumbels + utility_row)
            else:
                candidate_values = utility_row + row_gumbels[safe_alt_nrs[row_num]]
                candidate_values[masked[row_num]] = utility_row[masked[row_num]]
                positions[row_num] = np.argmax(candidate_values)

        self.row_states.loc[utilities.index, "offset"] += n_rands
        return positions

    def normal_for_df(self, df, step_name, mu, sigma, lognormal=False, size=None):
        """
        Return a floating point random number in normal (or lognormal) distribution
        for each row in df using the appropriate random channel for each row.

        Subsequent calls (in the same step) will return the next rand for each df row

        The resulting array will be the same length (and order) as df
        This method is designed to support alternative selection from a probability array

        The columns in df are ignored; the index name and values are used to determine
        which random number sequence to to use.

        If "true pseudo random" behavior is desired (i.e. NOT repeatable) the set_base_seed
        method (q.v.) may be used to globally reseed all random streams.

        Parameters
        ----------
        df : pandas.DataFrame or Series
            df or series with index name and values corresponding to a registered channel

        mu : float or pd.Series or array of floats with one value per df row
        sigma : float or array of floats with one value per df row

        Returns
        -------
        rands : 2-D ndarray
            array the same length as df, with n floats in range [0, 1) for each df row
        """

        assert self.step_name
        assert self.step_name == step_name

        def to_series(x):
            if np.isscalar(x):
                return [x] * len(df)
            elif isinstance(x, pd.Series):
                return x.values
            return x

        # - reminder: prng must be called when yielded as generated sequence, not serialized
        generators = self._generators_for_df(df)

        mu = to_series(mu)
        sigma = to_series(sigma)

        if lognormal:
            rands = np.asanyarray(
                [
                    prng.lognormal(mean=mu[i], sigma=sigma[i], size=size)
                    for i, prng in enumerate(generators)
                ]
            )
        else:
            rands = np.asanyarray(
                [
                    prng.normal(loc=mu[i], scale=sigma[i], size=size)
                    for i, prng in enumerate(generators)
                ]
            )

        # update offset for rows we handled
        if size is not None:
            consume_offsets = int(size)
        else:
            consume_offsets = 1
        self.row_states.loc[df.index, "offset"] += consume_offsets

        return rands

    def choice_for_df(self, df, step_name, a, size, replace):
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

        The remaining parameters are passed through as arguments to numpy.random.choice

        a : 1-D array-like or int
            If an ndarray, a random sample is generated from its elements.
            If an int, the random sample is generated as if a was np.arange(n)
        size : int or tuple of ints
            Output shape
        replace : boolean
            Whether the sample is with or without replacement

        Returns
        -------
        choices : 1-D ndarray of length: size * len(df.index)
            The generated random samples for each row concatenated into a single (flat) array
        """

        assert self.step_name
        assert self.step_name == step_name

        # initialize the generator iterator
        generators = self._generators_for_df(df)

        sample = np.concatenate(
            tuple(prng.choice(a, size, replace) for prng in generators)
        )

        if not self.multi_choice_offset:
            # FIXME - if replace, should we estimate rands_consumed?
            if replace:
                logger.warning("choice_for_df MULTI_CHOICE_FF with replace")
            # update offset for rows we handled
            self.row_states.loc[df.index, "offset"] += size

        return sample


class Random(object):
    def __init__(self):
        self.channels = {}

        # dict mapping df index name to channel name
        self.index_to_channel = {}

        self.step_name = None
        self.step_seed = None
        self.base_seed = 0
        self.global_rng = np.random.RandomState()

    def get_channel_for_df(self, df):
        """
        Return the channel for this df. Channel should already have been loaded/added.

        Parameters
        ----------
        df : pandas.dataframe
            either a domain_df for a channel being added or extended
            or a df for which random values are to be generated
        """

        channel_name = self.index_to_channel.get(df.index.name, None)
        if channel_name is None:
            raise TableIndexError("No channel with index name '%s'" % df.index.name)
        return self.channels[channel_name]

    def reset_offsets_for_step(self, step_name):
        """
        Reset offsets for all channels for a step

        Parameters
        ----------
        step_name : str
            pipeline step name for this step
        """

        assert self.step_name == step_name

        for c in self.channels:
            self.channels[c].row_states["offset"] = 0

    def reset_offsets_for_df(self, df):
        """
        Reset offsets for all choosers in df if the channel for a step

        Parameters
        ----------
        step_name : str
            pipeline step name for this step
        df : pandas.DataFrame
            df with index name and values corresponding to a registered channel
        """
        channel = self.get_channel_for_df(df)
        channel.row_states.loc[df.index, "offset"] = 0
        logger.info(
            f"RNG: resetting random number generator offsets for channel '{channel.channel_name}' for {len(df)} rows"
            + f" with index name '{df.index.name}'. Total lenght df: {len(channel.row_states)}"
        )

    def begin_step(self, step_name):
        """
        Register that the pipeline has entered a new step and that global and channel streams
        should transition to the new stream.

        Parameters
        ----------
        step_name : str
            pipeline step name
        """

        if self.step_name is not None:
            raise ValueError(f"already in step {self.step_name}")
        assert step_name is not None

        self.step_name = step_name

        self.step_seed = hash32(step_name)

        seed = [self.base_seed, self.step_seed]
        self.global_rng = np.random.RandomState(seed)

        for c in self.channels:
            self.channels[c].begin_step(self.step_name)

    def end_step(self, step_name):
        """
        This is mostly just for internal consistency checking -
        I'm not sure it serves any useful purpose except to catch "mis-steps" in the pipeline code

        Parameters
        ----------
        step_name : str
            name of current step (just a consistency check)
        """
        if self.step_name is None:
            # maybe a step was aborted, this is fine
            return
        assert self.step_name == step_name

        for c in self.channels:
            self.channels[c].end_step(self.step_name)

        self.step_name = None
        self.step_seed = None
        self.global_rng = None

    # channel management

    def add_channel(self, channel_name, domain_df):
        """
        Create or extend a channel for generating random number streams for domain_df.

        We need to be prepared to extend an existing channel because mandatory and non-mandatory
        tours are generated separately by different sub-models, but end up members of a common
        tours channel.

        Parameters
        ----------
        domain_df : pandas.DataFrame
            domain dataframe with index values for which random streams are to be generated
            and well-known index name corresponding to the channel

        channel_name : str
            expected channel name provided as a consistency check

        """

        if channel_name in self.channels:
            assert channel_name == self.index_to_channel[domain_df.index.name]
            logger.debug(
                "Random: extending channel '%s' %s ids"
                % (channel_name, len(domain_df.index))
            )
            channel = self.channels[channel_name]

            channel.extend_domain(domain_df)

        else:
            logger.debug(
                "Adding channel '%s' %s ids" % (channel_name, len(domain_df.index))
            )

            channel = SimpleChannel(
                channel_name, self.base_seed, domain_df, self.step_name
            )

            self.channels[channel_name] = channel
            self.index_to_channel[domain_df.index.name] = channel_name

    def drop_channel(self, channel_name):
        """
        Drop channel that won't be used again (saves memory)

        Parameters
        ----------
        channel_name
        """

        if channel_name in self.channels:
            logger.debug("Dropping channel '%s'" % (channel_name,))
            del self.channels[channel_name]
        else:
            logger.error(
                "drop_channel called with unknown channel '%s'" % (channel_name,)
            )

    def set_base_seed(self, seed=None):
        """
        Like seed for numpy.random.RandomState, but generalized for use with all random streams.

        Provide a base seed that will be added to the seeds of all random streams.
        The default base seed value is 0, so set_base_seed(0) is a NOP

        set_base_seed(1) will (e.g.) provide a different set of random streams than the default
        but will provide repeatable results re-running or resuming the simulation

        set_base_seed(None) will set the base seed to a random and unpredictable integer and so
        provides "fully pseudo random" non-repeatable streams with different results every time

        Must be called before first step (before any channels are added or rands are consumed)

        Parameters
        ----------
        seed : int or None
        """

        if self.step_name is not None or self.channels:
            raise DuplicateLoadableObjectError(
                "Can only call set_base_seed before the first step."
            )

        assert len(list(self.channels.keys())) == 0

        if seed is None:
            self.base_seed = np.random.RandomState().randint(_MAX_SEED, dtype=np.uint32)
            logger.debug("Set random seed randomly to %s" % self.base_seed)
        else:
            logger.debug("Set random seed base to %s" % seed)
            self.base_seed = seed

    def get_global_rng(self):
        """
        Return a numpy random number generator for use within current step.

        This method is designed to provide random numbers for uses that do not correspond to
        known channel domains. e.g. to select a subset of households to use for the simulation.

        global_rng is reseeded to a predictable value at the beginning of every step so that
        it behaves repeatably when simulation is resumed or re-run.

        If "true pseudo random" behavior is desired (i.e. NOT repeatable) the set_base_seed
        method (q.v.) may be used to globally reseed all random streams.

        Returns
        -------
        global_rng : numpy.random.RandomState()
            numpy random number generator for use within current step

        """
        assert self.step_name is not None
        return self.global_rng

    def get_external_rng(self, one_off_step_name):
        """
        Return a numpy random number generator for step-independent one_off use

        exists to allow sampling of input tables consistent no matter what step they are called in
        """

        seed = [self.base_seed, hash32(one_off_step_name)]
        return np.random.RandomState(seed)

    def random_for_df(self, df, n=1):
        """
        Return a single floating point random number in range [0, 1) for each row in df
        using the appropriate random channel for each row.

        Subsequent calls (in the same step) will return the next rand for each df row

        The resulting array will be the same length (and order) as df
        This method is designed to support alternative selection from a probability array

        The columns in df are ignored; the index name and values are used to determine
        which random number sequence to to use.

        We assume that we can identify the channel to used based on the name of df.index
        This channel should have already been registered by a call to add_channel (q.v.)

        If "true pseudo random" behavior is desired (i.e. NOT repeatable) the set_base_seed
        method (q.v.) may be used to globally reseed all random streams.

        Parameters
        ----------
        df : pandas.DataFrame
            df with index name and values corresponding to a registered channel

        n : int
            number of rands desired (default 1)

        Returns
        -------
        choices : 1-D ndarray the same length as df
            a single float in range [0, 1) for each row in df
        """

        # FIXME - for tests
        if not self.channels:
            rng = np.random.RandomState(0)
            rands = np.asanyarray([rng.rand(n) for _ in range(len(df))])
            return rands

        channel = self.get_channel_for_df(df)
        rands = channel.random_for_df(df, self.step_name, n)
        return rands

    def random_for_df_stable_alt_positions(
        self,
        df,
        stable_alt_positions,
        n_total_alts,
    ):
        """
        Return per-row uniform draws aligned to active alternatives using a stable
        larger alternative universe.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with one row per chooser and one column per active alternative.
        stable_alt_positions : 1-D ndarray
            Mapping from active columns to positions in the larger stable alternative
            universe.
        n_total_alts : int
            Number of alternatives in the larger stable universe.

        Returns
        -------
        rands : 2-D ndarray
            Array with shape `(len(df), df.shape[1])` containing uniforms aligned to
            the active alternatives.
        """

        n_alts = df.shape[1]
        stable_alt_positions = np.asarray(stable_alt_positions)
        if stable_alt_positions.shape != (n_alts,):
            raise ValueError(
                "stable_alt_positions must be a 1-D array aligned to df columns"
            )
        if stable_alt_positions.min() < 0 or stable_alt_positions.max() >= n_total_alts:
            raise ValueError(
                "stable_alt_positions values must be within [0, n_total_alts)"
            )

        if not self.channels:
            rng = np.random.RandomState(0)
            return np.asanyarray(
                [rng.rand(n_total_alts)[stable_alt_positions] for _ in range(len(df))]
            )

        channel = self.get_channel_for_df(df)
        return channel.random_for_df_stable_alt_positions(
            df,
            self.step_name,
            stable_alt_positions,
            n_total_alts,
        )

    def gumbel_for_df(self, df, n=1):
        """
        Return a single floating point gumbel for each row in df
        using the appropriate random channel for each row.

        Subsequent calls (in the same step) will return the next rand for each df row

        The resulting array will be the same length (and order) as df
        This method is designed to support alternative selection from a probability array

        The columns in df are ignored; the index name and values are used to determine
        which random number sequence to to use.

        We assume that we can identify the channel to used based on the name of df.index
        This channel should have already been registered by a call to add_channel (q.v.)

        If "true pseudo random" behavior is desired (i.e. NOT repeatable) the set_base_seed
        method (q.v.) may be used to globally reseed all random streams.

        Parameters
        ----------
        df : pandas.DataFrame
            df with index name and values corresponding to a registered channel

        n : int
            number of rands desired (default 1)

        Returns
        -------
        choices : 1-D ndarray the same length as df
            a single float in range [0, 1) for each row in df
        """
        channel = self.get_channel_for_df(df)
        rands = channel.gumbel_for_df(df, self.step_name, n)
        return rands

    def gumbel_max_positions_for_df(
        self,
        utilities,
        sample_size,
        stable_alt_positions=None,
        n_total_alts=None,
    ):
        """
        Return the winning alternative position for each chooser/sample pair
        using the appropriate channel for each chooser row.

        Parameters
        ----------
        utilities : pandas.DataFrame
            DataFrame with one row per chooser and one column per alternative.
        sample_size : int
            Number of repeated sampled choices to make per chooser.
        stable_alt_positions : 1-D ndarray, optional
            Mapping from active utility columns to positions in a larger stable
            alternative universe.
        n_total_alts : int, optional
            Number of alternatives in the larger stable universe.

        Returns
        -------
        positions : 2-D ndarray of int32
            Array with shape (len(utilities), sample_size) containing the column
            position of the winning alternative for each chooser/sample pair.
        """
        if not self.channels:
            utility_values = utilities.to_numpy()
            n_rows, n_alts = utility_values.shape
            positions = np.empty((n_rows, sample_size), dtype=np.int32)
            rng = np.random.RandomState(0)

            if stable_alt_positions is not None or n_total_alts is not None:
                if stable_alt_positions is None or n_total_alts is None:
                    raise ValueError(
                        "stable_alt_positions and n_total_alts must both be provided or omitted together"
                    )
                stable_alt_positions = np.asarray(stable_alt_positions)
                if stable_alt_positions.shape != (n_alts,):
                    raise ValueError(
                        "stable_alt_positions must be a 1-D array aligned to utilities columns"
                    )
                if stable_alt_positions.min() < 0 or stable_alt_positions.max() >= n_total_alts:
                    raise ValueError(
                        "stable_alt_positions values must be within [0, n_total_alts)"
                    )
                n_gumbels = n_total_alts
            else:
                n_gumbels = n_alts

            for row_num, utility_row in enumerate(utility_values):
                row_gumbels = -np.log(-np.log(rng.rand(n_gumbels * sample_size))).reshape(
                    (sample_size, n_gumbels)
                )
                if stable_alt_positions is not None:
                    row_gumbels = row_gumbels[:, stable_alt_positions]
                positions[row_num, :] = np.argmax(
                    row_gumbels + utility_row[np.newaxis, :],
                    axis=1,
                )

            return positions

        channel = self.get_channel_for_df(utilities)
        return channel.gumbel_max_positions_for_df(
            utilities,
            self.step_name,
            sample_size,
            stable_alt_positions=stable_alt_positions,
            n_total_alts=n_total_alts,
        )

    def gumbel_choice_positions_for_df(self, utilities, alt_nrs_df=None, n_rands=None):
        """
        Return the winning alternative position for each chooser row.

        Parameters
        ----------
        utilities : pandas.DataFrame
            DataFrame with one row per chooser and one column per available alternative.
        alt_nrs_df : pandas.DataFrame, optional
            Dense-alternative mapping aligned to `utilities`.
        n_rands : int, optional
            Number of EV1 draws to generate per chooser row.

        Returns
        -------
        positions : 1-D ndarray of int32
        """
        if not self.channels:
            rng = np.random.RandomState(0)
            utility_values = utilities.to_numpy()
            positions = np.empty(len(utilities), dtype=np.int32)

            if alt_nrs_df is not None:
                if n_rands is None:
                    raise ValueError("n_rands is required when alt_nrs_df is provided")
                alt_nr_values = alt_nrs_df.to_numpy()
                masked = alt_nr_values == -999
                safe_alt_nrs = np.where(masked, 0, alt_nr_values)
                for row_num, utility_row in enumerate(utility_values):
                    row_gumbels = -np.log(-np.log(rng.rand(n_rands)))
                    candidate_values = utility_row + row_gumbels[safe_alt_nrs[row_num]]
                    candidate_values[masked[row_num]] = utility_row[masked[row_num]]
                    positions[row_num] = np.argmax(candidate_values)
            else:
                if n_rands is None:
                    n_rands = utility_values.shape[1]
                for row_num, utility_row in enumerate(utility_values):
                    positions[row_num] = np.argmax(
                        -np.log(-np.log(rng.rand(n_rands))) + utility_row
                    )

            return positions

        channel = self.get_channel_for_df(utilities)
        return channel.gumbel_choice_positions_for_df(
            utilities,
            self.step_name,
            alt_nrs_df=alt_nrs_df,
            n_rands=n_rands,
        )

    def normal_for_df(self, df, mu=0, sigma=1, broadcast=False, size=None):
        """
        Return a single floating point normal random number in range (-inf, inf) for each row in df
        using the appropriate random channel for each row.

        Subsequent calls (in the same step) will return the next rand for each df row

        The resulting array will be the same length (and order) as df
        This method is designed to support alternative selection from a probability array

        The columns in df are ignored; the index name and values are used to determine
        which random number sequence to to use.

        We assume that we can identify the channel to used based on the name of df.index
        This channel should have already been registered by a call to add_channel (q.v.)

        If "true pseudo random" behavior is desired (i.e. NOT repeatable) the set_base_seed
        method (q.v.) may be used to globally reseed all random streams.

        Parameters
        ----------
        df : pandas.DataFrame
            df with index name and values corresponding to a registered channel

        mu : float or array of floats with one value per df row
        sigma : float or array of floats with one value per df row

        Returns
        -------
        rands : 1-D ndarray the same length as df (or Series with same index as df)
            a single float in lognormal distribution for each row in df
        """

        channel = self.get_channel_for_df(df)

        if broadcast:
            alts_df = df
            df = df.index.unique().to_series()
            rands = channel.normal_for_df(
                df, self.step_name, mu=0, sigma=1, lognormal=False, size=size
            )
            if size is not None:
                rands = reindex(pd.DataFrame(rands, index=df.index), alts_df.index)
            else:
                rands = reindex(pd.Series(rands, index=df.index), alts_df.index)
            rands = rands * sigma + mu
        else:
            rands = channel.normal_for_df(
                df, self.step_name, mu, sigma, lognormal=False, size=size
            )

        return rands

    def lognormal_for_df(self, df, mu, sigma, broadcast=False, scale=False):
        """
        Return a single floating point lognormal random number in range [0, inf) for each row in df
        using the appropriate random channel for each row.

        Note that by default (scale=False) the mean and standard deviation are not the values for
        the distribution itself, but of the underlying normal distribution it is derived from.
        This is perhaps counter-intuitive, but it is the way the numpy standard works,
        and so we are conforming to it here.

        If scale=True, then mu and sigma are the desired mean and standard deviation of the
        lognormal distribution instead of the numpy standard where mu and sigma which are the
        values for the distribution itself, rather than of the underlying normal distribution
        it is derived from.

        Subsequent calls (in the same step) will return the next rand for each df row

        The resulting array will be the same length (and order) as df
        This method is designed to support alternative selection from a probability array

        The columns in df are ignored; the index name and values are used to determine
        which random number sequence to to use.

        We assume that we can identify the channel to used based on the name of df.index
        This channel should have already been registered by a call to add_channel (q.v.)

        If "true pseudo random" behavior is desired (i.e. NOT repeatable) the set_base_seed
        method (q.v.) may be used to globally reseed all random streams.

        Parameters
        ----------
        df : pandas.DataFrame, Series, or Index
            df with index name and values corresponding to a registered channel

        mu : float or array of floats with one value per df row
        sigma : float or array of floats with one value per df row

        Returns
        -------
        rands : 1-D ndarray the same length as df (or Series with same index as df)
            a single float in lognormal distribution for each row in df
        """

        if scale:
            # location = ln(mean/sqrt(1 + std_dev^2/mean^2))
            # scale = sqrt(ln(1 + std_dev^2/mean^2))
            x = 1 + ((sigma * sigma) / (mu * mu))
            mu = np.log(mu / (np.sqrt(x)))
            sigma = np.sqrt(np.log(x))

        if broadcast:
            rands = self.normal_for_df(df, mu=mu, sigma=sigma, broadcast=True)
            rands = np.exp(rands)
        else:
            channel = self.get_channel_for_df(df)
            rands = channel.normal_for_df(
                df, self.step_name, mu=mu, sigma=sigma, lognormal=True
            )

        return rands

    def choice_for_df(self, df, a, size, replace):
        """
        Apply numpy.random.choice once for each row in df
        using the appropriate random channel for each row.

        Concatenate the the choice arrays for every row into a single 1-D ndarray
        The resulting array will be of length: size * len(df.index)
        This method is designed to support creation of a interaction_dataset

        The columns in df are ignored; the index name and values are used to determine
        which random number sequence to to use.

        We assume that we can identify the channel to used based on the name of df.index
        This channel should have already been registered by a call to add_channel (q.v.)

        Parameters
        ----------
        df : pandas.DataFrame
            df with index name and values corresponding to a registered channel

        The remaining parameters are passed through as arguments to numpy.random.choice

        a : 1-D array-like or int
            If an ndarray, a random sample is generated from its elements.
            If an int, the random sample is generated as if a was np.arange(n)
        size : int or tuple of ints
            Output shape
        replace : boolean
            Whether the sample is with or without replacement

        Returns
        -------
        choices : 1-D ndarray of length: size * len(df.index)
            The generated random samples for each row concatenated into a single (flat) array
        """

        # FIXME - for tests
        if not self.channels:
            rng = np.random.RandomState(0)
            choices = np.concatenate(
                tuple(rng.choice(a, size, replace) for _ in range(len(df)))
            )
            return choices

        t0 = print_elapsed_time()
        channel = self.get_channel_for_df(df)
        choices = channel.choice_for_df(df, self.step_name, a, size, replace)
        t0 = print_elapsed_time(
            "choice_for_df for %s rows" % len(df.index), t0, debug=True
        )
        return choices
