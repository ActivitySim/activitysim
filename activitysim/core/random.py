# ActivitySim
# See full license in LICENSE.txt.

import hashlib
import logging
from builtins import object, range

import numpy as np
import pandas as pd

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

    def normal_for_df(self, df, step_name, mu, sigma, lognormal=False):
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
                    prng.lognormal(mean=mu[i], sigma=sigma[i])
                    for i, prng in enumerate(generators)
                ]
            )
        else:
            rands = np.asanyarray(
                [
                    prng.normal(loc=mu[i], scale=sigma[i])
                    for i, prng in enumerate(generators)
                ]
            )

        # update offset for rows we handled
        self.row_states.loc[df.index, "offset"] += 1

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
            raise RuntimeError("No channel with index name '%s'" % df.index.name)
        return self.channels[channel_name]

    # step handling

    def begin_step(self, step_name):
        """
        Register that the pipeline has entered a new step and that global and channel streams
        should transition to the new stream.

        Parameters
        ----------
        step_name : str
            pipeline step name
        """

        assert self.step_name is None
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
        assert self.step_name is not None
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
            raise RuntimeError("Can only call set_base_seed before the first step.")

        assert len(list(self.channels.keys())) == 0

        if seed is None:
            self.base_seed = np.random.RandomState().randint(_MAX_SEED)
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

    def normal_for_df(self, df, mu=0, sigma=1, broadcast=False):
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
                df, self.step_name, mu=0, sigma=1, lognormal=False
            )
            rands = reindex(pd.Series(rands, index=df.index), alts_df.index)
            rands = rands * sigma + mu
        else:
            rands = channel.normal_for_df(
                df, self.step_name, mu, sigma, lognormal=False
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
