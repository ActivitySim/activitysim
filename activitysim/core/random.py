import collections

import numpy as np
import pandas as pd

import inject
from .tracing import print_elapsed_time


import logging

logger = logging.getLogger(__name__)

# one more than 0xFFFFFFFF so we can wrap using: int64 % _MAX_SEED
_MAX_SEED = (1 << 32)

# not arbitrary, as we count on incrementing step_num from NULL_STEP_NUM to 0
NULL_STEP_NUM = -1

MAX_STEPS = 100


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

    So instead, multiply the domain_df index by the number of steps required for the channel
    add the step_num to the row_seed to get a unique seed for each (domain_df index, step_num)
    tuple.

    Currently, it is possible that random streams for rows in different tables may coincide.
    This would be easy to avoid with either seed arrays or fast jump/offset.

    numpy random seeds are unsigned int32 so there are 4,294,967,295 available seeds.
    That is probably just about enough to distribute evenly, for most cities, depending on the
    number of households, persons, tours, trips, and steps.

    We do read in the whole households and persons tables at start time, so we could note the
    max index values. But we might then want a way to ensure stability between the test, example,
    and full datasets. I am punting on this for now.
    """

    def __init__(self, channel_name, base_seed, domain_df, step_num):

        self.name = channel_name
        self.base_seed = base_seed

        # ensure that every channel is different, even for the same df index values and max_steps
        self.unique_channel_seed = hash(self.name) % _MAX_SEED

        assert (step_num == NULL_STEP_NUM) or step_num >= 0
        self.step_num = step_num

        assert (self.step_num < MAX_STEPS)

        # create dataframe to hold state for every df row
        self.row_states = self.create_row_states_for_domain(domain_df)

    def create_row_states_for_domain(self, domain_df):
        """
        Create a dataframe with same index as domain_df and a single column
        with stable, predictable, repeatable row_seeds for that domain_df index value

        See notes on the seed generation strategy in class comment above.

        Parameters
        ----------
        domain_df : pandas.dataframe
            domain dataframe with index values for which random streams are to be generated

        Returns
        -------
        row_states : pandas.DataFrame
        """

        # dataframe to hold state for every df row
        row_states = pd.DataFrame(columns=['row_seed', 'offset'], index=domain_df.index)

        if not row_states.empty:
            row_states['row_seed'] = (self.base_seed + self.unique_channel_seed +
                                      row_states.index * MAX_STEPS) % _MAX_SEED
            row_states['offset'] = 0

        return row_states

    def extend_domain(self, domain_df):
        """
        Extend existing row_state df by adding seed info for each row in domain_df

        It is assumed that the index values of the component tables are disjoint and
        there will be no ambiguity/collisions between them

        Parameters
        ----------
        domain_df : pandas.DataFrame
            domain dataframe with index values for which random streams are to be generated
            and well-known index name corresponding to the channel

        step_name : str or None
            provided when reloading so we can restore step_name and step_num

        step_num : int or None
        """

        # these should be new rows, no intersection with existing row_states
        assert len(self.row_states.index.intersection(domain_df.index)) == 0

        new_row_states = self.create_row_states_for_domain(domain_df)
        self.row_states = pd.concat([self.row_states, new_row_states])

    def begin_step(self, step_num):
        """
        Reset channel state for a new state

        Parameters
        ----------
        step_name : str
            pipeline step name for this step
        """

        self.step_num = step_num

        if self.step_num >= MAX_STEPS:
            raise RuntimeError("Too many steps: %s (max %s) for channel '%s'"
                               % (self.step_num, MAX_STEPS, self.name))

        # number of rands pulled this step
        self.row_states['offset'] = 0

        # standard constant to use for choice_for_df instead of fast-forwarding rand stream
        self.multi_choice_offset = None

    def _generators_for_df(self, df):
        """
        Python generator function for iterating over numpy prngs (nomenclature collision!)
        seeded and fast-forwarded on-the-fly to the appropriate position in the channel's
        random number stream for each row in df.

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

            seed = (row.row_seed + self.step_num) % _MAX_SEED
            prng.seed(seed)

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
        generators = self._generators_for_df(df)
        rands = np.asanyarray([prng.rand(n) for prng in generators])
        # update offset for rows we handled
        self.row_states.loc[df.index, 'offset'] += n
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

        # initialize the generator iterator
        generators = self._generators_for_df(df)

        sample = np.concatenate(tuple(prng.choice(a, size, replace) for prng in generators))

        if not self.multi_choice_offset:
            # FIXME - if replace, should we estimate rands_consumed?
            if replace:
                logger.warn("choice_for_df MULTI_CHOICE_FF with replace")
            # update offset for rows we handled
            self.row_states.loc[df.index, 'offset'] += size

        return sample


class Random(object):

    def __init__(self):

        # dict mapping df index_name to channel (table) name
        self.index_to_channel_map = {}

        self.channels = {}
        self.step_name = None
        self.step_num = NULL_STEP_NUM
        self.step_seed = None
        self.base_seed = 0
        self.global_rng = np.random.RandomState()

    def set_channel_info(self, channel_info):

        assert len(self.channels) == 0
        assert len(self.index_to_channel_map) == 0

        # for mapping index name to channel name
        self.index_to_channel_map = \
            {index_name: channel_name for channel_name, index_name in channel_info.iteritems()}

    def get_channel_name_for_df(self, df):
        """
        Return the channel name corresponding to the index name of df

        We expect that the random number channel can be determined by the name of the index of the
        dataframe accompanying the request. This mapping was specified in channel_info

        This function internally encapsulates the knowledge of that mapping.

        Parameters
        ----------
        df : pandas.DataFrame
            domain_df or a df passed to random number/choice methods with well known index name

        Returns
        -------
            channel_name : str
        """
        channel_name = self.index_to_channel_map.get(df.index.name, None)
        if channel_name is None:
            raise RuntimeError("No channel with index name '%s'" % df.index.name)
        return channel_name

    def get_channel_for_df(self, df):
        """
        Return the channel for this df. Channel should already have been loaded/added.

        Parameters
        ----------
        df : pandas.dataframe
            either a domain_df for a channel being added or extended
            or a df for which random values are to be generated
        """
        channel_name = self.get_channel_name_for_df(df)
        if channel_name not in self.channels:
            raise RuntimeError("Channel '%s' has not yet been added." % channel_name)
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
        assert step_name != self.step_name

        self.step_name = step_name
        self.step_num += 1

        self.step_seed = hash(step_name) % _MAX_SEED

        seed = [self.base_seed, self.step_seed]
        self.global_rng = np.random.RandomState(seed)

        for c in self.channels:
            self.channels[c].begin_step(self.step_num)

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

        self.step_name = None
        self.step_seed = None
        self.global_rng = None

    # channel management

    def add_channel(self, domain_df, channel_name):
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

        step_name : str or None
            for channels being loaded (resumed) we need the step_name and step_num to maintain
            consistent step numbering

        step_num : int or NULL_STEP_NUM
            for channels being loaded (resumed) we need the step_name and step_num to maintain
            consistent step numbering
        """

        assert channel_name == self.get_channel_name_for_df(domain_df)

        if channel_name in self.channels:
            logger.debug("Random: extending channel '%s' %s ids" %
                         (channel_name, len(domain_df.index)))
            channel = self.channels[channel_name]

            channel.extend_domain(domain_df)

        else:
            logger.debug("Random: adding channel '%s' %s ids" %
                         (channel_name, len(domain_df.index)))

            channel = SimpleChannel(channel_name,
                                    self.base_seed,
                                    domain_df,
                                    self.step_num
                                    )

            self.channels[channel_name] = channel

    def load_channels(self, step_num):
        """
        Called on resume to initialize channels for existing saved tables
        All channels should have been registered by a call to set_channel_info.
        This method checks to see if any of them have an injectable table

        Parameters
        ----------
        step_num

        Returns
        -------

        """

        self.step_num = step_num
        for index_name in self.index_to_channel_map:

            channel_name = self.index_to_channel_map[index_name]
            df = inject.get_table(channel_name, None)

            if df is not None:
                logger.debug("loading channel %s" % (channel_name,))
                self.add_channel(df, channel_name=channel_name)
                self.channels[channel_name].begin_step(step_num)
            else:
                logger.debug("skipping channel %s" % (channel_name,))

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

        assert len(self.channels.keys()) == 0

        if seed is None:
            self.base_seed = np.random.RandomState().randint(_MAX_SEED)
            logger.info("Set random seed randomly to %s" % self.base_seed)
        else:
            logger.info("Set random seed base to %s" % seed)
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
            choices = np.concatenate(tuple(rng.choice(a, size, replace) for _ in range(len(df))))
            return choices

        t0 = print_elapsed_time()
        channel = self.get_channel_for_df(df)
        choices = channel.choice_for_df(df, self.step_name, a, size, replace)
        t0 = print_elapsed_time("choice_for_df for %s rows" % len(df.index), t0, debug=True)
        return choices
