import collections

import numpy as np
import pandas as pd
import orca

from tracing import print_elapsed_time, log_memory_info

import logging

logger = logging.getLogger(__name__)

_MAX_SEED = (1 << 32)


SavedChannelState = collections.namedtuple('SavedChannelState', 'channel_name step_num step_name')


def channel_name_from_index(df):

    _INDEX_CHANNEL_NAMES = {
        'PERID': 'persons',
        'HHID': 'households',
        'tour_id': 'tours',
        'trip_id': 'trips',
    }
    index_name = df.index.name

    if index_name not in _INDEX_CHANNEL_NAMES:
        raise RuntimeError("can't determine get_df_channel_name for index '%s'" % (index_name,))

    return _INDEX_CHANNEL_NAMES[index_name]


class SimpleChannel(object):

    def __init__(self, channel_name, base_seed, domain_df, max_steps, step_name, step_num):

        assert channel_name_from_index(domain_df) == channel_name

        self.name = channel_name
        self.base_seed = base_seed

        # ensure that every channel is different, even for the same df index values and max_steps
        self.unique_channel_seed = hash(self.name) % _MAX_SEED

        self.step_name = step_name
        self.step_num = step_num if step_num is not None else -1
        self.max_steps = max_steps

        assert (self.step_num < self.max_steps)

        # create dataframe to hold state for every df row
        self.row_states = self.create_row_states_for_domain(domain_df)

    def create_row_states_for_domain(self, domain_df):

        # dataframe to hold state for every df row
        row_states = pd.DataFrame(index=domain_df.index)

        # ensure that every channel is different, even for the same df index values and max_steps
        unique_channel_seed = hash(self.name) % _MAX_SEED

        row_states['row_seed'] = (self.base_seed + self.unique_channel_seed +
                                  row_states.index * self.max_steps) % _MAX_SEED

        return row_states

    def extend_domain(self, domain_df, step_name=None, step_num=None):
        # add additional domain_df rows to an existing channel
        assert channel_name_from_index(domain_df) == self.name

        # these should be new rows, no intersection with existing row_states
        assert len(self.row_states.index.intersection(domain_df.index)) == 0

        self.step_name = step_name
        if step_num:
            assert step_num >= self.step_num
            self.step_num = step_num

        new_row_states = self.create_row_states_for_domain(domain_df)
        self.row_states = pd.concat([self.row_states, new_row_states])

    def begin_step(self, step_name):

        if self.step_name == step_name:
            return

        self.step_name = step_name
        self.step_num += 1

        if self.step_num >= self.max_steps:
            raise RuntimeError("Too many steps (%s) maxstep %s for channel '%s'"
                               % (self.step_num, self.max_steps, self.name))

        # number of rands pulled this step
        self.row_states['offset'] = 0

        # standard constant to use for choice_for_df instead of fast-forwarding rand stream
        self.multi_choice_offset = None

        logger.info("begin_step '%s' for channel '%s'" % (step_name, self.name, ))

    def _generators_for_df(self, df, override_offset=None):

        # assert no dupes
        assert len(df.index.unique() == len(df.index))

        df_row_states = self.row_states.loc[df.index]

        prng = np.random.RandomState()
        for row in df_row_states.itertuples():

            seed = (row.row_seed + self.step_num) % _MAX_SEED
            prng.seed(seed)

            offset = override_offset or row.offset
            if offset:
                # consume rands
                prng.rand(offset)

            yield prng

    def set_multi_choice_offset(self, offset, step_name):

        # must do this before step begins
        assert self.step_name != step_name

        # expect an int or None
        assert offset is None or type(offset) == int

        self.begin_step(step_name)
        self.multi_choice_offset = offset

    def random_for_df(self, df, step_name):

        self.begin_step(step_name)
        generators = self._generators_for_df(df)
        r = [prng.rand(1) for prng in generators]
        # update offset for rows we handled
        self.row_states.loc[df.index, 'offset'] += 1
        return r

    def choice_for_df(self, df, step_name, a, size, replace):

        self.begin_step(step_name)

        generators = self._generators_for_df(df, self.multi_choice_offset)

        sample = np.concatenate(tuple(prng.choice(a, size, replace) for prng in generators))

        if not self.multi_choice_offset:
            # FIXME - if replace, should we estimate rands_consumed?
            if replace:
                logger.warn("choice_for_df MULTI_CHOICE_FF with replace")
            # update offset for rows we handled
            self.row_states.loc[df.index, 'offset'] += size

        return sample


class Random(object):

    def __init__(self, max_channel_steps={}):

        self.max_channel_steps = max_channel_steps
        self.channels = {}
        self.step_name = None
        self.step_seed = None

        self.base_seed = 0

        self.global_rng = np.random.RandomState()

    # step handling

    def begin_step(self, step_name):

        assert self.step_name is None
        assert step_name is not None
        assert step_name != self.step_name

        self.step_name = step_name
        self.step_seed = hash(step_name) % _MAX_SEED

        seed = [self.base_seed, self.step_seed]
        self.global_rng = np.random.RandomState(seed)

    def end_step(self, step_name):

        assert self.step_name is not None
        assert self.step_name == step_name

        self.step_name = None
        self.step_seed = None
        self.global_rng = None

    # channel management

    def get_channel_for_df(self, df):
        """
        INTERNAL

        return the canonical channel name for use with this df, based on its index name

        Parameters
        ----------
        df : pandas.dataframe
            either a domain_df for a channel being added or extended
            or a df for which random values are to be generated
        """
        channel_name = channel_name_from_index(df)
        assert channel_name in self.channels
        return self.channels[channel_name]

    def add_channel(self, domain_df, channel_name, step_name=None, step_num=None):
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

        step_num : int or None
            for channels being loaded (resumed) we need the step_name and step_num to maintain
            consistent step numbering
        """
        assert channel_name == channel_name_from_index(domain_df)
        assert (step_name is None) == (step_num is None)

        logger.debug("Random: add_channel step_num %s step_name '%s'" % (step_num, step_name))

        if channel_name not in self.max_channel_steps:
            logger.warn("Random.add_channel unknown channel '%s'" % channel_name)

        if channel_name in self.channels:
            logger.debug("extending channel '%s' %s ids" % (channel_name, len(domain_df.index)))
            channel = self.channels[channel_name]
            channel.extend_domain(domain_df, step_name, step_num)

        else:
            logger.debug("adding channel '%s' %s ids" % (channel_name, len(domain_df.index)))

            max_steps = self.max_channel_steps.get(channel_name, 1)

            channel = SimpleChannel(channel_name,
                                    self.base_seed,
                                    domain_df,
                                    max_steps,
                                    step_name,
                                    step_num
                                    )

            self.channels[channel_name] = channel

    def get_channels(self):

        return [SavedChannelState(channel_name=channel_name,
                                  step_num=c.step_num,
                                  step_name=c.step_name)
                for channel_name, c in self.channels.iteritems()]

    def load_channels(self, saved_channels):

        for channel_state in saved_channels:

            logger.debug("load_channels channel %s" % (channel_state.channel_name,))

            if channel_state.channel_name == 'tours':
                for table_name in ["non_mandatory_tours", "mandatory_tours"]:
                    if orca.is_table(table_name):
                        df = orca.get_table(table_name).local
                        self.add_channel(df,
                                         channel_name=channel_state.channel_name,
                                         step_num=channel_state.step_num,
                                         step_name=channel_state.step_name)
            else:
                df = orca.get_table(channel_state.channel_name).local
                self.add_channel(df,
                                 channel_name=channel_state.channel_name,
                                 step_num=channel_state.step_num,
                                 step_name=channel_state.step_name)

    # random number generation

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

    def set_multi_choice_offset(self, df, offset):
        """
        Specify a fixed offset into the df channel's random number streams to use for all calls
        made to choice_for_df for the duration of the current step.

        A value of None means that a different set of random values should be used for each
        subsequent call to choice_for_df (for the same df row index)

        Recall that since each row in the df has its own distinct random stream,
        this means that each random stream is offset by the specified amount.

        This method has no particular utility if choice_for_df is only called once for each
        domain_df row, other than to (potentially) make subsequent calls to random_for_df
        faster if choice_for_df consumes a large number of random numbers, as random_for_df
        will not need to fast-forward as much.

        This method must be invoked before any random numbers are consumed in hte current step.
        The multi_choice_offset is reset to the default (None) at the beginning of each step.

        If "true pseudo random" behavior is desired (i.e. NOT repeatable) the set_base_seed
        method (q.v.) may be used to globally reseed all random streams.

        Parameters
        ----------
        df : pandas.dataframe
        offset : int or None
            absolute integer offset to fast-forward to in random streams in choice_for_df
        """

        channel = self.get_channel_for_df(df)
        channel.set_multi_choice_offset(offset, self.step_name)
        logging.info("set_multi_choice_offset to %s for channel %s"
                     % (channel.multi_choice_offset, channel.name))

    def random_for_df(self, df):
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
            rands = [rng.rand(1) for _ in range(len(df))]
            return rands

        t0 = print_elapsed_time()
        channel = self.get_channel_for_df(df)
        rands = channel.random_for_df(df, self.step_name)
        t0 = print_elapsed_time("random_for_df for %s rows" % len(df.index), t0, debug=True)
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

        Depending on the value of the multi_choice_offset setting
        (set by calling set_multi_choice_offset method, q,v,)
        for subsequent calls in the same step, this routine will
        EITHER use the same rand sequence for choosing values
        OR use fresh random values for choices.

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
