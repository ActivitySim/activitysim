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

    def set_base_seed(self, seed=None):

        if self.step_name is not None:
            raise RuntimeError("Can only call set_base_seed before the first step.")

        assert len(self.channels.keys()) == 0

        if seed is None:
            self.base_seed = np.random.RandomState().randint(_MAX_SEED)
            logger.info("Set random seed randomly to %s" % self.base_seed)
        else:
            logger.info("Set random seed base to %s" % seed)
            self.base_seed = seed

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

    def add_channel(self, domain_df, channel_name, step_name=None, step_num=None):

        assert channel_name == channel_name_from_index(domain_df)

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

    def get_channel_for_df(self, df):
        channel_name = channel_name_from_index(df)
        assert channel_name in self.channels
        return self.channels[channel_name]

    # random number generation

    def get_global_rng(self):
        assert self.step_name is not None
        return self.global_rng

    def set_multi_choice_offset(self, df, offset):
        channel = self.get_channel_for_df(df)
        channel.set_multi_choice_offset(offset, self.step_name)
        logging.info("set_multi_choice_offset to %s for channel %s"
                     % (channel.multi_choice_offset, channel.name))

    def random_for_df(self, df):

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
