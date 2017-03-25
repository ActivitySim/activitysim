import collections

import numpy as np
import pandas as pd
import orca

from tracing import print_elapsed_time, log_memory_info

import logging

logger = logging.getLogger(__name__)

_MAX_SEED = (1 << 32)


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

    def __init__(self, name, base_seed, domain_df, max_steps, step_name=None, step_num=None):

        assert channel_name_from_index(domain_df) == name

        self.name = name
        self.base_seed = base_seed

        self.step_name = step_name
        self.step_num = step_num if step_num is not None else -1
        self.max_steps = max_steps

        # dataframe to hold state for every df row
        self.row_states = pd.DataFrame(index=domain_df.index)

        # ensure that every channel is different, even for the same df index values and max_steps
        unique_channel_seed = hash(step_name) % _MAX_SEED

        self.row_states['row_seed'] = (base_seed + unique_channel_seed +
                                       self.row_states.index * max_steps) % _MAX_SEED

        self.begin_step(step_name)

    def begin_step(self, step_name):

        if self.step_name == step_name:
            return

        self.step_name = step_name
        self.step_num += 1

        assert self.step_num < self.max_steps

        # number of rands pulled this step
        self.row_states['offset'] = 0

        # number of choice calls this step
        self.row_states['choice_count'] = 0

        logger.info("begin_step '%s' for channel '%s'" % (step_name, self.name, ))

    def _generators_for_df(self, df, step_name):

        self.begin_step(step_name)

        t0 = print_elapsed_time()
        df_row_states = self.row_states.loc[df.index]
        #df_row_states = pd.merge(pd.DataFrame(index=df.index), row_state_df,
        #                         left_index=True, right_index=True, how="left")
        t0 = print_elapsed_time("create df_row_states", t0)

        # assert no dupes
        assert len(df_row_states.index.unique() == len(df.index))

        # not reusing a generator used for choice()
        assert not df_row_states.choice_count.any()

        prng = np.random.RandomState()
        for row in df_row_states.itertuples():

            seed = (row.row_seed + self.step_num) % _MAX_SEED

            prng.seed(seed)
            if row.offset > 0:
                # consume rands
                prng.rand(row.offset)

            yield prng

        t0 = print_elapsed_time("loop df_row_states", t0)

    def random_for_df(self, df, step_name):

        generators = self._generators_for_df(df, step_name)

        r = [prng.rand(1)[0] for prng in generators]

        self.row_states.loc[df.index, 'offset'] += 1

        return r

    def choice_for_df(self, df, step_name, a, size, replace):

        generators = self._generators_for_df(df, step_name)

        sample = np.concatenate(tuple(prng.choice(a, size, replace) for prng in generators))

        self.row_states.loc[df.index, 'choice_count'] += 1

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
        assert channel_name not in self.channels

        logger.debug("Random: add_channel '%s' %s ids" % (channel_name, len(domain_df.index)))
        logger.debug("Random: add_channel step_num %s step_name '%s'" % (step_num, step_name))

        if channel_name not in self.max_channel_steps:
            logger.warn("Prng.add_channel unknown channel '%s'" % channel_name)

        max_steps = self.max_channel_steps.get(channel_name, 1)
        assert(step_num < max_steps)

        channel = SimpleChannel(channel_name,
                                self.base_seed,
                                domain_df,
                                max_steps,
                                step_name,
                                step_num
                                )

        self.channels[channel_name] = channel

    def get_channels(self):

        SavedChannelState = collections.namedtuple('SavedChannelState',
                                                   'channel_name step_num step_name')

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
                                         offset=channel_state.offset,
                                         step_name=channel_state.step_name)
            else:
                df = orca.get_table(channel_state.channel_name).local
                self.add_channel(df,
                                 channel_name=channel_state.channel_name,
                                 offset=channel_state.offset,
                                 step_name=channel_state.step_name)

    # random number generation

    def get_global_rng(self):
        assert self.step_name is not None
        return self.global_rng

    def random_for_df(self, df):

        channel_name = channel_name_from_index(df)

        assert channel_name in self.channels

        channel = self.channels[channel_name]

        return channel.random_for_df(df, self.step_name)

    def choice_for_df(self, df, a, size, replace):

        channel_name = channel_name_from_index(df)

        assert channel_name in self.channels

        channel = self.channels[channel_name]

        return channel.choice_for_df(df, self.step_name, a, size, replace)
