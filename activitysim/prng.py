import numpy as np
import pandas as pd
import orca

from tracing import print_elapsed_time


import logging

logger = logging.getLogger(__name__)


_MAX_ID = (1<<31)
_NO_STEP = 'no_step'

import collections
SavedChannelState = collections.namedtuple('SavedChannelState', 'channel_name offset step_name')


class Channel(object):

    def __init__(self, offset, max_offset, prngs=None, step_name=None):
        self.offset = offset
        self.max_offset = max_offset
        self.prngs = prngs
        self.step_name = step_name


def get_df_channel_name(df):

    index_name = df.index.name

    if index_name == 'PERID': # or slicer == orca.get_injectable('persons_index_name'):
        channel_name = 'persons'
    elif index_name == 'HHID': # or slicer == orca.get_injectable('hh_index_name'):
        channel_name = 'households'
    elif index_name == 'tour_id':
        channel_name = 'tours'
    else:
        raise RuntimeError("get_df_channel_name - cant determine get_df_channel_name - index is %s" % (index_name,))

    return channel_name


def canonical_tour_sub_channels():

    # the problem is we don't know what the possible tour_types and their max tour_nums are

    # FIXME - should get this from alts table
    # alts = orca.get_table('non_mandatory_tour_frequency_alts').local
    # non_mandatory_tour_flavors = {c : alts[alts].max() for c in alts.columns.names}
    non_mandatory_tour_flavors = {'escort': 2, 'shopping': 1, 'othmaint': 1, 'othdiscr': 1,
                                  'eatout': 1, 'social': 1}

    # this logic is hardwired in process_mandatory_tours()
    mandatory_tour_flavors = {'work': 2, 'school': 2}

    tour_flavors = dict(non_mandatory_tour_flavors, **mandatory_tour_flavors)

    sub_channels = [tour_type + str(tour_num)
                    for tour_type, max_count in tour_flavors.iteritems()
                    for tour_num in range(1, max_count + 1)]

    sub_channels.sort()
    return sub_channels


class Prng(object):

    def __init__(self, max_offsets):

        self.channels = {}
        self.max_offsets = max_offsets
        self.current_step = _NO_STEP


    def reseed_if_necessary(self, channel_name, caller):

        assert self.current_step  and self.current_step != _NO_STEP
        assert channel_name and channel_name in self.channels

        channel = self.channels[channel_name]

        logger.debug("reseed_if_necessary channel '%s' caller '%s'" % (channel_name, caller))

        if channel.step_name == _NO_STEP:

            # channel has never been used - no need to reseed
            logger.debug("reseed_if_necessary - First use of channel '%s' step '%s' offset %s" % (channel_name, self.current_step, channel.offset))
            channel.step_name = self.current_step

        elif channel.step_name != self.current_step:

            # channel was last used on a previous step - need to reseed

            assert channel.offset < channel.max_offset

            channel.step_name = self.current_step
            channel.offset += 1

            logger.debug("reseed_if_necessary - Reseeding channel '%s' step '%s' offset %s max_offset %s"
                         % (channel_name, self.current_step, channel.offset, channel.max_offset))

            for row in channel.prngs.itertuples():
                row.generator.seed(row.seed + channel.offset)

        else:
            logger.debug("reseed_if_necessary - Reuse of channel '%s' step '%s' offset %s" % (channel_name, self.current_step, channel.offset))


    def generators_for_df(self, df):

        if not self.channels:
            return None

        channel_name = get_df_channel_name(df)

        if not channel_name in self.channels:
            print "couldn't find channel '%s' in %s" % (channel_name, self.channels.keys())

        assert channel_name in self.channels


        self.reseed_if_necessary(channel_name, 'generators_for_df')

        prngs = self.channels[channel_name].prngs
        generators = prngs.ix[df.index].generator

        assert not generators.isnull().any()

        return generators


    def random_for_df(self, df):

        if not self.channels:
            return np.random.random((len(df), 1))

        channel_name = get_df_channel_name(df)

        if channel_name in self.channels:

            self.reseed_if_necessary(channel_name, 'random_for_df')

            prngs = self.channels[channel_name].prngs
            generators = prngs.ix[df.index].generator

            # this will raise error if any df.index values not in prngs.index (rows all NaNs)
            r = [prng.rand(1) for prng in generators]

            # logger.info("dbgprng - random_for_df '%s' step '%s' offset %s" % (channel_name, self.current_step, self.channels[channel_name].offset))
            # logger.info("dbgprng - random_for_df %s" % ([item for sublist in r for item in sublist],))

            return r

        else:
            # if any channels have been added, then all channels should have been
            assert not self.channels

            return np.random.random((len(df), 1))

    def begin_step(self, step_name):

        assert self.current_step == _NO_STEP
        assert step_name and step_name != _NO_STEP

        self.current_step = step_name

        print "\n##### begin_step %s" % step_name

    def end_step(self, step_name):

        assert self.current_step and self.current_step != _NO_STEP
        assert self.current_step == step_name

        self.current_step = _NO_STEP

        print "\n##### end_step %s" % step_name
        print self.get_channels()


    def create_prngs_for_tour_channels(self, tours, max_seed_offset, offset=0):

        assert 'person_id' in tours.columns.values
        assert 'tour_type' in tours.columns.values
        assert 'tour_num' in tours.columns.values

        sub_channels = canonical_tour_sub_channels()
        sub_channel_count = len(sub_channels)

        max_seed = tours.index.max() * sub_channel_count * (max_seed_offset + 1)
        if max_seed >= _MAX_ID:
            msg = "max_seed  %s too big for unsigned int32" % (max_seed, )
            raise RuntimeError(msg)

        prngs = pd.DataFrame(index=tours.index)

        # concat tour_type + tour_num
        channel_offset = tours.tour_type + tours.tour_num.map(str)
        # map recognized strings to ints
        channel_offset = channel_offset.replace(to_replace=sub_channels, value=range(sub_channel_count))
        # convert to numeric - shouldn't be any NaNs - this will raise error if there are
        channel_offset = pd.to_numeric(channel_offset, errors='coerce').astype(int)

        prngs['seed'] = (tours.person_id*sub_channel_count) + channel_offset

        # make room for max_seed_offset offsets
        prngs.seed = (prngs.seed * max_seed_offset).astype(int)

        prngs['generator'] = [np.random.RandomState(seed + offset) for seed in prngs['seed']]

        return prngs

    def create_prngs_for_channel(self, df, max_seed_offset, offset=0):

        max_seed = df.index.max() * (max_seed_offset + 1)
        if max_seed >= _MAX_ID:
            msg = "max_seed  %s too big for unsigned int32" % (max_seed, )
            raise RuntimeError(msg)

        prngs = pd.DataFrame(index=df.index)
        prngs['seed'] = (prngs.index * max_seed_offset)
        prngs['generator'] = [np.random.RandomState(seed + offset) for seed in prngs['seed']]

        return prngs


    def add_channel(self, df, channel_name, offset=0, step_name=None):

        # only passed these if we are reloading
        if step_name is None:
            step_name = self.current_step

        logger.info("prng add_channel '%s' %s ids" % (channel_name, len(df.index)))
        logger.info("prng add_channel offset %s step_name '%s'" % (offset, step_name))

        if channel_name not in self.max_offsets:
            logger.warn("Prng.add_channel max_seed_opffset not found in self.max_offsets")

        max_seed_offset = self.max_offsets.get(channel_name, 0)
        assert(offset <= max_seed_offset)

        t0 = print_elapsed_time()

        if channel_name == 'tours':
            prngs = self.create_prngs_for_tour_channels(df, max_seed_offset, offset)
        else:
            prngs = self.create_prngs_for_channel(df, max_seed_offset, offset)


        if channel_name in self.channels:
            logger.info("prng add_channel - extending %s " % len(df.index))
            channel = self.channels[channel_name]
            assert channel.max_offset == max_seed_offset
            assert len(channel.prngs.index.intersection(prngs.index)) == 0
            channel.prngs = pd.concat([channel.prngs, prngs])
        else:
            logger.info("prng add_tour_channels - first time")
            self.channels[channel_name] = \
                Channel(offset=offset, max_offset=max_seed_offset, prngs=prngs, step_name=step_name)

        print_elapsed_time('Prng.add_channel %s' % (channel_name), t0)


    def add_tour_channels(self, df):

        # FIXME - should have a better mechanism - this is to avoid breaking when called from test
        if not self.channels:
            return

        self.add_channel(df, 'tours')


    def get_channels(self):

        return [SavedChannelState(channel_name=channel_name, offset=c.offset, step_name=c.step_name)
                for channel_name, c in self.channels.iteritems()]

        return { channel_name: c.offset for channel_name, c in self.channels.iteritems() }


    def load_channels(self, saved_channels):


        for channel_state in saved_channels:

            assert channel_state.channel_name not in self.channels

            if channel_state.channel_name == 'tours':

                for table_name in ["non_mandatory_tours", "mandatory_tours"]:
                    if orca.is_table(table_name):
                        df = orca.get_table(table_name).local
                        self.add_channel(df, channel_name=channel_state.channel_name, offset=channel_state.offset, step_name=channel_state.step_name)

            else:
                df = orca.get_table(channel_state.channel_name).local
                self.add_channel(df, channel_name=channel_state.channel_name, offset=channel_state.offset, step_name=channel_state.step_name)
