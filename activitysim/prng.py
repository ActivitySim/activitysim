import numpy as np
import pandas as pd

from tracing import print_elapsed_time


import logging

logger = logging.getLogger(__name__)


_MAX_ID = (1<<31)

import collections


class Channel(object):

    def __init__(self, offset, max_offset, prngs):
        self.offset = offset
        self.max_offset = max_offset
        self.prngs = prngs


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
    # FIXME - should get this form alts table
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


    def add_channel(self, df, channel_name):

        logger.info("prng add_channel '%s' %s ids" % (channel_name, len(df.index)))

        if channel_name not in self.max_offsets:
            logger.warn("Prng.add_channel max_seed_opffset not found in self.max_offsets")

        max_seed_offset = self.max_offsets.get(channel_name, 0)

        max_seed = df.index.max() * (max_seed_offset + 1)
        if max_seed >= _MAX_ID:
            msg = "max_seed  %s too big for unsigned int32" % (max_seed, )
            raise RuntimeError(msg)

        t0 = print_elapsed_time()

        prngs = pd.DataFrame(index=df.index)
        prngs['seed'] = prngs.index * max_seed_offset
        prngs['generator'] = [np.random.RandomState(seed) for seed in prngs['seed']]

        self.channels[channel_name] = Channel(offset=0, max_offset=max_seed_offset, prngs=prngs)

        print_elapsed_time('Prng.add_index %s' % (channel_name), t0)


    def prngs_for_tour_channel(self, tours, max_seed_offset):

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

        prngs['seed'] = (tours.person_id*sub_channel_count)+ channel_offset

        print "prngs['seed']", prngs['seed']

        # make room for seed offsets
        prngs.seed = (prngs.seed * max_seed_offset).astype(int)

        prngs['generator'] = [np.random.RandomState(seed) for seed in prngs['seed']]

        return prngs


    def add_tour_channels(self, df):

        channel_name = 'tours'

        if channel_name not in self.max_offsets:
            logger.warn("Prng.add_channel max_seed_opffset not found in self.max_offsets")

        max_seed_offset = self.max_offsets.get(channel_name, 0)

        t0 = print_elapsed_time()
        prngs = self.prngs_for_tour_channel(df, max_seed_offset)
        print_elapsed_time('Prng.add_tour_channels %s' % (channel_name), t0)

        if channel_name in self.channels:

            logger.info("prng add_tour_channels - extending %s tours" % len(df.index))

            channel = self.channels[channel_name]

            assert channel.max_offset == max_seed_offset

            assert len(channel.prngs.index.intersection(prngs.index)) == 0

            channel.prngs = pd.concat([channel.prngs, prngs])

        else:

            logger.info("prng add_tour_channels - first time")

            self.channels[channel_name] = Channel(offset=0, max_offset=max_seed_offset, prngs=prngs)


    def reseed(self, channel_name, offset):

        logger.info("prng reseed '%s' offset %s" % (channel_name, offset))

        assert channel_name in self.channels

        channel = self.channels[channel_name]
        assert channel.offset < channel.max_offset

        channel.offset += 1

        offset = channel.offset
        for row in channel.prngs.generators.itertuples():
            row.generator.seed(row.seed + offset)


    def choice(self, df, id, a, size=None, replace=True, p=None):

        if not self.channels:
            return np.random.choice(a, size, replace, p)

        channel_name = get_df_channel_name(df)

        if channel_name in self.channels:

            generator = self.channels[channel_name].prngs.loc[id].generator
            return generator.choice(a, size, replace, p)

        else:
            return np.random.choice(a, size, replace, p)


    def random_for_df(self, df):

        if not self.channels:
            return np.random.random((len(df), 1))

        channel_name = get_df_channel_name(df)
        count = 1

        if channel_name in self.channels:

            print "\nFOUND random_for_df channel_name '%s'\n" % (channel_name, )

            t0 = print_elapsed_time()

            ids = df.index
            prngs = self.channels[channel_name].prngs

            if channel_name=='tours':
                print prngs
                print "type(prngs) %s" % type(prngs)
                print prngs.ix[ids]

            generators = prngs.ix[ids].generator

            if channel_name=='tours':
                print "type(generators) %s" % type(generators)

            r = [prng.rand(count) for prng in generators]

            print_elapsed_time('Prng.random', t0)

            return r

        else:
            print "\nNOT FOUND random_for_df channel_name '%s'\n" % (channel_name, )

            return np.random.random((len(df), count))


