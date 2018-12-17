# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402

from future.utils import iteritems

import logging
import time
import multiprocessing
import ctypes
import os

from collections import OrderedDict

import numpy as np
import pandas as pd

from activitysim.core import inject
from activitysim.core import util
from activitysim.core import config
from activitysim.core import tracing

from activitysim.abm.tables.size_terms import tour_destination_size_terms


logger = logging.getLogger(__name__)

# - reverse semaphores to synchronize concurrent access to shared data buffer
# we use the first two rows of the final column in numpy-wrapped shared data as 'reverse semaphores'
# (synchronize concurrent access to shared data resource rather than throttling access)
TALLY_CHECKIN = (0, -1)
TALLY_CHECKOUT = (1, -1)


def size_table_name(selector, scaled=True):
    if scaled:
        table_name = "scaled_%s_destination_size" % selector
    else:
        table_name = "raw_%s_destination_size" % selector
    return table_name


def get_size_table(selector, scaled=True):
    return inject.get_table(size_table_name(selector, scaled)).to_frame()


USE_RAW_SIZE = True


class ShadowPriceCalculator(object):

    def __init__(self, model_settings, shared_data=None, shared_data_lock=None):

        self.use_shadow_pricing = bool(config.setting('use_shadow_pricing'))
        self.use_saved_shadow_prices = bool(config.setting('use_saved_shadow_prices'))

        self.selector = model_settings['SELECTOR']

        full_model_run = config.setting('households_sample_size') == 0
        if self.use_shadow_pricing and not full_model_run:
            logging.warning("deprecated combination of use_shadow_pricing and not full_model_run")

        self.segment_ids = model_settings['SEGMENT_IDS']

        # - modeled_size (set by call to set_choices/synchronize_choices)
        self.modeled_size = None

        # - convergence criteria for check_fit
        # ignore criteria for zones smaller than size_threshold
        self.size_threshold = model_settings['SIZE_THRESHOLD']
        # zone passes if modeled is within percent_tolerance of  predicted_size
        self.percent_tolerance = model_settings['PERCENT_TOLERANCE']
        # max percentage of zones allowed to fail
        self.fail_threshold = model_settings['FAIL_THRESHOLD']

        # - destination_size_table (predicted_size)
        if USE_RAW_SIZE:
            self.raw_predicted_size = get_size_table(self.selector, scaled=False)
        self.predicted_size = get_size_table(self.selector, scaled=True)

        # - shared_data
        if shared_data is not None:
            assert shared_data.shape[0] == self.predicted_size.shape[0]
            assert shared_data.shape[1] == self.predicted_size.shape[1] + 1  # tally column
            assert shared_data_lock is not None
        self.shared_data = shared_data
        self.shared_data_lock = shared_data_lock

        # - load saved shadow_prices (if available) and set max_iterations accordingly
        self.shadow_prices = None
        if self.use_shadow_pricing:
            if self.load_saved_shadow_prices:
                self.shadow_prices = self.load_saved_shadow_prices(model_settings)

            if self.shadow_prices is None:
                self.max_iterations = model_settings.get('MAX_SHADOW_PRICE_ITERATIONS', 5)
            else:
                self.max_iterations = model_settings.get('MAX_SHADOW_PRICE_ITERATIONS_SAVED', 1)
        else:
            self.max_iterations = 1

        # - if we did't load saved shadow_prices, initialize shadow_prices to all ones
        # this will start first iteration with no shadow price adjustment,
        if self.shadow_prices is None:
            self.shadow_prices = \
                pd.DataFrame(data=1.0,
                             columns=self.predicted_size.columns,
                             index=self.predicted_size.index)

        self.num_fail = pd.DataFrame(index=self.predicted_size.columns)
        self.max_abs_diff = pd.DataFrame(index=self.predicted_size.columns)
        self.max_rel_diff = pd.DataFrame(index=self.predicted_size.columns)

    def load_saved_shadow_prices(self, model_settings):

        shadow_prices = None

        # - load saved shadow_prices
        saved_shadow_price_file_name = model_settings.get('SAVED_SHADOW_PRICE_TABLE_NAME')
        if saved_shadow_price_file_name:
            # FIXME - where should we look for this file?
            file_path = config.data_file_path(saved_shadow_price_file_name, mandatory=False)
            if file_path:
                shadow_prices = pd.read_csv(file_path, index_col=0)
                logging.info("loading saved_shadow_prices from %s" % (file_path))
            else:
                logging.warning("Could not find saved_shadow_prices file %s" % (file_path))

        return shadow_prices

    def synchronize_choices(self, local_modeled_size):

        if self.shared_data is None:
            return local_modeled_size

        num_processes = inject.get_injectable("num_processes")

        def get_tally(t):
            with self.shared_data_lock:
                return self.shared_data[t]

        def wait(tally, target, tally_name):
            while get_tally(tally) != target:
                time.sleep(1)

        # - nobody checks in until checkout clears
        wait(TALLY_CHECKOUT, 0, 'TALLY_CHECKOUT')

        # - add local_modeled_size data
        with self.shared_data_lock:
            first_in = self.shared_data[TALLY_CHECKIN] == 0
            # add local data from df to shared data buffer
            # final column is used for tallys, hence the negative index
            self.shared_data[..., 0:-1] += local_modeled_size.values
            self.shared_data[TALLY_CHECKIN] += 1

        # - wait until everybody else has checked in
        wait(TALLY_CHECKIN, num_processes, 'TALLY_CHECKIN')

        # - copy shared data and check out
        with self.shared_data_lock:
            logger.info("copy shared_data")
            # numpy array with sum of local_modeled_size.values from all processes
            global_modeled_size_array = self.shared_data[..., 0:-1].copy()
            self.shared_data[TALLY_CHECKOUT] += 1

        # - first in waits until all other processes have checked out, and cleans tub
        if first_in:
            wait(TALLY_CHECKOUT, num_processes, 'TALLY_CHECKOUT')
            with self.shared_data_lock:
                self.shared_data[:] = 0
            logger.info("first_in clearing shared_data")

        # convert summed numpy array data to conform to original dataframe
        return pd.DataFrame(data=global_modeled_size_array,
                            index=local_modeled_size.index,
                            columns=local_modeled_size.columns)

    def set_choices(self, choices_df):

        assert 'dest_choice' in choices_df

        modeled_size = pd.DataFrame(index=self.predicted_size.index)
        for c in self.predicted_size:
            segment_choices = \
                choices_df[choices_df['segment_id'] == self.segment_ids[c]]
            modeled_size[c] = segment_choices.groupby('dest_choice').size()
        modeled_size = modeled_size.fillna(0).astype(int)

        self.modeled_size = self.synchronize_choices(modeled_size)

    def check_fit(self, iteration):
        """
        Check convergence criteria fit of modeled_size to target predicted_size
        (For multiprocessing, this is global modeled_size summed across processes,
        so each process will independently calculate the same result.)

        Parameters
        ----------
        iteration: int
            iteration number (informational, for num_failand max_diff history columns)

        Returns
        -------
        converged: boolean

        """

        assert self.modeled_size is not None
        assert self.predicted_size is not None

        modeled_size = self.modeled_size
        predicted_size = self.predicted_size

        abs_diff = (predicted_size - modeled_size).abs()

        rel_diff = abs_diff / modeled_size

        # ignore zones where predicted_size < threshold
        rel_diff.where(predicted_size >= self.size_threshold, 0, inplace=True)

        # ignore zones where rel_diff < percent_tolerance
        rel_diff.where(rel_diff > (self.percent_tolerance / 100.0), 0, inplace=True)

        self.num_fail['iter%s' % iteration] = (rel_diff > 0).sum()
        self.max_abs_diff['iter%s' % iteration] = abs_diff.max()
        self.max_rel_diff['iter%s' % iteration] = rel_diff.max()

        total_fails = (rel_diff > 0).values.sum()

        max_fail = (self.fail_threshold / 100.0) * predicted_size.shape[0]

        converged = (total_fails <= max_fail)

        # for c in predicted_size:
        #     print("check_fit %s segment %s" % (self.selector, c))
        #     print("  modeled %s" % (modeled_size[c].sum()))
        #     print("  predicted %s" % (predicted_size[c].sum()))
        #     print("  max abs diff %s" % (abs_diff[c].max()))
        #     print("  max rel diff %s" % (rel_diff[c].max()))

        logging.info("check_fit %s iteration: %s converged: %s max_fail: %s total_fails: %s" %
                     (self.selector, iteration, converged, max_fail, total_fails))

        return converged

    def update_shadow_prices(self):
        """
        CTRAMP:
        if ( modeledDestinationLocationsByDestZone > 0 )
            shadowPrice *= ( scaledSize / modeledDestinationLocationsByDestZone );
        // else
        //    shadowPrice *= scaledSize;

        Daysim:
        targ = prediction > total
            ? Math.Min(prediction,
                       Math.Min(total * (1 + percentTolerance / 100D), total + absoluteTolerance))
            : Math.Max(prediction,
                       Math.Max(total * (1 - percentTolerance / 100D), total - absoluteTolerance));

        shadowPrice =
            previousShadowPrice + Math.Log(Math.Max(targ, .01) * 1D / Math.Max(prediction, .01));
        """

        assert self.use_shadow_pricing

        # can't update_shadow_prices until after first iteration
        # modeled_size should have been set by set_choices at end of previous iteration
        assert self.modeled_size is not None
        assert self.predicted_size is not None
        assert self.shadow_prices is not None

        new_scale_factor = self.predicted_size / self.modeled_size

        # FIXME - need to decide if following CTRAMP code quoted above, and if so, which version
        # following CTRAMP (original version - later commented out)
        # avoid zero-divide for 0 modeled_size, by setting scale_factor same as modeled_size of 1
        # new_scale_factor.where(self.modeled_size > 0, self.predicted_size)

        new_shadow_prices = self.shadow_prices * new_scale_factor

        # following CTRAMP (revised version - with 0 dest zone case lines commented out)
        # avoid zero-divide for 0 modeled_size, by leaving shadow_prices unchanged
        new_shadow_prices.where(self.modeled_size > 0, self.shadow_prices, inplace=True)

        # print("\nself.predicted_size\n", self.predicted_size.head())
        # print("\nself.modeled_size\n", self.modeled_size.head())

        self.shadow_prices = new_shadow_prices

    def shadow_price_adjusted_predicted_size(self):

        if USE_RAW_SIZE:
            return self.raw_predicted_size * self.shadow_prices
        else:
            return self.predicted_size * self.shadow_prices

    def write_trace_files(self, iteration):
        logger.info("write_trace_files iteration %s" % iteration)
        if iteration == 0:
            tracing.write_csv(self.predicted_size,
                              'shadow_price_%s_predicted_size_%s' % (self.selector, iteration),
                              transpose=False)
        tracing.write_csv(self.modeled_size,
                          'shadow_price_%s_modeled_size_%s' % (self.selector, iteration),
                          transpose=False)
        tracing.write_csv(self.shadow_prices,
                          'shadow_price_%s_shadow_prices_%s' % (self.selector, iteration),
                          transpose=False)


def block_name(selector):
    return selector


def get_shadow_pricing_info():

    land_use = inject.get_table('land_use')
    size_terms = inject.get_injectable('size_terms')

    # shadow_pricing_models is dict of {<selector>: <model_name>}
    shadow_pricing_models = config.setting('shadow_pricing_models')

    blocks = OrderedDict()
    for selector in shadow_pricing_models:

        sp_rows = len(land_use)
        sp_cols = len(size_terms[size_terms.selector == selector])

        # extra tally column
        blocks[block_name(selector)] = (sp_rows, sp_cols + 1)

    sp_dtype = np.int64

    shadow_pricing_info = {
        'dtype': sp_dtype,
        'blocks': blocks,
    }

    return shadow_pricing_info


def buffers_for_shadow_pricing(shadow_pricing_info, shared=False):

    assert shared

    dtype = shadow_pricing_info['dtype']
    blocks = shadow_pricing_info['blocks']

    data_buffers = {}
    for block_key, block_shape in iteritems(blocks):

        # buffer_size must be int (or p2.7 long), not np.int64
        buffer_size = int(np.prod(block_shape))

        csz = buffer_size * np.dtype(dtype).itemsize
        logger.info("allocating shared buffer %s %s buffer_size %s (%s)" %
                    (block_key, buffer_size, block_shape, util.GB(csz)))

        if np.issubdtype(dtype, np.int64):
            typecode = ctypes.c_long
        else:
            raise RuntimeError("buffer_for_shadow_pricing unrecognized dtype %s" % dtype)

        buffer = multiprocessing.Array(typecode, buffer_size)

        logger.info("buffer_for_shadow_pricing added block %s" % (block_key))

        data_buffers[block_key] = buffer

    return data_buffers


def shadow_price_data_from_buffers(data_buffers, shadow_pricing_info, selector):

    assert type(data_buffers) == dict

    dtype = shadow_pricing_info['dtype']
    blocks = shadow_pricing_info['blocks']

    if selector not in blocks:
        raise RuntimeError("Selector %s not in shadow_pricing_info" % selector)

    if block_name(selector) not in data_buffers:
        raise RuntimeError("Block %s not in data_buffers" % block_name(selector))

    shape = blocks[selector]
    data = data_buffers[block_name(selector)]

    return np.frombuffer(data.get_obj(), dtype=dtype).reshape(shape), data.get_lock()


def load_shadow_price_calculator(model_settings):

    selector = model_settings['SELECTOR']

    # - data_buffers (if shared data)
    data_buffers = inject.get_injectable('data_buffers', None)
    if data_buffers is not None:
        logger.info('Using existing data_buffers for shadow_price')

        # - shadow_pricing_info
        shadow_pricing_info = inject.get_injectable('shadow_pricing_info', None)
        if shadow_pricing_info is None:
            shadow_pricing_info = get_shadow_pricing_info()
            inject.add_injectable('shadow_pricing_info', shadow_pricing_info)

        # - extract data buffer and reshape as numpy array
        data, lock = shadow_price_data_from_buffers(data_buffers, shadow_pricing_info, selector)
    else:
        data = None  # ShadowPriceCalculator will allocate its own data
        lock = None

    # - ShadowPriceCalculator
    spc = ShadowPriceCalculator(
        model_settings,
        data, lock)

    return spc


def add_predicted_size_table():

    shadow_pricing_models = config.setting('shadow_pricing_models')

    if shadow_pricing_models is None:
        logger.warning('add_predicted_size_table: shadow_pricing_models not in settings')
        return

    # shadow_pricing_models is dict of {<selector>: <model_name>}
    # since these are scaled to model size, they have to be created while single-process

    for selector, model_name in iteritems(shadow_pricing_models):

        model_settings = config.read_model_settings(model_name)

        assert selector == model_settings['SELECTOR']

        segment_ids = model_settings['SEGMENT_IDS']
        chooser_table_name = model_settings['CHOOSER_TABLE_NAME']
        chooser_segment_column = model_settings['CHOOSER_SEGMENT_COLUMN']

        choosers_df = inject.get_table(chooser_table_name).to_frame()
        if 'CHOOSER_FILTER_COLUMN' in model_settings:
            choosers_df = choosers_df[choosers_df[model_settings['CHOOSER_FILTER_COLUMN']] != 0]

        # - raw_predicted_size
        land_use = inject.get_table('land_use')
        size_terms = inject.get_injectable('size_terms')
        raw_size = tour_destination_size_terms(land_use, size_terms, selector).astype(np.float64)
        assert set(raw_size.columns) == set(segment_ids.keys())

        if USE_RAW_SIZE:
            inject.add_table(size_table_name(selector, scaled=False), raw_size)

        # - global number of choosers in each segment
        segment_chooser_counts = \
            {segment_name: (choosers_df[chooser_segment_column] == segment_id).sum()
             for segment_name, segment_id in iteritems(segment_ids)}

        # - segment scale factor (modeled / predicted) keyed by segment_name
        # scaling reconciles differences between synthetic population and zone demographics
        # in a partial sample, it also scales predicted_size targets to sample population
        segment_scale_factors = {}
        for c in raw_size:
            # number of zone demographics predicted destination choices
            segment_predicted_size = raw_size[c].astype(np.float64).sum()

            # number of synthetic population choosers in segment
            segment_chooser_count = (choosers_df[chooser_segment_column] == segment_ids[c]).sum()

            segment_scale_factors[c] = \
                segment_chooser_count / np.maximum(segment_predicted_size, 1)

            logger.info("add_predicted_size_table %s segment %s "
                        "predicted %s modeled %s scale_factor %s" %
                        (chooser_table_name, c,
                         segment_predicted_size,
                         segment_chooser_count,
                         segment_scale_factors[c]))

            # segment_scale_factors[c] = \
            #     segment_chooser_counts[c] / np.maximum(segment_predicted_size, 1)

        # - scaled_size = zone_size * (total_segment_modeled / total_segment_predicted)
        scaled_size = raw_size * segment_scale_factors
        inject.add_table(size_table_name(selector, scaled=True), scaled_size)
