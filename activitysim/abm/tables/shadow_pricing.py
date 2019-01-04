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

from collections import OrderedDict

import numpy as np
import pandas as pd

from activitysim.core import inject
from activitysim.core import util
from activitysim.core import config
from activitysim.core import tracing

from activitysim.abm.tables.size_terms import tour_destination_size_terms


logger = logging.getLogger(__name__)

"""
ShadowPriceCalculator and associated utility methods

See docstrings for documentation on:

update_shadow_prices    how shadow_price coefficients are calculated
synchronize_choices     interprocess communication to compute aggregate modeled_size
check_fit               convergence criteria for shadow_pric iteration

"""


"""
Artisanal reverse semaphores to synchronize concurrent access to shared data buffer

we use the first two rows of the final column in numpy-wrapped shared data as 'reverse semaphores'
(they synchronize concurrent access to shared data resource rather than throttling access)

ShadowPriceCalculator.synchronize_choices coordinates access to the global aggregate zone counts
(local_modeled_size summed across all sub-processes) using these two semaphores
(which are really only tuples of indexes of locations in the shared data array.
"""
TALLY_CHECKIN = (0, -1)
TALLY_CHECKOUT = (1, -1)


def size_table_name(selector):
    """
    Returns canonical destination size table name

    Parameters
    ----------
    selector : str
        e.g. school or workplace

    Returns
    -------
    table_name : str
    """
    return "%s_destination_size" % selector


def get_size_table(selector):
    return inject.get_table(size_table_name(selector)).to_frame()


class ShadowPriceCalculator(object):

    def __init__(self, model_settings, shared_data=None, shared_data_lock=None):
        """

        Presence of shared_data is used as a flag for multiprocessing
        If we are multiprocessing, shared_data should be a multiprocessing.RawArray buffer
        to aggregate modeled_size across all sub-processes, and shared_data_lock should be
        a multiprocessing.Lock object to coordinate access to that buffer.

        Optionally load saved shadow_prices from data_dir if config setting use_shadow_pricing
        and shadow_setting LOAD_SAVED_SHADOW_PRICES are both True

        Parameters
        ----------
        model_settings : dict
        shared_data : multiprocessing.Array or None (if single process)
        shared_data_lock : numpy array wrapping multiprocessing.RawArray or None (if single process)
        """

        self.use_shadow_pricing = bool(config.setting('use_shadow_pricing'))
        self.saved_shadow_price_file_path = None  # set by read_saved_shadow_prices if loaded

        self.selector = model_settings['SELECTOR']

        full_model_run = config.setting('households_sample_size') == 0
        if self.use_shadow_pricing and not full_model_run:
            logging.warning("deprecated combination of use_shadow_pricing and not full_model_run")

        self.segment_ids = model_settings['SEGMENT_IDS']

        # - modeled_size (set by call to set_choices/synchronize_choices)
        self.modeled_size = None

        if self.use_shadow_pricing:
            self.shadow_settings = config.read_model_settings('shadow_pricing.yaml')

        # - destination_size_table (predicted_size)
        self.predicted_size = get_size_table(self.selector)

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
            if self.shadow_settings['LOAD_SAVED_SHADOW_PRICES']:
                # read_saved_shadow_prices logs error and returns None if file not found
                self.shadow_prices = self.read_saved_shadow_prices(model_settings)

            if self.shadow_prices is None:
                self.max_iterations = self.shadow_settings.get('MAX_ITERATIONS', 5)
            else:
                self.max_iterations = self.shadow_settings.get('MAX_ITERATIONS_SAVED', 1)
        else:
            self.max_iterations = 1

        # - if we did't load saved shadow_prices, initialize all shadow_prices to one
        # this will start first iteration with no shadow price adjustment,
        if self.shadow_prices is None:
            self.shadow_prices = \
                pd.DataFrame(data=1.0,
                             columns=self.predicted_size.columns,
                             index=self.predicted_size.index)

        self.num_fail = pd.DataFrame(index=self.predicted_size.columns)
        self.max_abs_diff = pd.DataFrame(index=self.predicted_size.columns)
        self.max_rel_diff = pd.DataFrame(index=self.predicted_size.columns)

    def read_saved_shadow_prices(self, model_settings):
        """
        Read saved shadow_prices from csv file in data_dir (so-called warm start)
        returns None if no saved shadow price file name specified or named file not found

        Parameters
        ----------
        model_settings : dict

        Returns
        -------
        shadow_prices : pandas.DataFrame or None
        """

        shadow_prices = None

        # - load saved shadow_prices
        saved_shadow_price_file_name = model_settings.get('SAVED_SHADOW_PRICE_TABLE_NAME')
        if saved_shadow_price_file_name:
            # FIXME - where should we look for this file?
            file_path = config.data_file_path(saved_shadow_price_file_name, mandatory=False)
            if file_path:
                shadow_prices = pd.read_csv(file_path, index_col=0)
                self.saved_shadow_price_file_path = file_path  # informational
                logging.info("loaded saved_shadow_prices from %s" % (file_path))
            else:
                logging.warning("Could not find saved_shadow_prices file %s" % (file_path))

        return shadow_prices

    def synchronize_choices(self, local_modeled_size):
        """
        We have to wait until all processes have computed choices and aggregated them by segment
        and zone before we can compute global aggregate zone counts (by segment). Since the global
        zone counts are in shared data, we have to coordinate access to the data structure across
        sub-processes.

        Note that all access to self.shared_data has to be protected by acquiring shared_data_lock

        ShadowPriceCalculator.synchronize_choices coordinates access to the global aggregate
        zone counts (local_modeled_size summed across all sub-processes).

        * All processes wait (in case we are iterating) until any stragglers from the previous
          iteration have exited the building. (TALLY_CHECKOUT goes to zero)

        * Processes then add their local counts into the shared_data and increment TALLY_CHECKIN

        * All processes wait until everybody has checked in (TALLY_CHECKIN == num_processes)

        * Processes make local copy of shared_data and check out (increment TALLY_CHECKOUT)

        * first_in process waits until all processes have checked out, then zeros shared_data
          and clears semaphores

        Parameters
        ----------
        local_modeled_size : pandas DataFrame


        Returns
        -------
        global_modeled_size_df : pandas DataFrame
            local copy of shared global_modeled_size data as dataframe
            with same shape and columns as local_modeled_size
        """

        # shouldn't be called if we are not multiprocessing
        assert self.shared_data is not None

        num_processes = inject.get_injectable("num_processes")
        assert num_processes > 1

        def get_tally(t):
            with self.shared_data_lock:
                return self.shared_data[t]

        def wait(tally, target, tally_name):
            while get_tally(tally) != target:
                time.sleep(1)

        # - nobody checks in until checkout clears
        wait(TALLY_CHECKOUT, 0, 'TALLY_CHECKOUT')

        # - add local_modeled_size data, increment TALLY_CHECKIN
        with self.shared_data_lock:
            first_in = self.shared_data[TALLY_CHECKIN] == 0
            # add local data from df to shared data buffer
            # final column is used for tallys, hence the negative index
            self.shared_data[..., 0:-1] += local_modeled_size.values
            self.shared_data[TALLY_CHECKIN] += 1

        # - wait until everybody else has checked in
        wait(TALLY_CHECKIN, num_processes, 'TALLY_CHECKIN')

        # - copy shared data, increment TALLY_CHECKIN
        with self.shared_data_lock:
            logger.info("copy shared_data")
            # numpy array with sum of local_modeled_size.values from all processes
            global_modeled_size_array = self.shared_data[..., 0:-1].copy()
            self.shared_data[TALLY_CHECKOUT] += 1

        # - first in waits until all other processes have checked out, and cleans tub
        if first_in:
            wait(TALLY_CHECKOUT, num_processes, 'TALLY_CHECKOUT')
            with self.shared_data_lock:
                # zero shared_data, clear TALLY_CHECKIN, and TALLY_CHECKOUT semaphores
                self.shared_data[:] = 0
            logger.info("first_in clearing shared_data")

        # convert summed numpy array data to conform to original dataframe
        global_modeled_size_df = \
            pd.DataFrame(data=global_modeled_size_array,
                         index=local_modeled_size.index,
                         columns=local_modeled_size.columns)

        return global_modeled_size_df

    def set_choices(self, choices_df):
        """
        aggregate individual location choices to modeled_size by zone and segment

        Parameters
        ----------
        choices_df : pandas.DataFrame
            dataframe with disaggregate location choices and at least two columns:
                'segment_id' : segment id tag for this individual
                'dest_choice' : zone id of location choice
        Returns
        -------
        updates self.modeled_size
        """

        assert 'dest_choice' in choices_df

        modeled_size = pd.DataFrame(index=self.predicted_size.index)
        for c in self.predicted_size:
            segment_choices = \
                choices_df[choices_df['segment_id'] == self.segment_ids[c]]
            modeled_size[c] = segment_choices.groupby('dest_choice').size()
        modeled_size = modeled_size.fillna(0).astype(int)

        if self.shared_data is None:
            # - not multiprocessing
            self.modeled_size = modeled_size
        else:
            # - if we are multiprocessing, we have to aggregate across sub-processes
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

        # fixme

        if not self.use_shadow_pricing:
            return False

        assert self.modeled_size is not None
        assert self.predicted_size is not None

        # - convergence criteria for check_fit
        # - convergence criteria for check_fit
        # ignore convergence criteria for zones smaller than size_threshold
        size_threshold = self.shadow_settings['SIZE_THRESHOLD']
        # zone passes if modeled is within percent_tolerance of  predicted_size
        percent_tolerance = self.shadow_settings['PERCENT_TOLERANCE']
        # max percentage of zones allowed to fail
        fail_threshold = self.shadow_settings['FAIL_THRESHOLD']

        modeled_size = self.modeled_size
        predicted_size = self.predicted_size

        abs_diff = (predicted_size - modeled_size).abs()

        rel_diff = abs_diff / modeled_size

        # ignore zones where predicted_size < threshold
        rel_diff.where(predicted_size >= size_threshold, 0, inplace=True)

        # ignore zones where rel_diff < percent_tolerance
        rel_diff.where(rel_diff > (percent_tolerance / 100.0), 0, inplace=True)

        self.num_fail['iter%s' % iteration] = (rel_diff > 0).sum()
        self.max_abs_diff['iter%s' % iteration] = abs_diff.max()
        self.max_rel_diff['iter%s' % iteration] = rel_diff.max()

        total_fails = (rel_diff > 0).values.sum()

        # FIXME - should not count zones where predicted_size < threshold? (could calc in init)
        max_fail = (fail_threshold / 100.0) * np.prod(predicted_size.shape)

        converged = (total_fails <= max_fail)

        # for c in predicted_size:
        #     print("check_fit %s segment %s" % (self.selector, c))
        #     print("  modeled %s" % (modeled_size[c].sum()))
        #     print("  predicted %s" % (predicted_size[c].sum()))
        #     print("  max abs diff %s" % (abs_diff[c].max()))
        #     print("  max rel diff %s" % (rel_diff[c].max()))

        logging.info("check_fit %s iteration: %s converged: %s max_fail: %s total_fails: %s" %
                     (self.selector, iteration, converged, max_fail, total_fails))

        # - convergence stats
        if converged or iteration == self.max_iterations:
            logging.info("\nshadow_pricing max_abs_diff\n%s" % self.max_abs_diff)
            logging.info("\nshadow_pricing max_rel_diff\n%s" % self.max_rel_diff)
            logging.info("\nshadow_pricing num_fail\n%s" % self.num_fail)

        return converged

    def update_shadow_prices(self):
        """
        Adjust shadow_prices based on relative values of modeled_size and predicted_size.

        This is the heart of the shadow pricing algorithm.

        The presumption is that shadow_price_adjusted_predicted_size (along with other attractors)
        is being used in a utility expression in a location choice model. The goal is to get the
        aggregate location modeled size (choice aggregated by selector segment and zone) to match
        predicted_size. Since the location choice model may not achieve that goal initially, we
        create a 'shadow price' that tweaks the size_term to encourage the aggregate choices to
        approach the desired target predicted_sizes.

        shadow_prices is a table of coefficient (for each zone and segment) that is increases or
        decreases the size term according to whether the modelled population is less or greater
        than the predicted_size. If too few total choices are made for a particular zone and
        segment, then its shadow_price is increased, if too many, then it is decreased.

        Since the location choice is being made according to a variety of utilities in the
        expression file, whose relative weights are unknown to this algorithm, the choice of
        how to adjust the shadow_price is not completely straightforward. CTRAMP and daysim use
        different strategies (see below) and there may not be a single method that works best for
        all expression files. This would be a nice project for the mathematically inclined.

        Returns
        -------
        updates self.shadow_prices
        """

        assert self.use_shadow_pricing

        shadow_price_method = self.shadow_settings['SHADOW_PRICE_METHOD']

        # can't update_shadow_prices until after first iteration
        # modeled_size should have been set by set_choices at end of previous iteration
        assert self.modeled_size is not None
        assert self.predicted_size is not None
        assert self.shadow_prices is not None

        if shadow_price_method == 'ctramp':
            # - CTRAMP
            """
            if ( modeledDestinationLocationsByDestZone > 0 )
                shadowPrice *= ( scaledSize / modeledDestinationLocationsByDestZone );
            // else
            //    shadowPrice *= scaledSize;
            """
            damping_factor = self.shadow_settings['DAMPING_FACTOR']
            assert 0 < damping_factor <= 1

            new_scale_factor = self.predicted_size / self.modeled_size
            damped_scale_factor = 1 + (new_scale_factor - 1) * damping_factor
            new_shadow_prices = self.shadow_prices * damped_scale_factor

            # following CTRAMP (revised version - with 0 dest zone case lines commented out)
            # avoid zero-divide for 0 modeled_size, by leaving shadow_prices unchanged
            new_shadow_prices.where(self.modeled_size > 0, self.shadow_prices, inplace=True)

        elif shadow_price_method == 'daysim':
            # - Daysim
            """
            if predicted > modeled:  # if modeled is too low, increase shadow price
              target = min(
                predicted,
                modeled + modeled * percent_tolerance,
                modeled + absolute_tolerance)

            if modeled > predicted  # modeled is too high, decrease shadow price
                target = max of:
                    predicted
                    modeled - modeled * percentTolerance
                    modeled - absoluteTolerance

            shadow_price = shadow_price + log(np.maximum(target, 0.01) / np.maximum(modeled, 0.01))
            """
            # FIXME should these be the same as PERCENT_TOLERANCE and FAIL_THRESHOLD above?
            absolute_tolerance = self.shadow_settings['DAYSIM_ABSOLUTE_TOLERANCE']
            percent_tolerance = self.shadow_settings['DAYSIM_PERCENT_TOLERANCE'] / 100.0
            assert 0 <= percent_tolerance <= 1

            target = np.where(
                self.predicted_size > self.modeled_size,
                np.minimum(self.predicted_size,
                           np.minimum(self.modeled_size * (1 + percent_tolerance),
                                      self.modeled_size + absolute_tolerance)),
                np.maximum(self.predicted_size,
                           np.maximum(self.modeled_size * (1 - percent_tolerance),
                                      self.modeled_size - absolute_tolerance)))

            adjustment = np.log(np.maximum(target, 0.01) / np.maximum(self.modeled_size, 0.01))

            # def like_df(data, df):
            #     return pd.DataFrame(data=data, columns=df.columns, index=df.index)
            # print("\ntarget\n", like_df(target, self.shadow_prices).head())
            # print("\nadjustment\n", like_df(adjustment, self.shadow_prices).head())

            new_shadow_prices = self.shadow_prices + adjustment

        else:
            raise RuntimeError("unknown SHADOW_PRICE_METHOD %s" % shadow_price_method)

        # print("\nself.predicted_size\n", self.predicted_size.head())
        # print("\nself.modeled_size\n", self.modeled_size.head())
        # print("\nprevious shadow_prices\n", self.shadow_prices.head())
        # print("\nnew_shadow_prices\n", new_shadow_prices.head())

        self.shadow_prices = new_shadow_prices

    def shadow_price_adjusted_predicted_size(self):
        """
        return predicted_sizes adjusted by current shadow_price for use in utility expressions

        Returns
        -------
        pandas.DataFrame with same shape as predicted_size
        """

        return self.predicted_size * self.shadow_prices

    def write_trace_files(self, iteration):
        """
        Write trace files for this iteration
        Writes predicted_size, modeled_size, and shadow_prices tables

        Trace file names are tagged with selector and iteration number
        (e.g. self.predicted_size => shadow_price_school_predicted_size_1)

        Parameters
        ----------
        iteration: int
            current iteration to tag trace file
        """
        logger.info("write_trace_files iteration %s" % iteration)
        if iteration == 1:
            # write predicted_size only on first iteration, as it doesn't change
            tracing.write_csv(self.predicted_size,
                              'shadow_price_%s_predicted_size' % self.selector,
                              transpose=False)

        tracing.write_csv(self.modeled_size,
                          'shadow_price_%s_modeled_size_%s' % (self.selector, iteration),
                          transpose=False)
        tracing.write_csv(self.shadow_prices,
                          'shadow_price_%s_shadow_prices_%s' % (self.selector, iteration),
                          transpose=False)


def block_name(selector):
    """
    return canonical block name for selector

    Ordinarilly and ideally this wold just be selector, but since mp_tasks saves all shared data
    blocks in a common dict to pass to sub-tasks, we want to be able to handle an possible
    collision between selector names and skim names. Otherwise, just use selector name.

    Parameters
    ----------
    selector

    Returns
    -------
    block_name : str
        canonical block name
    """
    return selector


def get_shadow_pricing_info():
    """
    return dict with info about dtype and shapes of predicted and modeled size tables

    block shape is (num_zones, num_segments + 1)


    Returns
    -------
    shadow_pricing_info: dict
        'dtype': <sp_dtype>,
        'block_shapes': dict {<selector>: <block_shape>}
    """

    land_use = inject.get_table('land_use')
    size_terms = inject.get_injectable('size_terms')

    shadow_settings = config.read_model_settings('shadow_pricing.yaml')

    # shadow_pricing_models is dict of {<selector>: <model_name>}
    shadow_pricing_models = shadow_settings['shadow_pricing_models']

    blocks = OrderedDict()
    for selector in shadow_pricing_models:

        sp_rows = len(land_use)
        sp_cols = len(size_terms[size_terms.selector == selector])

        # extra tally column for TALLY_CHECKIN and TALLY_CHECKOUT semaphores
        blocks[block_name(selector)] = (sp_rows, sp_cols + 1)

    sp_dtype = np.int64

    shadow_pricing_info = {
        'dtype': sp_dtype,
        'block_shapes': blocks,
    }

    return shadow_pricing_info


def buffers_for_shadow_pricing(shadow_pricing_info):
    """
    Allocate shared_data buffers for multiprocess shadow pricing

    Allocates one buffer per selector. Buffer datatype and shape specified by shadow_pricing_info

    buffers are multiprocessing.Array (RawArray protected by a multiprocessing.Lock wrapper)
    We don't actually use the wrapped version as it slows access down and doesn't provide
    protection for numpy-wrapped arrays, but it does provide a convenient way to bundle
    RawArray and an associated lock. (ShadowPriceCalculator uses the lock to coordinate access to
    the numpy-wrapped RawArray.)

    Parameters
    ----------
    shadow_pricing_info : dict

    Returns
    -------
        data_buffers : dict {<selector> : <shared_data_buffer>}
        dict of multiprocessing.Array keyed by selector
    """

    dtype = shadow_pricing_info['dtype']
    block_shapes = shadow_pricing_info['block_shapes']

    data_buffers = {}
    for block_key, block_shape in iteritems(block_shapes):

        # buffer_size must be int (or p2.7 long), not np.int64
        buffer_size = int(np.prod(block_shape))

        csz = buffer_size * np.dtype(dtype).itemsize
        logger.info("allocating shared buffer %s %s buffer_size %s (%s)" %
                    (block_key, buffer_size, block_shape, util.GB(csz)))

        if np.issubdtype(dtype, np.int64):
            typecode = ctypes.c_long
        else:
            raise RuntimeError("buffer_for_shadow_pricing unrecognized dtype %s" % dtype)

        shared_data_buffer = multiprocessing.Array(typecode, buffer_size)

        logger.info("buffer_for_shadow_pricing added block %s" % (block_key))

        data_buffers[block_key] = shared_data_buffer

    return data_buffers


def shadow_price_data_from_buffers(data_buffers, shadow_pricing_info, selector):
    """

    Parameters
    ----------
    data_buffers : dict of {<selector_name> : <multiprocessing.Array>}
        multiprocessing.Array is simply a convenient way to bundle Array and Lock
        we extract the lock and wrap the RawArray in a numpy array for convenience in indexing

        The shared data buffer has shape (<num_zones, <num_segments> + 1)
        extra column is for reverse semaphores with TALLY_CHECKIN and TALLY_CHECKOUT
    shadow_pricing_info : dict
        dict of useful info
          'dtype': sp_dtype,
          'block_shapes' : OrderedDict({<selector>: <shape tuple>})
            dict mapping selector to block shape (including extra column for semaphores)
            e.g. {'school': (num_zones, num_segments + 1)
    selector : str
        location type selector (e.g. school or workplace)

    Returns
    -------
    shared_data, shared_data_lock
        shared_data : multiprocessing.Array or None (if single process)
        shared_data_lock : numpy array wrapping multiprocessing.RawArray or None (if single process)
    """

    assert type(data_buffers) == dict

    dtype = shadow_pricing_info['dtype']
    block_shapes = shadow_pricing_info['block_shapes']

    if selector not in block_shapes:
        raise RuntimeError("Selector %s not in shadow_pricing_info" % selector)

    if block_name(selector) not in data_buffers:
        raise RuntimeError("Block %s not in data_buffers" % block_name(selector))

    shape = block_shapes[selector]
    data = data_buffers[block_name(selector)]

    return np.frombuffer(data.get_obj(), dtype=dtype).reshape(shape), data.get_lock()


def load_shadow_price_calculator(model_settings):
    """
    Initialize ShadowPriceCalculator for model selector (e.g. school or workplace)

    If multiprocessing, get the shared_data buffer to coordinate global_predicted_size
    calculation across sub-processes

    Parameters
    ----------
    model_settings : dict

    Returns
    -------
    spc : ShadowPriceCalculator
    """

    selector = model_settings['SELECTOR']

    # - get shared_data from data_buffers (if multiprocessing)
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


def add_predicted_size_tables():
    """
    inject tour_destination_size_terms tables for each selector (e.g. school, workplace)

    Size tables are pandas dataframes with locations counts for selector by zone and segment
    tour_destination_size_terms

    if using shadow pricing, we scale size_table counts to sample population
    (in which case, they have to be created while single-process)

    Scaling is problematic as it breaks household result replicability across sample sizes
    It also changes the magnitude of the size terms so if they are used as utilities in
    expression files, their importance will diminish relative to other utilities as the sample
    size decreases.

    Scaling makes most sense for a full sample in conjunction with shadow pricing, where
    shadow prices can be adjusted iteratively to bring modelled counts into line with predicted
    (size table) counts.
    """

    use_shadow_pricing = bool(config.setting('use_shadow_pricing'))

    shadow_settings = config.read_model_settings('shadow_pricing.yaml')
    shadow_pricing_models = shadow_settings['shadow_pricing_models']

    if shadow_pricing_models is None:
        logger.warning('shadow_pricing_models list not found in shadow_pricing settings')
        return

    # shadow_pricing_models is dict of {<selector>: <model_name>}
    # since these are scaled to model size, they have to be created while single-process

    for selector, model_name in iteritems(shadow_pricing_models):

        model_settings = config.read_model_settings(model_name)

        assert selector == model_settings['SELECTOR']

        segment_ids = model_settings['SEGMENT_IDS']
        chooser_table_name = model_settings['CHOOSER_TABLE_NAME']
        chooser_segment_column = model_settings['CHOOSER_SEGMENT_COLUMN_NAME']

        choosers_df = inject.get_table(chooser_table_name).to_frame()
        if 'CHOOSER_FILTER_COLUMN_NAME' in model_settings:
            choosers_df = \
                choosers_df[choosers_df[model_settings['CHOOSER_FILTER_COLUMN_NAME']] != 0]

        # - raw_predicted_size
        land_use = inject.get_table('land_use')
        size_terms = inject.get_injectable('size_terms')
        raw_size = tour_destination_size_terms(land_use, size_terms, selector).astype(np.float64)
        assert set(raw_size.columns) == set(segment_ids.keys())

        if use_shadow_pricing:

            # - scale size_table counts to sample population
            # scaled_size = zone_size * (total_segment_modeled / total_segment_predicted)

            # segment scale factor (modeled / predicted) keyed by segment_name
            segment_scale_factors = {}
            for c in raw_size:
                # number of zone demographics predicted destination choices
                segment_predicted_size = raw_size[c].astype(np.float64).sum()

                # number of synthetic population choosers in segment
                segment_chooser_count = \
                    (choosers_df[chooser_segment_column] == segment_ids[c]).sum()

                segment_scale_factors[c] = \
                    segment_chooser_count / np.maximum(segment_predicted_size, 1)

                logger.info("add_predicted_size_tables %s segment %s "
                            "predicted %s modeled %s scale_factor %s" %
                            (chooser_table_name, c,
                             segment_predicted_size,
                             segment_chooser_count,
                             segment_scale_factors[c]))

            # FIXME - should we be rounding?
            scaled_size = (raw_size * segment_scale_factors).round()
        else:
            # don't scale if not shadow_pricing (breaks partial sample replicability)
            scaled_size = raw_size

        inject.add_table(size_table_name(selector), scaled_size)
