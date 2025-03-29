# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import ctypes
import logging
import multiprocessing
import time
from collections import OrderedDict
from typing import Any, Literal

import numpy as np
import pandas as pd

from activitysim.abm.tables.size_terms import size_terms as get_size_terms
from activitysim.abm.tables.size_terms import tour_destination_size_terms
from activitysim.core import logit, tracing, util, workflow
from activitysim.core.configuration import PydanticReadable
from activitysim.core.configuration.logit import TourLocationComponentSettings
from activitysim.core.input import read_input_table

logger = logging.getLogger(__name__)

"""
ShadowPriceCalculator and associated utility methods

See docstrings for documentation on:

update_shadow_prices      how shadow_price coefficients are calculated
synchronize_modeled_size  interprocess communication to compute aggregate modeled_size
check_fit                 convergence criteria for shadow_pric iteration

Import concepts and variables:

model_selector: str
    Identifies a specific location choice model (e.g. 'school', 'workplace')
    The various models work similarly, but use different expression files, model settings, etc.

segment: str
    Identifies a specific demographic segment of a model (e.g. 'elementary' segment of 'school')
    Models can have different size term coefficients (in destinatin_choice_size_terms file) and
    different utility coefficients in models's location and location_sample csv expression files

size_table: pandas.DataFrame


"""


"""
Artisanal reverse semaphores to synchronize concurrent access to shared data buffer

we use the first two rows of the final column in numpy-wrapped shared data as 'reverse semaphores'
(they synchronize concurrent access to shared data resource rather than throttling access)

ShadowPriceCalculator.synchronize_modeled_size coordinates access to the global aggregate zone counts
(local_modeled_size summed across all sub-processes) using these two semaphores
(which are really only tuples of indexes of locations in the shared data array.
"""
TALLY_CHECKIN = (0, -1)
TALLY_CHECKOUT = (1, -1)
TALLY_PENDING_PERSONS = (2, -1)

default_segment_to_name_dict = {
    # model_selector : persons_segment_name
    "school": "school_segment",
    "workplace": "income_segment",
}

default_segment_to_name_dict = {
    # model_selector : persons_segment_name
    "school": "school_segment",
    "workplace": "income_segment",
}


def size_table_name(model_selector):
    """
    Returns canonical name of injected destination desired_size table

    Parameters
    ----------
    model_selector : str
        e.g. school or workplace

    Returns
    -------
    table_name : str
    """
    return "%s_destination_size" % model_selector


class ShadowPriceSettings(PydanticReadable, extra="forbid"):
    """Settings used for shadow pricing."""

    shadow_pricing_models: dict[str, str] | None = None
    """List model_selectors and model_names of models that use shadow pricing.
  This list identifies which size_terms to preload which must be done in single process mode, so
  predicted_size tables can be scaled to population"""

    LOAD_SAVED_SHADOW_PRICES: bool = True
    """Global switch to enable/disable loading of saved shadow prices.

    This is ignored if global use_shadow_pricing switch is False
    """

    MAX_ITERATIONS: int = 5
    """Number of shadow price iterations for cold start."""

    MAX_ITERATIONS_SAVED: int = 1
    """Number of shadow price iterations for warm start.

    A warm start means saved shadow_prices were found in a file and loaded."""

    SIZE_THRESHOLD: float = 10
    """ignore criteria for zones smaller than size_threshold"""

    PERCENT_TOLERANCE: float = 5
    """zone passes if modeled is within percent_tolerance of  predicted_size"""

    FAIL_THRESHOLD: float = 10
    """max percentage of zones allowed to fail"""

    SHADOW_PRICE_METHOD: Literal["ctramp", "daysim", "simulation"] = "ctramp"

    DAMPING_FACTOR: float = 1
    """ctramp-style damping factor"""

    SCALE_SIZE_TABLE: bool = False

    DAYSIM_ABSOLUTE_TOLERANCE: float = 50
    DAYSIM_PERCENT_TOLERANCE: float = 10

    TARGET_THRESHOLD: float = 20
    """ignore criteria for zones smaller than target_threshold (total employmnet or enrollment)"""

    workplace_segmentation_targets: dict[str, str] | None = None
    school_segmentation_targets: dict[str, str] | None = None

    WRITE_ITERATION_CHOICES: bool = False

    SEGMENT_TO_NAME: dict[str, str] = {
        "school": "school_segment",
        "workplace": "income_segment",
    }  # pydantic uses deep copy, so mutable default value is ok here
    """Mapping from model_selector to persons_segment_name."""


class ShadowPriceCalculator:
    def __init__(
        self,
        state: workflow.State,
        model_settings: TourLocationComponentSettings,
        num_processes,
        shared_data=None,
        shared_data_lock=None,
        shared_data_choice=None,
        shared_data_choice_lock=None,
        shared_sp_choice_df=None,
    ):
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

        self.num_processes = num_processes
        self.use_shadow_pricing = bool(state.settings.use_shadow_pricing)
        self.saved_shadow_price_file_path = (
            None  # set by read_saved_shadow_prices if loaded
        )

        self.model_selector = model_settings.MODEL_SELECTOR

        if (self.num_processes > 1) and not state.settings.fail_fast:
            # if we are multiprocessing, then fail_fast should be true or we will wait forever for failed processes
            logger.warning(
                "deprecated combination of multiprocessing and not fail_fast"
            )
            raise RuntimeError(
                "Shadow pricing requires fail_fast setting in multiprocessing mode"
            )

        self.segment_ids = model_settings.SEGMENT_IDS

        # - modeled_size (set by call to set_choices/synchronize_modeled_size)
        self.modeled_size = None

        if self.use_shadow_pricing:
            self.shadow_settings = ShadowPriceSettings.read_settings_file(
                state.filesystem, "shadow_pricing.yaml"
            )

            for k, k_value in self.shadow_settings:
                logger.debug(f"shadow_settings {k}: {k_value}")

        full_model_run = state.settings.households_sample_size == 0
        if (
            self.use_shadow_pricing
            and not full_model_run
            and self.shadow_settings.SHADOW_PRICE_METHOD != "simulation"
        ):
            # ctramp and daysim methods directly compare desired and modeled size to compute shadow prices.
            # desination size terms are scaled in add_size_tables only for full model runs
            logger.warning(
                "only 'simulation' shadow price method can use_shadow_pricing and not full_model_run"
            )
            logger.warning(f"Not using shadow pricing for {self.model_selector}")
            self.use_shadow_pricing = False

        if (
            self.use_shadow_pricing
            and self.model_selector not in ["workplace", "school"]
            and self.shadow_settings.SHADOW_PRICE_METHOD == "simulation"
        ):
            logger.warning(
                "Shadow price simulation method is only implemented for workplace and school."
            )
            logger.warning(f"Not using shadow pricing for {self.model_selector}")
            self.use_shadow_pricing = False

        # - destination_size_table (desired_size)
        self.desired_size = state.get_dataframe(size_table_name(self.model_selector))
        self.desired_size = self.desired_size.sort_index()

        assert (
            self.desired_size.index.is_monotonic_increasing
        ), f"{size_table_name(self.model_selector)} not is_monotonic_increasing"

        # - shared_data
        if shared_data is not None:
            assert shared_data.shape[0] == self.desired_size.shape[0]
            assert (
                shared_data.shape[1] == self.desired_size.shape[1] + 1
            )  # tally column
            assert shared_data_lock is not None
        self.shared_data = shared_data
        self.shared_data_lock = shared_data_lock

        self.shared_data_choice = shared_data_choice
        self.shared_data_choice_lock = shared_data_choice_lock

        self.shared_sp_choice_df = shared_sp_choice_df
        if shared_sp_choice_df is not None:
            self.shared_sp_choice_df = self.shared_sp_choice_df.astype("int")
            self.shared_sp_choice_df = self.shared_sp_choice_df.set_index("person_id")
            self.shared_sp_choice_df["choice"] = int(0)

        # - load saved shadow_prices (if available) and set max_iterations accordingly
        if self.use_shadow_pricing:
            self.shadow_prices = None
            self.shadow_price_method = self.shadow_settings.SHADOW_PRICE_METHOD
            assert self.shadow_price_method in ["daysim", "ctramp", "simulation"]
            # ignore convergence criteria for zones smaller than target_threshold
            self.target_threshold = self.shadow_settings.TARGET_THRESHOLD

            if self.shadow_settings.LOAD_SAVED_SHADOW_PRICES:
                # read_saved_shadow_prices logs error and returns None if file not found
                self.shadow_prices = self.read_saved_shadow_prices(
                    state, model_settings
                )

            if self.shadow_prices is None:
                self.max_iterations = self.shadow_settings.MAX_ITERATIONS
            else:
                self.max_iterations = self.shadow_settings.MAX_ITERATIONS_SAVED

            # initial_shadow_price if we did not load
            if self.shadow_prices is None:
                # initial value depends on method
                initial_shadow_price = (
                    1.0 if self.shadow_price_method == "ctramp" else 0.0
                )
                self.shadow_prices = pd.DataFrame(
                    data=initial_shadow_price,
                    columns=self.desired_size.columns,
                    index=self.desired_size.index,
                )
        else:
            self.max_iterations = 1

        self.num_fail = pd.DataFrame(index=self.desired_size.columns)
        self.max_abs_diff = pd.DataFrame(index=self.desired_size.columns)
        self.max_rel_diff = pd.DataFrame(index=self.desired_size.columns)
        self.choices_by_iteration = pd.DataFrame()
        self.global_pending_persons = 1
        self.sampled_persons = pd.DataFrame()

        if (
            self.use_shadow_pricing
            and self.shadow_settings.SHADOW_PRICE_METHOD == "simulation"
        ):
            assert self.model_selector in ["workplace", "school"]
            self.target = {}
            land_use = state.get_dataframe("land_use")

            if self.model_selector == "workplace":
                employment_targets = (
                    self.shadow_settings.workplace_segmentation_targets or {}
                )
                assert (
                    employment_targets
                ), "Need to supply workplace_segmentation_targets in shadow_pricing.yaml"

                for segment, target in employment_targets.items():
                    assert (
                        segment in self.shadow_prices.columns
                    ), f"{segment} is not in {self.shadow_prices.columns}"
                    assert (
                        target in land_use.columns
                    ), f"{target} is not in {land_use.columns}"
                    self.target[segment] = land_use[target]

            elif self.model_selector == "school":
                school_targets = self.shadow_settings.school_segmentation_targets or {}
                assert (
                    school_targets
                ), "Need to supply school_segmentation_targets in shadow_pricing.yaml"

                for segment, target in school_targets.items():
                    assert (
                        segment in self.shadow_prices.columns
                    ), f"{segment} is not in {self.shadow_prices.columns}"
                    assert (
                        target in land_use.columns
                    ), f"{target} is not in landuse columns: {land_use.columns}"
                    self.target[segment] = land_use[target]

    def read_saved_shadow_prices(
        self, state: workflow.State, model_settings: TourLocationComponentSettings
    ):
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
        saved_shadow_price_file_name = model_settings.SAVED_SHADOW_PRICE_TABLE_NAME
        if saved_shadow_price_file_name:
            # FIXME - where should we look for this file?
            file_path = state.filesystem.get_data_file_path(
                saved_shadow_price_file_name, mandatory=False
            )
            if file_path:
                shadow_prices = pd.read_csv(file_path, index_col=0)
                self.saved_shadow_price_file_path = file_path  # informational
                logger.info("loaded saved_shadow_prices from %s" % file_path)
            else:
                logger.warning("Could not find saved_shadow_prices file %s" % file_path)

        return shadow_prices

    def synchronize_modeled_size(self, local_modeled_size):
        """
        We have to wait until all processes have computed choices and aggregated them by segment
        and zone before we can compute global aggregate zone counts (by segment). Since the global
        zone counts are in shared data, we have to coordinate access to the data structure across
        sub-processes.
        Note that all access to self.shared_data has to be protected by acquiring shared_data_lock
        ShadowPriceCalculator.synchronize_modeled_size coordinates access to the global aggregate
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
        assert self.num_processes > 1

        def get_tally(t):
            with self.shared_data_lock:
                return self.shared_data[t]

        def wait(tally, target):
            while get_tally(tally) != target:
                time.sleep(1)

        # - nobody checks in until checkout clears
        wait(TALLY_CHECKOUT, 0)

        # - add local_modeled_size data, increment TALLY_CHECKIN
        with self.shared_data_lock:
            first_in = self.shared_data[TALLY_CHECKIN] == 0
            # add local data from df to shared data buffer
            # final column is used for tallys, hence the negative index
            # Ellipsis expands : to fill available dims so [..., 0:-1] is the whole array except for the tallys
            self.shared_data[..., 0:-1] += local_modeled_size.values
            self.shared_data[TALLY_CHECKIN] += 1
            if len(self.sampled_persons) > 0:
                self.shared_data[TALLY_PENDING_PERSONS] += 1

        # - wait until everybody else has checked in
        wait(TALLY_CHECKIN, self.num_processes)

        # - copy shared data, increment TALLY_CHECKOUT
        with self.shared_data_lock:
            logger.info("copy shared_data")
            # numpy array with sum of local_modeled_size.values from all processes
            global_modeled_size_array = self.shared_data[..., 0:-1].copy()
            self.global_pending_persons = self.shared_data[TALLY_PENDING_PERSONS]
            self.shared_data[TALLY_CHECKOUT] += 1

        # - first in waits until all other processes have checked out, and cleans tub
        if first_in:
            wait(TALLY_CHECKOUT, self.num_processes)
            with self.shared_data_lock:
                # zero shared_data, clear TALLY_CHECKIN, and TALLY_CHECKOUT semaphores
                self.shared_data[:] = 0
            logger.info("first_in clearing shared_data")

        # convert summed numpy array data to conform to original dataframe
        global_modeled_size_df = pd.DataFrame(
            data=global_modeled_size_array,
            index=local_modeled_size.index,
            columns=local_modeled_size.columns,
        )

        return global_modeled_size_df

    def synchronize_choices(self, local_modeled_size):
        """
        Same thing as the above synchronize_modeled_size method with the small
        difference of keeping track of the individual choices instead of the
        aggregate modeled choices between processes.

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
        assert self.shared_data_choice is not None
        assert self.num_processes > 1

        def get_tally(t):
            with self.shared_data_choice_lock:
                return self.shared_data_choice[t]

        def wait(tally, target):
            while get_tally(tally) != target:
                time.sleep(1)

        # - nobody checks in until checkout clears
        wait(TALLY_CHECKOUT, 0)

        # - add local_modeled_size data, increment TALLY_CHECKIN
        with self.shared_data_choice_lock:
            first_in = self.shared_data_choice[TALLY_CHECKIN] == 0
            # add local data from df to shared data buffer
            # final column is used for tallys, hence the negative index
            # Ellipsis expands : to fill available dims so [..., 0:-1] is the whole array except for the tallys
            self.shared_data_choice[..., 0:-1] += local_modeled_size.values.astype(
                np.int64
            )
            self.shared_data_choice[TALLY_CHECKIN] += 1

        # - wait until everybody else has checked in
        wait(TALLY_CHECKIN, self.num_processes)

        # - copy shared data, increment TALLY_CHECKIN
        with self.shared_data_choice_lock:
            logger.info("copy shared_data")
            # numpy array with sum of local_modeled_size.values from all processes
            global_modeled_size_array = self.shared_data_choice[..., 0:-1].copy()
            self.shared_data_choice[TALLY_CHECKOUT] += 1

        # - first in waits until all other processes have checked out, and cleans tub
        if first_in:
            wait(TALLY_CHECKOUT, self.num_processes)
            with self.shared_data_choice_lock:
                # zero shared_data, clear TALLY_CHECKIN, and TALLY_CHECKOUT semaphores
                self.shared_data_choice[:] = 0
            logger.info("first_in clearing shared_data")

        # convert summed numpy array data to conform to original dataframe
        global_modeled_size_df = pd.DataFrame(
            data=global_modeled_size_array,
            index=local_modeled_size.index,
            columns=local_modeled_size.columns,
        )

        return global_modeled_size_df

    def set_choices(self, choices, segment_ids):
        """
        aggregate individual location choices to modeled_size by zone and segment

        Parameters
        ----------
        choices : pandas.Series
            zone id of location choice indexed by person_id
        segment_ids : pandas.Series
            segment id tag for this individual indexed by person_id

        Returns
        -------
        updates self.modeled_size
        """

        modeled_size = pd.DataFrame(index=self.desired_size.index)
        for seg_name in self.desired_size:
            segment_choices = choices[(segment_ids == self.segment_ids[seg_name])]

            modeled_size[seg_name] = segment_choices.value_counts()

        modeled_size = modeled_size.fillna(0).astype(int)

        if self.num_processes == 1:
            # - not multiprocessing
            self.choices_synced = choices
            self.modeled_size = modeled_size
        else:
            # - if we are multiprocessing, we have to aggregate across sub-processes
            self.modeled_size = self.synchronize_modeled_size(modeled_size)

            # need to also store individual choices if simulation approach
            choice_merged = pd.merge(
                self.shared_sp_choice_df,
                choices,
                left_index=True,
                right_index=True,
                how="left",
                suffixes=("_x", "_y"),
            )

            choice_merged["choice_y"] = choice_merged["choice_y"].fillna(0)
            choice_merged["choice"] = (
                choice_merged["choice_x"] + choice_merged["choice_y"]
            )
            choice_merged = choice_merged.drop(columns=["choice_x", "choice_y"])

            self.choices_synced = self.synchronize_choices(choice_merged)

    def check_fit(self, state: workflow.State, iteration):
        """
        Check convergence criteria fit of modeled_size to target desired_size
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
        assert self.desired_size is not None

        # - convergence criteria for check_fit
        # ignore convergence criteria for zones smaller than size_threshold
        size_threshold = self.shadow_settings.SIZE_THRESHOLD
        # zone passes if modeled is within percent_tolerance of  desired_size
        percent_tolerance = self.shadow_settings.PERCENT_TOLERANCE
        # max percentage of zones allowed to fail
        fail_threshold = self.shadow_settings.FAIL_THRESHOLD
        # option to write out choices by iteration for each person to trace folder
        write_choices = self.shadow_settings.WRITE_ITERATION_CHOICES
        if write_choices:
            self.choices_by_iteration[iteration] = self.choices_synced

        if self.shadow_settings.SHADOW_PRICE_METHOD != "simulation":
            modeled_size = self.modeled_size
            desired_size = self.desired_size

            abs_diff = (desired_size - modeled_size).abs()

            self.rel_diff = abs_diff / modeled_size

            # ignore zones where desired_size < threshold
            self.rel_diff.where(desired_size >= size_threshold, 0, inplace=True)

            # ignore zones where rel_diff < percent_tolerance
            self.rel_diff.where(
                self.rel_diff > (percent_tolerance / 100.0), 0, inplace=True
            )

            self.num_fail["iter%s" % iteration] = (self.rel_diff > 0).sum()
            self.max_abs_diff["iter%s" % iteration] = abs_diff.max()
            self.max_rel_diff["iter%s" % iteration] = self.rel_diff.max()

            total_fails = (self.rel_diff > 0).values.sum()

            # FIXME - should not count zones where desired_size < threshold? (could calc in init)
            max_fail = (fail_threshold / 100.0) * util.iprod(desired_size.shape)

            converged = total_fails <= max_fail

        else:
            rel_diff_df = pd.DataFrame(index=self.shadow_prices.index)
            abs_diff_df = pd.DataFrame(index=self.shadow_prices.index)
            # checking each segment
            for segment in self.segment_ids:
                desired_size = self.target[segment]
                modeled_size = self.modeled_size[segment]

                # loop over other segments and add to modeled share if they have the same target
                for other_segment in self.segment_ids:
                    if (segment != other_segment) & (
                        self.target[segment].equals(self.target[other_segment])
                    ):
                        modeled_size = modeled_size + self.modeled_size[other_segment]

                # want to match distribution, not absolute numbers so share is computed
                desired_share = desired_size / desired_size.sum()
                modeled_share = modeled_size / modeled_size.sum()

                abs_diff_df[segment] = (desired_size - modeled_size).abs()

                rel_diff = desired_share / modeled_share
                rel_diff = np.where(
                    # is the desired size below the threshold?
                    (desired_size <= self.target_threshold)
                    # is the difference within the tolerance?
                    | (np.abs(1 - rel_diff) < (percent_tolerance / 100.0)),
                    0,
                    rel_diff,
                )
                rel_diff_df[segment] = rel_diff

            # relative difference is set to max across segments
            self.rel_diff = rel_diff_df.max(axis=1)
            abs_diff = abs_diff_df.max(axis=1)

            self.num_fail["iter%s" % iteration] = (self.rel_diff > 0).sum()
            self.max_abs_diff["iter%s" % iteration] = abs_diff.max()
            self.max_rel_diff["iter%s" % iteration] = rel_diff.max()

            total_fails = (self.rel_diff > 0).values.sum()

            # FIXME - should not count zones where desired_size < threshold? (could calc in init)
            max_fail = (fail_threshold / 100.0) * util.iprod(desired_size.shape)

            converged = (total_fails <= np.ceil(max_fail)) | (
                (iteration > 1) & (self.global_pending_persons == 0)
            )

        logger.info(
            "check_fit %s iteration: %s converged: %s max_fail: %s total_fails: %s"
            % (self.model_selector, iteration, converged, max_fail, total_fails)
        )

        # - convergence stats
        if converged or iteration == self.max_iterations:
            logger.info("\nshadow_pricing max_abs_diff\n%s" % self.max_abs_diff)
            logger.info("\nshadow_pricing max_rel_diff\n%s" % self.max_rel_diff)
            logger.info("\nshadow_pricing num_fail\n%s" % self.num_fail)

            if write_choices:
                state.tracing.write_csv(
                    self.choices_by_iteration,
                    "%s_choices_by_shadow_price_iteration" % self.model_selector,
                    transpose=False,
                )

        return converged

    def update_shadow_prices(self, state):
        """
        Adjust shadow_prices based on relative values of modeled_size and desired_size.

        This is the heart of the shadow pricing algorithm.

        The presumption is that shadow_price_adjusted_desired_size (along with other attractors)
        is being used in a utility expression in a location choice model. The goal is to get the
        aggregate location modeled size (choice aggregated by model_selector segment and zone) to
        match desired_size. Since the location choice model may not achieve that goal initially,
        we create a 'shadow price' that tweaks the size_term to encourage the aggregate choices to
        approach the desired target desired_sizes.

        shadow_prices is a table of coefficient (for each zone and segment) that is increases or
        decreases the size term according to whether the modelled population is less or greater
        than the desired_size. If too few total choices are made for a particular zone and
        segment, then its shadow_price is increased, if too many, then it is decreased.

        Since the location choice is being made according to a variety of utilities in the
        expression file, whose relative weights are unknown to this algorithm, the choice of
        how to adjust the shadow_price is not completely straightforward. CTRAMP and Daysim use
        different strategies (see below) and there may not be a single method that works best for
        all expression files. This would be a nice project for the mathematically inclined.

        Returns
        -------
        updates self.shadow_prices
        """

        assert self.use_shadow_pricing

        shadow_price_method = self.shadow_settings.SHADOW_PRICE_METHOD

        # can't update_shadow_prices until after first iteration
        # modeled_size should have been set by set_choices at end of previous iteration
        assert self.modeled_size is not None
        assert self.desired_size is not None
        assert self.shadow_prices is not None

        if shadow_price_method == "ctramp":
            # - CTRAMP
            """
            if ( modeledDestinationLocationsByDestZone > 0 )
                shadowPrice *= ( scaledSize / modeledDestinationLocationsByDestZone );
            // else
            //    shadowPrice *= scaledSize;
            """
            damping_factor = self.shadow_settings.DAMPING_FACTOR
            assert 0 < damping_factor <= 1

            new_scale_factor = self.desired_size / self.modeled_size
            damped_scale_factor = 1 + (new_scale_factor - 1) * damping_factor
            new_shadow_prices = self.shadow_prices * damped_scale_factor

            # following CTRAMP (revised version - with 0 dest zone case lines commented out)
            # avoid zero-divide for 0 modeled_size, by leaving shadow_prices unchanged
            new_shadow_prices.where(
                self.modeled_size > 0, self.shadow_prices, inplace=True
            )
            self.shadow_prices = new_shadow_prices

        elif shadow_price_method == "daysim":
            # - Daysim
            """
            if modeled > desired:  # if modeled is too high, increase shadow price
              target = min(
                modeled,
                desired * (1 + percent_tolerance),
                desired + absolute_tolerance)

            if modeled < desired  # modeled is too low, decrease shadow price
              target = max(
                modeled,
                desired * (1 - percentTolerance),
                desired - absoluteTolerance)

            shadow_price = shadow_price + log(np.maximum(target, 0.01) / np.maximum(modeled, 0.01))
            """
            # FIXME should these be the same as PERCENT_TOLERANCE and FAIL_THRESHOLD above?
            absolute_tolerance = self.shadow_settings.DAYSIM_ABSOLUTE_TOLERANCE
            percent_tolerance = self.shadow_settings.DAYSIM_PERCENT_TOLERANCE / 100.0
            assert 0 <= percent_tolerance <= 1

            target = np.where(
                self.modeled_size > self.desired_size,
                np.minimum(
                    self.modeled_size,
                    np.minimum(
                        self.desired_size * (1 + percent_tolerance),
                        self.desired_size + absolute_tolerance,
                    ),
                ),
                np.maximum(
                    self.modeled_size,
                    np.maximum(
                        self.desired_size * (1 - percent_tolerance),
                        self.desired_size - absolute_tolerance,
                    ),
                ),
            )

            # adjustment = np.log(np.maximum(target, 0.01) / np.maximum(self.modeled_size, 0.01))
            adjustment = np.log(
                np.maximum(target, 0.01) / np.maximum(self.modeled_size, 1)
            )

            new_shadow_prices = self.shadow_prices + adjustment
            self.shadow_prices = new_shadow_prices

        elif shadow_price_method == "simulation":
            # - NewMethod
            """
            C_j = (emp_j/sum(emp_j))/(workers_j/sum(workers_j))

            if C_j > 1: #under-estimate workers in zone

                shadow_price_j = 0

            elif C_j < 1: #over-estimate workers in zone

                shadow_price_j = -999
                resimulate n workers from zone j, with n = int(workers_j-emp_j/sum(emp_j*workers_j))
            """
            percent_tolerance = self.shadow_settings.PERCENT_TOLERANCE
            sampled_persons = pd.DataFrame()
            persons_merged = state.get_dataframe("persons_merged")

            # need to join the segment to the choices to sample correct persons
            segment_to_name_dict = self.shadow_settings.SEGMENT_TO_NAME
            segment_name = segment_to_name_dict[self.model_selector]

            if type(self.choices_synced) != pd.DataFrame:
                self.choices_synced = self.choices_synced.to_frame()

            choices_synced = self.choices_synced.merge(
                persons_merged[segment_name],
                how="left",
                left_index=True,
                right_index=True,
            ).rename(columns={segment_name: "segment"})

            for segment in self.segment_ids:
                desired_size = self.target[segment]
                modeled_size = self.modeled_size[segment]

                # loop over other segments and add to modeled share if they have the same target
                for other_segment in self.segment_ids:
                    if (segment != other_segment) & (
                        self.target[segment].equals(self.target[other_segment])
                    ):
                        modeled_size = modeled_size + self.modeled_size[other_segment]

                # want to match distribution, not absolute numbers so share is computed
                desired_share = desired_size / desired_size.sum()
                modeled_share = modeled_size / modeled_size.sum()

                sprice = desired_share / modeled_share
                sprice.fillna(0, inplace=True)
                sprice.replace([np.inf, -np.inf], 0, inplace=True)

                # shadow prices are set to -999 if overassigned or 0 if the zone still has room for this segment
                self.shadow_prices[segment] = np.where(
                    (sprice <= 1 + percent_tolerance / 100), -999, 0
                )

                zonal_sample_rate = 1 - sprice
                overpredicted_zones = self.shadow_prices[
                    self.shadow_prices[segment] == -999
                ].index
                zones_outside_tol = zonal_sample_rate[
                    zonal_sample_rate > percent_tolerance / 100
                ].index
                small_zones = desired_size[desired_size <= self.target_threshold].index

                choices = choices_synced[
                    (choices_synced["choice"].isin(overpredicted_zones))
                    & (choices_synced["choice"].isin(zones_outside_tol))
                    & ~(choices_synced["choice"].isin(small_zones))
                    # sampling only from people in this segment
                    & (choices_synced["segment"] == self.segment_ids[segment])
                ]["choice"]

                # segment is converged if all zones are overpredicted / within tolerance
                # do not want people to be re-simulated if no open zone exists
                converged = len(overpredicted_zones) == len(self.shadow_prices)

                # draw persons assigned to overassigned zones to re-simulate if not converged
                if (len(choices) > 0) & (~converged):
                    # person's probability of being selected for re-simulation is from the zonal sample rate
                    sample_rates = choices.map(zonal_sample_rate.to_dict())
                    probs = pd.DataFrame(
                        data={"0": 1 - sample_rates, "1": sample_rates},
                        index=choices.index,
                    )
                    # using ActivitySim's RNG to make choices for repeatability
                    current_sample, rands = logit.make_choices(state, probs)
                    current_sample = current_sample[current_sample == 1]

                    if len(sampled_persons) == 0:
                        sampled_persons = current_sample
                    else:
                        sampled_persons = pd.concat([sampled_persons, current_sample])

            self.sampled_persons = sampled_persons

        else:
            raise RuntimeError("unknown SHADOW_PRICE_METHOD %s" % shadow_price_method)

    def dest_size_terms(self, segment):
        assert segment in self.segment_ids

        size_term_adjustment = 1
        utility_adjustment = 0

        if self.use_shadow_pricing:
            shadow_price_method = self.shadow_settings.SHADOW_PRICE_METHOD

            if shadow_price_method == "ctramp":
                size_term_adjustment = self.shadow_prices[segment]
            elif shadow_price_method == "daysim":
                utility_adjustment = self.shadow_prices[segment]
            elif shadow_price_method == "simulation":
                utility_adjustment = self.shadow_prices[segment]
            else:
                raise RuntimeError(
                    "unknown SHADOW_PRICE_METHOD %s" % shadow_price_method
                )

        size_terms = pd.DataFrame(
            {
                "size_term": self.desired_size[segment],
                "shadow_price_size_term_adjustment": size_term_adjustment,
                "shadow_price_utility_adjustment": utility_adjustment,
            },
            index=self.desired_size.index,
        )

        assert size_terms.index.is_monotonic_increasing

        return size_terms

    def write_trace_files(self, state: workflow.State, iteration):
        """
        Write trace files for this iteration
        Writes desired_size, modeled_size, and shadow_prices tables

        Trace file names are tagged with model_selector and iteration number
        (e.g. self.desired_size => shadow_price_school_desired_size_1)

        Parameters
        ----------
        iteration: int
            current iteration to tag trace file
        """
        logger.info("write_trace_files iteration %s" % iteration)
        if iteration == 1:
            # write desired_size only on first iteration, as it doesn't change
            state.tracing.write_csv(
                self.desired_size,
                "shadow_price_%s_desired_size" % self.model_selector,
                transpose=False,
            )

        state.tracing.write_csv(
            self.modeled_size,
            "shadow_price_%s_modeled_size_%s" % (self.model_selector, iteration),
            transpose=False,
        )

        if self.use_shadow_pricing:
            state.tracing.write_csv(
                self.shadow_prices,
                "shadow_price_%s_shadow_prices_%s" % (self.model_selector, iteration),
                transpose=False,
            )


def block_name(model_selector):
    """
    return canonical block name for model_selector

    Ordinarily and ideally this would just be model_selector, but since mp_tasks saves all
    shared data blocks in a common dict to pass to sub-tasks, we want to be able override
    block naming convention to handle any collisions between model_selector names and skim names.
    Until and unless that happens, we just use model_selector name.

    Parameters
    ----------
    model_selector

    Returns
    -------
    block_name : str
        canonical block name
    """
    return model_selector


def buffers_for_shadow_pricing(shadow_pricing_info):
    """
    Allocate shared_data buffers for multiprocess shadow pricing

    Allocates one buffer per model_selector.
    Buffer datatype and shape specified by shadow_pricing_info

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
        data_buffers : dict {<model_selector> : <shared_data_buffer>}
        dict of multiprocessing.Array keyed by model_selector
    """

    dtype = shadow_pricing_info["dtype"]
    block_shapes = shadow_pricing_info["block_shapes"]

    data_buffers = {}
    for block_key, block_shape in block_shapes.items():
        # buffer_size must be int, not np.int64
        buffer_size = util.iprod(block_shape)

        csz = buffer_size * np.dtype(dtype).itemsize
        logger.info(
            "allocating shared shadow pricing buffer %s %s buffer_size %s bytes %s (%s)"
            % (block_key, buffer_size, block_shape, csz, util.GB(csz))
        )

        if np.issubdtype(dtype, np.int64):
            typecode = ctypes.c_int64
        else:
            raise RuntimeError(
                "buffer_for_shadow_pricing unrecognized dtype %s" % dtype
            )

        shared_data_buffer = multiprocessing.Array(typecode, buffer_size)

        logger.info("buffer_for_shadow_pricing added block %s" % block_key)

        data_buffers[block_key] = shared_data_buffer

    return data_buffers


def buffers_for_shadow_pricing_choice(state, shadow_pricing_choice_info):
    """
    Same as above buffers_for_shadow_price function except now we need to store
    the actual choices for the simulation based shadow pricing method

    This allocates a multiprocessing.Array that can store the choice for each person
    and then wraps a dataframe around it.  That means the dataframe can be shared
    and accessed across all threads.
    Parameters
    ----------
    shadow_pricing_info : dict
    Returns
    -------
        data_buffers : dict {<model_selector> : <shared_data_buffer>}
        dict of multiprocessing.Array keyed by model_selector
          and wrapped in a pandas dataframe
    """

    dtype = shadow_pricing_choice_info["dtype"]
    block_shapes = shadow_pricing_choice_info["block_shapes"]

    data_buffers = {}

    for block_key, block_shape in block_shapes.items():
        # buffer_size must be int, not np.int64
        buffer_size = util.iprod(block_shape)

        csz = buffer_size * np.dtype(dtype).itemsize
        logger.info(
            "allocating shared shadow pricing buffer for choices %s %s buffer_size %s bytes %s (%s)"
            % (block_key, buffer_size, block_shape, csz, util.GB(csz))
        )

        if np.issubdtype(dtype, np.int64):
            typecode = ctypes.c_int64
        else:
            raise RuntimeError(
                "buffer_for_shadow_pricing unrecognized dtype %s" % dtype
            )

        shared_data_buffer = multiprocessing.Array(typecode, buffer_size)

        logger.info("buffer_for_shadow_pricing_choice added block %s" % block_key)

        data_buffers[block_key + "_choice"] = shared_data_buffer

        persons = read_input_table(state, "persons")
        sp_choice_df = persons.reset_index()["person_id"].to_frame()

        # declare a shared Array with data from sp_choice_df
        mparr = multiprocessing.Array(ctypes.c_double, sp_choice_df.values.reshape(-1))

        # create a new df based on the shared array
        shared_sp_choice_df = pd.DataFrame(
            np.frombuffer(mparr.get_obj()).reshape(sp_choice_df.shape),
            columns=sp_choice_df.columns,
        )
        data_buffers["shadow_price_choice_df"] = shared_sp_choice_df

    return data_buffers


def shadow_price_data_from_buffers_choice(
    data_buffers, shadow_pricing_info, model_selector
):
    """

    Parameters
    ----------
    data_buffers : dict of {<model_selector> : <multiprocessing.Array>}
        multiprocessing.Array is simply a convenient way to bundle Array and Lock
        we extract the lock and wrap the RawArray in a numpy array for convenience in indexing
        The shared data buffer has shape (<num_zones, <num_segments> + 1)
        extra column is for reverse semaphores with TALLY_CHECKIN and TALLY_CHECKOUT
    shadow_pricing_info : dict
        dict of useful info
           dtype: sp_dtype,
           block_shapes : OrderedDict({<model_selector>: <shape tuple>})
           dict mapping model_selector to block shape (including extra column for semaphores)
           e.g. {'school': (num_zones, num_segments + 1)
    model_selector : str
        location type model_selector (e.g. school or workplace)

    Returns
    -------
    shared_data, shared_data_lock
        shared_data : multiprocessing.Array or None (if single process)
        shared_data_lock : numpy array wrapping multiprocessing.RawArray or None (if single process)
    """

    assert type(data_buffers) == dict

    dtype = shadow_pricing_info["dtype"]
    block_shapes = shadow_pricing_info["block_shapes"]

    if model_selector not in block_shapes:
        raise RuntimeError(
            "Model selector %s not in shadow_pricing_info" % model_selector
        )

    if block_name(model_selector + "_choice") not in data_buffers:
        raise RuntimeError(
            "Block %s not in data_buffers" % block_name(model_selector + "_choice")
        )

    data = data_buffers[block_name(model_selector + "_choice")]
    shape = (
        int(len(data) / block_shapes[model_selector][1]),
        int(block_shapes[model_selector][1]),
    )

    return np.frombuffer(data.get_obj(), dtype=dtype).reshape(shape), data.get_lock()


def shadow_price_data_from_buffers(data_buffers, shadow_pricing_info, model_selector):
    """

    Parameters
    ----------
    data_buffers : dict of {<model_selector> : <multiprocessing.Array>}
        multiprocessing.Array is simply a convenient way to bundle Array and Lock
        we extract the lock and wrap the RawArray in a numpy array for convenience in indexing
        The shared data buffer has shape (<num_zones, <num_segments> + 1)
        extra column is for reverse semaphores with TALLY_CHECKIN and TALLY_CHECKOUT
    shadow_pricing_info : dict
        dict of useful info
           dtype: sp_dtype,
           block_shapes : OrderedDict({<model_selector>: <shape tuple>})
           dict mapping model_selector to block shape (including extra column for semaphores)
           e.g. {'school': (num_zones, num_segments + 1)
    model_selector : str
        location type model_selector (e.g. school or workplace)

    Returns
    -------
    shared_data, shared_data_lock
        shared_data : multiprocessing.Array or None (if single process)
        shared_data_lock : numpy array wrapping multiprocessing.RawArray or None (if single process)
    """

    assert type(data_buffers) == dict

    dtype = shadow_pricing_info["dtype"]
    block_shapes = shadow_pricing_info["block_shapes"]

    if model_selector not in block_shapes:
        raise RuntimeError(
            "Model selector %s not in shadow_pricing_info" % model_selector
        )

    if block_name(model_selector) not in data_buffers:
        raise RuntimeError("Block %s not in data_buffers" % block_name(model_selector))

    shape = block_shapes[model_selector]
    data = data_buffers[block_name(model_selector)]

    return np.frombuffer(data.get_obj(), dtype=dtype).reshape(shape), data.get_lock()


def load_shadow_price_calculator(
    state: workflow.State, model_settings: TourLocationComponentSettings
):
    """
    Initialize ShadowPriceCalculator for model_selector (e.g. school or workplace)

    If multiprocessing, get the shared_data buffer to coordinate global_desired_size
    calculation across sub-processes

    Parameters
    ----------
    state : workflow.State
    model_settings : TourLocationComponentSettings

    Returns
    -------
    spc : ShadowPriceCalculator
    """
    if not isinstance(model_settings, TourLocationComponentSettings):
        model_settings = TourLocationComponentSettings.model_validate(model_settings)

    num_processes = state.get_injectable("num_processes", 1)

    model_selector = model_settings.MODEL_SELECTOR

    # - get shared_data from data_buffers (if multiprocessing)
    data_buffers = state.get_injectable("data_buffers", None)
    if data_buffers is not None:
        logger.info("Using existing data_buffers for shadow_price")

        # - shadow_pricing_info
        shadow_pricing_info = state.get_injectable("shadow_pricing_info", None)
        assert shadow_pricing_info is not None

        shadow_pricing_choice_info = state.get_injectable(
            "shadow_pricing_choice_info", None
        )
        assert shadow_pricing_choice_info is not None

        # - extract data buffer and reshape as numpy array
        data, lock = shadow_price_data_from_buffers(
            data_buffers, shadow_pricing_info, model_selector
        )
        data_choice, lock_choice = shadow_price_data_from_buffers_choice(
            data_buffers, shadow_pricing_choice_info, model_selector
        )
        if "shadow_price_choice_df" in data_buffers:
            shared_sp_choice_df = data_buffers["shadow_price_choice_df"]
        else:
            shared_sp_choice_df = None

    else:
        assert num_processes == 1
        data = None  # ShadowPriceCalculator will allocate its own data
        lock = None
        data_choice = None
        lock_choice = None
        shared_sp_choice_df = None

    # - ShadowPriceCalculator
    spc = ShadowPriceCalculator(
        state,
        model_settings,
        num_processes,
        data,
        lock,
        data_choice,
        lock_choice,
        shared_sp_choice_df,
    )

    return spc


@workflow.step
def add_size_tables(
    state: workflow.State,
    disaggregate_suffixes: dict[str, Any],
    scale: bool = True,
) -> None:
    """
    inject tour_destination_size_terms tables for each model_selector (e.g. school, workplace)

    Size tables are pandas dataframes with locations counts for model_selector by zone and segment
    tour_destination_size_terms

    if using shadow pricing, we scale size_table counts to sample population
    (in which case, they have to be created while single-process)

    Scaling is problematic as it breaks household result replicability across sample sizes
    It also changes the magnitude of the size terms so if they are used as utilities in
    expression files, their importance will diminish relative to other utilities as the sample
    size decreases.

    Scaling makes most sense for a full sample in conjunction with shadow pricing, where
    shadow prices can be adjusted iteratively to bring modelled counts into line with desired
    (size table) counts.
    """

    use_shadow_pricing = bool(state.settings.use_shadow_pricing)

    shadow_settings = ShadowPriceSettings.read_settings_file(
        state.filesystem, "shadow_pricing.yaml"
    )
    shadow_pricing_models = shadow_settings.shadow_pricing_models

    if shadow_pricing_models is None:
        logger.warning(
            "shadow_pricing_models list not found in shadow_pricing settings"
        )
        return

    # probably ought not scale if not shadow_pricing (breaks partial sample replicability)
    # but this allows compatability with existing CTRAMP behavior...
    scale_size_table = shadow_settings.SCALE_SIZE_TABLE

    # Suffixes for disaggregate accessibilities
    # Set default here incase None is explicitly passed
    disaggregate_suffixes = (
        {"SUFFIX": None, "ROOTS": []}
        if not disaggregate_suffixes
        else disaggregate_suffixes
    )
    suffix, roots = disaggregate_suffixes.get("SUFFIX"), disaggregate_suffixes.get(
        "ROOTS", []
    )

    assert isinstance(roots, list)
    assert (suffix is not None and roots) or (
        suffix is None and not roots
    ), "Expected to find both 'ROOTS' and 'SUFFIX', missing one"

    # shadow_pricing_models is dict of {<model_selector>: <model_name>}
    # since these are scaled to model size, they have to be created while single-process

    for model_selector, model_name in shadow_pricing_models.items():
        model_settings = TourLocationComponentSettings.read_settings_file(
            state.filesystem, model_name
        )

        if suffix is not None and roots:
            model_settings = util.suffix_tables_in_settings(
                model_settings, suffix, roots
            )

        assert model_selector == model_settings.MODEL_SELECTOR

        # assert (
        #     "SEGMENT_IDS" in model_settings
        # ), f"missing SEGMENT_IDS setting in {model_name} model_settings"
        segment_ids = model_settings.SEGMENT_IDS
        chooser_table_name = model_settings.CHOOSER_TABLE_NAME
        chooser_segment_column = model_settings.CHOOSER_SEGMENT_COLUMN_NAME

        choosers_df = state.get_dataframe(chooser_table_name)
        if model_settings.CHOOSER_FILTER_COLUMN_NAME:
            choosers_df = choosers_df[
                choosers_df[model_settings.CHOOSER_FILTER_COLUMN_NAME] != 0
            ]

        # - raw_desired_size
        land_use = state.get_dataframe("land_use")
        size_terms = get_size_terms(state)
        raw_size = tour_destination_size_terms(land_use, size_terms, model_selector)
        assert set(raw_size.columns) == set(segment_ids.keys())

        full_model_run = state.settings.households_sample_size == 0

        scale_size_table = scale and scale_size_table

        if (use_shadow_pricing and full_model_run) and scale_size_table:
            # need to scale destination size terms because ctramp and daysim approaches directly
            # compare modeled size and target size when computing shadow prices
            # Does not apply to simulation approach which compares proportions.

            # - scale size_table counts to sample population
            # scaled_size = zone_size * (total_segment_modeled / total_segment_desired)

            # segment scale factor (modeled / desired) keyed by segment_name
            segment_scale_factors = {}
            for c in raw_size:
                # number of zone demographics desired destination choices
                segment_desired_size = raw_size[c].astype(np.float64).sum()

                # number of synthetic population choosers in segment
                segment_chooser_count = (
                    choosers_df[chooser_segment_column] == segment_ids[c]
                ).sum()

                segment_scale_factors[c] = segment_chooser_count / np.maximum(
                    segment_desired_size, 1
                )

                logger.info(
                    "add_desired_size_tables %s segment %s "
                    "desired %s modeled %s scale_factor %s"
                    % (
                        chooser_table_name,
                        c,
                        segment_desired_size,
                        segment_chooser_count,
                        segment_scale_factors[c],
                    )
                )
                # FIXME - can get zero size if model_settings["CHOOSER_FILTER_COLUMN_NAME"] not yet determined / initialized to 0
                # using raw size if scaled size is 0. Is this an acceptable fix?
                # this is happening for external models where extenal identification is not run yet at this stage
                if segment_scale_factors[c] <= 0:
                    logger.warning(
                        f"scale_factor is <= 0 for {model_selector}:{c}, using raw size instead"
                    )
                    segment_scale_factors[c] = 1

            # FIXME - should we be rounding?
            # scaled_size = (raw_size * segment_scale_factors).round()
            # rounding can cause zero probability errors for small sample sizes
            scaled_size = raw_size * segment_scale_factors
        else:
            scaled_size = raw_size

        logger.debug(
            f"add_size_table {size_table_name(model_selector)} ({scaled_size.shape}) for {model_selector}"
        )

        assert (
            scaled_size.index.is_monotonic_increasing
        ), f"size table {size_table_name(model_selector)} not is_monotonic_increasing"

        state.add_table(size_table_name(model_selector), scaled_size)


def get_shadow_pricing_info(state):
    """
    return dict with info about dtype and shapes of desired and modeled size tables

    block shape is (num_zones, num_segments + 1)


    Returns
    -------
    shadow_pricing_info: dict
        dtype: <sp_dtype>,
        block_shapes: dict {<model_selector>: <block_shape>}
    """

    land_use = state.get_dataframe("land_use")
    size_terms = state.get_injectable("size_terms")

    shadow_settings = ShadowPriceSettings.read_settings_file(
        state.filesystem, "shadow_pricing.yaml"
    )

    # shadow_pricing_models is dict of {<model_selector>: <model_name>}
    shadow_pricing_models = shadow_settings.shadow_pricing_models or {}

    blocks = OrderedDict()
    for model_selector in shadow_pricing_models:
        sp_rows = len(land_use)
        sp_cols = len(size_terms[size_terms.model_selector == model_selector])

        # extra tally column for TALLY_CHECKIN and TALLY_CHECKOUT semaphores
        blocks[block_name(model_selector)] = (sp_rows, sp_cols + 1)

    sp_dtype = np.int64

    shadow_pricing_info = {
        "dtype": sp_dtype,
        "block_shapes": blocks,
    }

    for k in shadow_pricing_info:
        logger.debug(f"shadow_pricing_info {k}: {shadow_pricing_info.get(k)}")

    return shadow_pricing_info


def get_shadow_pricing_choice_info(state):
    """
    return dict with info about dtype and shapes of desired and modeled size tables

    block shape is (num_zones, num_segments + 1)


    Returns
    -------
    shadow_pricing_info: dict
        dtype: <sp_dtype>,
        block_shapes: dict {<model_selector>: <block_shape>}
    """

    persons = read_input_table(state, "persons")

    shadow_settings = ShadowPriceSettings.read_settings_file(
        state.filesystem, "shadow_pricing.yaml"
    )

    # shadow_pricing_models is dict of {<model_selector>: <model_name>}
    shadow_pricing_models = shadow_settings.shadow_pricing_models or {}

    blocks = OrderedDict()
    for model_selector in shadow_pricing_models:
        # each person will have a work or school location choice
        sp_rows = len(persons)

        # extra tally column for TALLY_CHECKIN and TALLY_CHECKOUT semaphores
        blocks[block_name(model_selector)] = (sp_rows, 2)

    sp_dtype = np.int64
    # sp_dtype = np.str

    shadow_pricing_choice_info = {
        "dtype": sp_dtype,
        "block_shapes": blocks,
    }

    for k in shadow_pricing_choice_info:
        logger.debug(
            f"shadow_pricing_choice_info {k}: {shadow_pricing_choice_info.get(k)}"
        )

    return shadow_pricing_choice_info


@workflow.cached_object
def shadow_pricing_info(state: workflow.State):
    # when multiprocessing with shared data mp_tasks has to call network_los methods
    # get_shadow_pricing_info() and buffers_for_shadow_pricing()
    logger.debug("loading shadow_pricing_info injectable")

    return get_shadow_pricing_info(state)


@workflow.cached_object
def shadow_pricing_choice_info(state: workflow.State):
    # when multiprocessing with shared data mp_tasks has to call network_los methods
    # get_shadow_pricing_info() and buffers_for_shadow_pricing()
    logger.debug("loading shadow_pricing_choice_info injectable")

    return get_shadow_pricing_choice_info(state)
