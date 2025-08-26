import pandas as pd
import numpy as np
import ctypes
import logging
import multiprocessing

from activitysim.core import util, logit

logger = logging.getLogger(__name__)

TALLY_CHECKIN = (0, -1)
TALLY_CHECKOUT = (1, -1)
TALLY_PENDING_PERSONS = (2, -1)


class ParkAndRideCapacity:
    """
    Class to handle park-and-ride lot capacity calculations.

    This class is used to get and set park-and-ride lot choices for tours
    and handles sending the choices across other multiprocess steps.
    It works very similarly to shadow pricing -- see `activitysim/abm/tables/shadow_pricing.py`
    """

    def __init__(self, state, model_settings):
        self.model_settings = model_settings
        self.num_processes = state.get_injectable("num_processes", 1)

        data_buffers = state.get_injectable("data_buffers", None)
        self.iteration = 0

        if data_buffers is not None:
            self.shared_pnr_occupancy_df = data_buffers["pnr_occupancy"]
            self.shared_pnr_choice_df = data_buffers["pnr_choices"]
            self.pnr_data_lock = data_buffers["pnr_data_lock"]

        else:
            assert (
                self.num_processes == 1
            ), "data_buffers must be provided for multiprocessing"
            self.shared_pnr_occupancy_df = pd.DataFrame(
                columns=["pnr_occupancy"], index=state.get_dataframe("land_use").index
            )
            self.shared_pnr_occupancy_df["pnr_occupancy"] = 0
            self.shared_pnr_choice_df = pd.DataFrame(columns=["tour_id", "pnr_zone_id"])
            self.pnr_data_lock = None

        self.scale_pnr_capacity(state)

    def set_choices(self, choices):
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
        # first remove non-park-and-ride choices
        pnr_modes = self.model_settings.PARK_AND_RIDE_MODES
        assert (
            pnr_modes is not None
        ), "PARK_AND_RIDE_MODES must be set in model settings"
        choices = choices[choices.tour_mode.isin(pnr_modes)]

        if self.num_processes == 1:
            # - not multiprocessing
            self.choices_synced = choices.pnr_zone_id

            # tally up the counts by zone
            # index of shared_pnr_occupancy_df is zone_id in the landuse table across all processes
            pnr_counts = self.choices_synced.value_counts().reindex(
                self.shared_pnr_occupancy_df.index
            )
            pnr_counts = pnr_counts.fillna(0).astype(int)

            # new occupancy is what was already at the lots after the last iteration + new pnr choices
            # (those selected for resimulation are removed in the select_new_choosers function)
            self.shared_pnr_occupancy_df["pnr_occupancy"].values[:] += pnr_counts.values

        else:
            # - multiprocessing
            self.choices_synced = self.synchronize_choices(choices)

    def synchronize_choices(self, choices):
        """
        Synchronize the choices across all processes.

        Parameters
        ----------
        choices : pandas.Series
            Series of choices indexed by person_id

        Returns
        -------
        pandas.Series
            Synchronized choices across all processes.
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

        with self.shared_data_choice_lock:
            first_in = self.shared_data_choice[TALLY_CHECKIN] == 0

        pass

    def scale_pnr_capacity(self, state):
        """
        scale the pnr capacity based on the simulation sample rate

        Parameters
        ----------
        choices_df : pandas.DataFrame
            DataFrame containing the choices made by each person.

        Returns
        -------
        None -- class variable self.scaled_pnr_capacity_df is updated
        """

        # scale the capacities based on simulation sample rate
        capacities = state.get_dataframe("land_use")[
            self.model_settings.LANDUSE_PNR_SPACES_COLUMN
        ]
        sample_rate = state.get_dataframe("households")["sample_rate"].mode()[0]
        scaled_capacities = np.ceil(capacities * sample_rate)
        scaled_capacities = scaled_capacities.clip(lower=0).fillna(0).astype(int)

        # update the shared DataFrame with the new capacities
        self.scaled_pnr_capacity_df = scaled_capacities.to_frame(name="pnr_capacity")

    def select_new_choosers(self, state, choosers):
        """
        Select new choosers for park-and-ride lots based on the current choices.

        Parameters
        ----------
        state : object
            The current state of the simulation.
        choosers : pandas.DataFrame
            DataFrame containing the current choosers.

        Returns
        -------
        pandas.DataFrame
            Updated DataFrame with the new choosers in this process.
        """
        # select tours from capacitated zones
        # note that this dataframe contains tours across all processes but choosers is only from the current process
        self.determine_capacitated_pnr_zones()
        tours_in_cap_zones = self.choices_synced[
            self.choices_synced.isin(self.capacitated_zones)
        ]

        # if no tours in capacitated zones, return empty DataFrame
        if tours_in_cap_zones.empty:
            return pd.DataFrame(columns=choosers.columns)

        if self.model_settings.RESAMPLE_STRATEGY == "latest":
            # determining how many tours to select from each over-capacitated zone
            num_over_limit = (
                self.shared_pnr_occupancy_df["pnr_occupancy"]
                - self.scaled_pnr_capacity_df["pnr_capacity"]
            )
            num_over_limit = num_over_limit[num_over_limit > 0].astype(int)

            tours_in_cap_zones = pd.merge(
                tours_in_cap_zones.to_frame(name="pnr_zone_id"),
                num_over_limit.to_frame(name="num_over_limit"),
                left_on="pnr_zone_id",
                right_index=True,
                how="left",
            )
            tours_in_cap_zones["start"] = choosers.loc[
                tours_in_cap_zones.index, "start"
            ]

            # sort tours by order arriving at each pnr zone
            tours_in_cap_zones.sort_values(
                by=["pnr_zone_id", "start"], ascending=[True, False], inplace=True
            )
            # counting tours in each pnr zone numbered by reverse arrival order
            tours_in_cap_zones["arrival_num_latest"] = (
                tours_in_cap_zones.groupby("pnr_zone_id").cumcount() + 1
            )

            # tours_in_cap_zones
            # tour_id | pnr_zone_id | num_over_limit | start | arrival_num_latest
            # 123         5              2               10          1
            # 456         5              2               9           2
            # 789         5              2               8           3

            # selecting tours that are over the capacity limit
            over_capacitated_tours = tours_in_cap_zones[
                tours_in_cap_zones["arrival_num_latest"]
                <= tours_in_cap_zones["num_over_limit"]
            ]

            choosers = choosers[choosers.index.isin(over_capacitated_tours.index)]

        elif self.model_settings.RESAMPLE_STRATEGY == "random":
            # first determine sample rate for each zone
            zonal_sample_rate = (
                self.shared_pnr_occupancy_df["pnr_occupancy"]
                / self.scaled_pnr_capacity_df["pnr_capacity"]
            )
            zonal_sample_rate = zonal_sample_rate[zonal_sample_rate > 1]

            # person's probability of being selected for re-simulation is from the zonal sample rate
            sample_rates = tours_in_cap_zones.pnr_zone_id.map(
                zonal_sample_rate.to_dict()
            )
            probs = pd.DataFrame(
                data={"0": 1 - sample_rates, "1": sample_rates},
                index=tours_in_cap_zones.index,
            )
            # using ActivitySim's RNG to make choices for repeatability
            current_sample, rands = logit.make_choices(state, probs)
            current_sample = current_sample[current_sample == 1]

            choosers = choosers[choosers.index.isin(current_sample.index)]

        # subtract the number of selected tours from the occupancy counts since they are getting resimulated
        pnr_counts = (
            choosers.pnr_zone_id.value_counts()
            .reindex(self.shared_pnr_occupancy_df.index)
            .fillna(0)
            .astype(int)
        )
        self.shared_pnr_occupancy_df["pnr_occupancy"].values[:] -= pnr_counts.values

        return choosers

    def determine_capacitated_pnr_zones(self):
        """
        Determine which park-and-ride zones are at or over capacity.

        Returns
        -------
        None -- class variable self.capacitated_zones is updated
        """
        tol = self.model_settings.ACCEPTED_TOLERANCE

        cap = self.scaled_pnr_capacity_df["pnr_capacity"]
        occ = self.shared_pnr_occupancy_df["pnr_occupancy"]
        assert cap.index.equals(
            occ.index
        ), "capacity and occupancy indices do not match"

        valid = cap > 0  # ignore zero-capacity zones
        # capacitated if at least tol fraction full (or over capacity)
        capacitated_zones_mask = valid & (occ >= np.ceil(cap * tol))

        self.capacitated_zones = cap.index[capacitated_zones_mask]

    def flag_capacitated_pnr_zones(self, pnr_alts):
        """
        Flag park-and-ride lots that are at capacity.
        Not removing them from the alternatives, just setting a flag for use in utility calculations.
        Don't want to change the number of alternatives between iterations for better tracing.

        Parameters
        ----------
        pnr_alts : pandas.DataFrame
            landuse table with park-and-ride alternatives

        Returns
        -------
        pandas.DataFrame
            updated landuse table
        """
        # capacitated pnr zones
        self.determine_capacitated_pnr_zones()
        capacitated_zones_mask = pnr_alts.index.isin(self.capacitated_zones)

        return np.where(capacitated_zones_mask, 1, 0)


def create_park_and_ride_capacity_data_buffers(state):
    """
    Sets up multiprocessing buffers for park-and-ride lot choice.

    One buffer for adding up the number of park-and-ride lot choice zone ids to calculate capacities.
    One other buffer keeping track of the choices for each tour so we can choose which ones to resimulate.
    This function is called before the multiprocesses are kicked off in activitysim/core/mp_tasks.py
    """

    # get landuse and person tables to determine the size of the buffers
    land_use = state.get_dataframe("land_use")
    persons = state.get_dataframe("persons")

    # canonical, identical across processes
    zone_ids = land_use.reset_index()["zone_id"].to_numpy(dtype=np.int64)
    n_zones = zone_ids.size

    # shared occupancy: 1 value per zone (zone_id itself is NOT shared)
    mparr_occupancy = multiprocessing.Array(ctypes.c_int64, n_zones, lock=True)
    occ_view = np.frombuffer(mparr_occupancy.get_obj(), dtype=np.int64, count=n_zones)
    occ_view[:] = 0  # init

    # DataFrame wrapper with fixed index in the same order in every process
    shared_pnr_occupancy_df = pd.DataFrame(
        {"pnr_occupancy": occ_view},
        index=pd.Index(zone_ids, name="zone_id"),
    )

    # don't know a-priori how many park-and-ride tours there are at the start of the model run
    # giving the buffer a size equal to the number of persons should be sufficient
    # need column for tour_id and column for choice
    pnr_choice_df = persons.reset_index()["person_id"].to_frame()
    pnr_choice_df["pnr_lot_id"] = -1  # default value for park-and-ride lot choice
    pnr_choice_df["tour_id"] = -1
    mparr_choice = multiprocessing.Array(
        ctypes.c_int64, pnr_choice_df.values.reshape(-1)
    )

    # create a new df based on the shared array
    shared_pnr_choice_df = pd.DataFrame(
        np.frombuffer(mparr_choice.get_obj()).reshape(pnr_choice_df.shape),
        columns=pnr_choice_df.columns,
    )

    total_bytes = mparr_occupancy.get_obj().nbytes + mparr_choice.get_obj().nbytes

    logger.info(
        f"allocating shared park-and-ride lot choice buffers with buffer_size {total_bytes} bytes ({util.GB(total_bytes)})"
    )

    data_buffers = {
        "pnr_occupancy": shared_pnr_occupancy_df,
        "pnr_choices": shared_pnr_choice_df,
        "pnr_data_lock": mparr_choice.get_lock(),
    }
    return data_buffers
