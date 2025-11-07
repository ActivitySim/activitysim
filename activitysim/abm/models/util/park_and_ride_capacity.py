import pandas as pd
import numpy as np
import ctypes
import logging
import multiprocessing as mp
import time

from activitysim.core import util, logit, tracing

logger = logging.getLogger(__name__)


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
            # grab shared data and locks created for multiprocessing
            self.shared_pnr_choice = data_buffers["shared_pnr_choice"]
            self.shared_pnr_choice_idx = data_buffers["shared_pnr_choice_idx"]
            self.shared_pnr_choice_start = data_buffers["shared_pnr_choice_start"]
            self.pnr_mp_tally = data_buffers["pnr_mp_tally"]
        else:
            assert (
                self.num_processes == 1
            ), "data_buffers must be provided for multiprocessing"
            self.shared_pnr_choice = None
            self.shared_pnr_choice_idx = None
            self.shared_pnr_choice_start = None
            self.pnr_mp_tally = None

        # occupancy counts for pnr zones is populated from choices synced across processes
        self.shared_pnr_occupancy_df = pd.DataFrame(
            columns=["pnr_occupancy"], index=state.get_dataframe("land_use").index
        )
        self.shared_pnr_occupancy_df["pnr_occupancy"] = 0
        self.capacity_snapshot = None

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
            self.choices_synced = choices[["pnr_zone_id", "start"]]

        else:
            # - multiprocessing
            self.choices_synced = self.synchronize_choices(choices)

        # tally up the counts by zone
        # index of shared_pnr_occupancy_df is zone_id in the landuse table across all processes
        # choices_synced object also now contains all choices across all processes
        pnr_counts = self.choices_synced.pnr_zone_id.value_counts().reindex(
            self.shared_pnr_occupancy_df.index
        )
        pnr_counts = pnr_counts.fillna(0).astype(int)

        # new occupancy is what was already at the lots after the last iteration + new pnr choices
        # (those selected for resimulation are removed in the select_new_choosers function)
        self.shared_pnr_occupancy_df["pnr_occupancy"].values[:] += pnr_counts.values

        # logging summary of this aggregation
        lots_with_demand = int((pnr_counts > 0).sum())
        logger.info(
            f"PNR iter {self.iteration}: aggregated {len(self.choices_synced)} choices across "
            f"{lots_with_demand} lots."
        )

    def synchronize_choices(self, choices):
        """
        Synchronize the choices across all processes.

        Parameters
        ----------
        choices : pandas.Series
            Series of choices indexed by tour_id

        Returns
        -------
        pandas.Series
            Synchronized choices across all processes.
        """
        assert self.shared_pnr_choice is not None, "shared_pnr_choice is not set"
        assert (
            self.shared_pnr_choice_idx is not None
        ), "shared_pnr_choice_idx is not set"
        assert self.num_processes > 1, "num_processes must be greater than 1"

        # barrier implemented with arrival count (idx 0) and generation (idx 1)
        # cannot just use mp.barrier() because we do not know how many processes
        # there will be when the tour mode choice iteration with pnr begins
        def barrier(reset_callback=None):
            while True:
                with self.pnr_mp_tally.get_lock():
                    gen = self.pnr_mp_tally[1]
                    self.pnr_mp_tally[0] += 1  # arrived
                    if self.pnr_mp_tally[0] == self.num_processes:
                        # last to arrive
                        if reset_callback is not None:
                            reset_callback()
                        # release all waiters by advancing generation and resetting arrival count
                        self.pnr_mp_tally[0] = 0
                        self.pnr_mp_tally[1] = gen + 1
                        return
                    # not last; remember current generation to wait on
                    wait_gen = gen
                # spin until generation changes
                while True:
                    with self.pnr_mp_tally.get_lock():
                        if self.pnr_mp_tally[1] != wait_gen:
                            break
                    time.sleep(1)
                return

        # can send in empty chocies to ensure all subprocesses will hit the barrier
        if not choices.empty:
            with self.shared_pnr_choice.get_lock():
                # first_in = self.pnr_mp_tally[0] == 0
                # create a dataframe of the already existing choices
                mp_choices_df = pd.DataFrame(
                    data={
                        "pnr_zone_id": np.frombuffer(
                            self.shared_pnr_choice.get_obj(), dtype=np.int64
                        ),
                        "start": np.frombuffer(
                            self.shared_pnr_choice_start.get_obj(), dtype=np.int64
                        ),
                    },
                    index=np.frombuffer(
                        self.shared_pnr_choice_idx.get_obj(), dtype=np.int64
                    ),
                )
                mp_choices_df.index.name = "tour_id"
                # discard zero entries
                mp_choices_df = mp_choices_df[mp_choices_df.pnr_zone_id > 0]
                # append the new choices
                synced_choices = pd.concat(
                    [mp_choices_df, choices[["pnr_zone_id", "start"]]],
                    axis=0,
                    ignore_index=False,
                )
                # sort by index (tour_id)
                synced_choices = synced_choices.sort_index()

                # now append any additional rows need to get size back to original length
                pad = len(self.shared_pnr_choice) - len(synced_choices)
                new_arr_values = np.concatenate(
                    [
                        synced_choices["pnr_zone_id"].to_numpy(np.int64),
                        np.zeros(pad, dtype=np.int64),
                    ]
                )
                new_arr_idx = np.concatenate(
                    [
                        synced_choices.index.to_numpy(np.int64),
                        np.zeros(pad, dtype=np.int64),
                    ]
                )
                new_arr_start = np.concatenate(
                    [
                        synced_choices["start"].to_numpy(np.int64),
                        np.zeros(pad, dtype=np.int64),
                    ]
                )

                # write the updated arrays back to the shared memory
                self.shared_pnr_choice_idx[:] = new_arr_idx.tolist()
                self.shared_pnr_choice[:] = new_arr_values.tolist()
                self.shared_pnr_choice_start[:] = new_arr_start.tolist()

        # Wait for all processes to finish writing
        barrier()

        # need to create the final synced_choices again since other processes may have written to the shared memory
        # don't need the lock since we are only reading at this stage
        synced_choices = pd.DataFrame(
            data={
                "pnr_zone_id": np.frombuffer(
                    self.shared_pnr_choice.get_obj(), dtype=np.int64
                ),
                "start": np.frombuffer(
                    self.shared_pnr_choice_start.get_obj(), dtype=np.int64
                ),
            },
            index=np.frombuffer(self.shared_pnr_choice_idx.get_obj(), dtype=np.int64),
        )
        synced_choices.index.name = "tour_id"
        synced_choices = synced_choices[synced_choices.pnr_zone_id > 0].copy()

        # barrier 2: last-out resets arrays for next iteration
        def reset_arrays():
            self.shared_pnr_choice[:] = np.zeros(
                len(self.shared_pnr_choice), dtype=np.int64
            ).tolist()
            self.shared_pnr_choice_idx[:] = np.zeros(
                len(self.shared_pnr_choice_idx), dtype=np.int64
            ).tolist()
            self.shared_pnr_choice_start[:] = np.zeros(
                len(self.shared_pnr_choice_start), dtype=np.int64
            ).tolist()

        barrier(reset_callback=reset_arrays)

        return synced_choices

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
        self.determine_capacitated_pnr_zones(state)
        tours_in_cap_zones = self.choices_synced[
            self.choices_synced.pnr_zone_id.isin(self.capacitated_zones)
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
                tours_in_cap_zones,
                num_over_limit.to_frame(name="num_over_limit"),
                left_on="pnr_zone_id",
                right_index=True,
                how="left",
            )

            # sort tours by order arriving at each pnr zone
            tours_in_cap_zones.sort_values(
                by=["pnr_zone_id", "start", "tour_id"],
                ascending=[True, False, True],
                inplace=True,
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

            # filtering choosers to only those tours selected for resimulation in this subprocess
            choosers = choosers[choosers.index.isin(over_capacitated_tours.index)]

            # count the total number of pnr choices being resimulated
            pnr_counts = (
                over_capacitated_tours.pnr_zone_id.value_counts()
                .reindex(self.shared_pnr_occupancy_df.index)
                .fillna(0)
                .astype(int)
            )

        elif self.model_settings.RESAMPLE_STRATEGY == "random":
            # first determine sample rate for each zone
            zonal_sample_rate = (
                self.shared_pnr_occupancy_df["pnr_occupancy"]
                / self.scaled_pnr_capacity_df["pnr_capacity"]
            )
            zonal_sample_rate = zonal_sample_rate[zonal_sample_rate > 1]
            zonal_sample_rate = (zonal_sample_rate - 1).clip(lower=0, upper=1)

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

            # filtering choosers to only those tours selected for resimulation in this subprocess
            choosers = choosers[choosers.index.isin(current_sample.index)]

            # count the total number of pnr choices being resimulated
            pnr_counts = (
                current_sample.pnr_zone_id.value_counts()
                .reindex(self.shared_pnr_occupancy_df.index)
                .fillna(0)
                .astype(int)
            )

        # subtract the counts of the resimulated tours from the occupancy
        # the pnr_counts here contains all tours across all processes
        self.shared_pnr_occupancy_df["pnr_occupancy"].values[:] -= pnr_counts.values

        return choosers

    def determine_capacitated_pnr_zones(self, state):
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

        # generating a df summarizing zonal capacity
        df = pd.DataFrame({"pnr_capacity": cap, "pnr_occupancy": occ})
        with np.errstate(divide="ignore", invalid="ignore"):
            df["pct_utilized"] = np.where(
                df.pnr_capacity > 0, df.pnr_occupancy / df.pnr_capacity * 100, np.nan
            ).round(2)
        df["over_by"] = (df.pnr_occupancy - df.pnr_capacity).clip(lower=0)
        df["capacitated"] = capacitated_zones_mask

        capacity_snapshot = df[df.pnr_capacity > 0].copy()
        capacity_snapshot.columns = df.columns + "_" + f"i{self.iteration}"
        capacity_snapshot.loc["Total"] = capacity_snapshot.sum(axis=0)

        if self.capacity_snapshot is None:
            self.capacity_snapshot = capacity_snapshot
        else:
            self.capacity_snapshot = pd.concat(
                [self.capacity_snapshot, capacity_snapshot], axis=1
            )

        # writing snapshot to output trace folder
        if self.model_settings.TRACE_PNR_CAPACITIES_PER_ITERATION:
            state.tracing.trace_df(
                df=self.capacity_snapshot,
                label=f"pnr_capacity_snapshot_i{self.iteration}",
                transpose=False,
            )

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

        return np.where(pnr_alts.index.isin(self.capacitated_zones), 1, 0)


def create_park_and_ride_capacity_data_buffers(state):
    """
    Sets up multiprocessing buffers for park-and-ride lot choice.

    One buffer for adding up the number of park-and-ride lot choice zone ids to calculate capacities.
    One other buffer keeping track of the choices for each tour so we can choose which ones to resimulate.
    This function is called before the multiprocesses are kicked off in activitysim/core/mp_tasks.py
    """

    # get landuse and person tables to determine the size of the buffers
    persons = state.get_dataframe("persons")

    # don't know a-priori how many park-and-ride tours there are at the start of the model run
    # giving the buffer a size equal to the number of persons should be sufficient
    n = len(persons)

    # creating one interprocess lock to be shared by all arrays
    shared_lock = mp.RLock()

    # need two arrays -- one for the choices and one for the IDs of the tours making the choice
    choice_arr = mp.Array(ctypes.c_int64, n, lock=shared_lock)
    choice_arr_idx = mp.Array(ctypes.c_int64, n, lock=shared_lock)
    choice_arr_start = mp.Array(ctypes.c_int64, n, lock=shared_lock)

    # init the arrays to 0
    choice_arr[:] = np.zeros(n, dtype=np.int64).tolist()
    choice_arr_idx[:] = np.zeros(n, dtype=np.int64).tolist()
    choice_arr_start[:] = np.zeros(n, dtype=np.int64).tolist()

    # one more shared array of length two to count processes as they check in and out
    pnr_mp_tally = mp.Array(ctypes.c_int64, 2, lock=shared_lock)
    pnr_mp_tally[:] = [0, 0]

    # recording memory size
    total_bytes = n * np.dtype(np.int64).itemsize * 3
    logger.info(
        f"allocating shared park-and-ride lot choice buffers with buffer_size {total_bytes} bytes ({util.GB(total_bytes)})"
    )

    data_buffers = {
        "shared_pnr_choice": choice_arr,
        "shared_pnr_choice_idx": choice_arr_idx,
        "shared_pnr_choice_start": choice_arr_start,
        "pnr_mp_tally": pnr_mp_tally,
    }

    return data_buffers
