import logging
import random
from functools import reduce

import numpy as np
import pandas as pd
from orca import orca
from sklearn.cluster import KMeans

from activitysim.abm.models import initialize, location_choice
from activitysim.abm.models.util import estimation, tour_destination
from activitysim.abm.tables import shadow_pricing
from activitysim.core import config, inject, los, pipeline, tracing, util
from activitysim.core.expressions import assign_columns

logger = logging.getLogger(__name__)


def read_disaggregate_accessibility_yaml(file_name):
    """
    Adds in default table suffixes 'proto_' if not defined in the settings file
    """
    model_settings = config.read_model_settings(file_name)
    if not model_settings.get("suffixes"):
        model_settings["suffixes"] = {
            "SUFFIX": "proto_",
            "ROOTS": [
                "persons",
                "households",
                "tours",
                "persons_merged",
                "person_id",
                "household_id",
                "tour_id",
            ],
        }
    # Convert decimal sample rate to integer sample size
    for sample in ["ORIGIN_SAMPLE_SIZE", "DESTINATION_SAMPLE_SIZE"]:
        size = model_settings.get(sample, 0)
        if size > 0 and size < 1:
            model_settings[sample] = round(
                size * len(pipeline.get_table("land_use").index)
            )

    return model_settings


class ProtoPop:
    def __init__(self, network_los, chunk_size):
        # Run necessary inits for later
        initialize.initialize_landuse()

        # Initialization
        self.proto_pop = {}
        self.zone_list = []
        self.land_use = pipeline.get_table("land_use")
        self.network_los = network_los
        self.chunk_size = chunk_size
        self.model_settings = read_disaggregate_accessibility_yaml(
            "disaggregate_accessibility.yaml"
        )

        # Random seed
        self.seed = self.model_settings.get("BASE_RANDOM_SEED", 0) + len(
            self.land_use.index
        )

        # Generation
        self.params = self.read_table_settings()
        self.create_proto_pop()
        logger.info(
            "Created a proto-population with %s households across %s origin zones to %s possible destination zones"
            % (
                len(self.proto_pop["proto_households"]),
                len(self.proto_pop["proto_households"].home_zone_id.unique()),
                self.model_settings["DESTINATION_SAMPLE_SIZE"],
            )
        )
        self.inject_tables()
        self.annotate_tables()
        self.merge_persons()

        # - initialize shadow_pricing size tables after annotating household and person tables
        # since these are scaled to model size, they have to be created while single-process
        # this can now be called as a standalone model step instead, add_size_tables
        add_size_tables = self.model_settings.get("add_size_tables", True)
        if add_size_tables:
            # warnings.warn(f"Calling add_size_tables from initialize will be removed in the future.", FutureWarning)
            shadow_pricing._add_size_tables(
                self.model_settings.get("suffixes"), scale=False
            )

    def zone_sampler(self):
        """
        This is a "pre"-sampling method, which selects a sample from the total zones and generates a proto-pop on it.
        This is particularly useful for multi-zone models where there are many MAZs.
        Otherwise it could cause memory usage and computation time to explode.

        Important to distinguish this zone sampling from the core destination_sample method. The destination method
        is destination oriented, and essentially weights samples by their size terms in order to sample important
        destinations. This is irrelevant for accessibility, which is concerned with the accessibility FROM origins
        to destinations.

        Thus, this sampling approach will weight the zones by their relative population.

        method:
            None/Default - Sample zones weighted by population, ensuring at least one TAZ is sampled per MAZ
                If n-samples > n-tazs then sample 1 MAZ from each TAZ until n-remaining-samples < n-tazs,
                then sample n-remaining-samples TAZs and sample an MAZ within each of those TAZs.
                If n-samples < n-tazs, then it proceeds to the above 'then' condition.

            uniform - Unweighted sample of N zones independent of each other.

            uniform-taz - Unweighted sample of 1 zone per taz up to the N samples specified.

            k-means - K-Means clustering is performed on the zone centroids (must be provided as maz_centroids.csv),
                weighted by population. The clustering yields k XY coordinates weighted by zone population
                for n-samples = k-clusters specified. Once k new cluster centroids are found, these are then
                approximated into the nearest available zone centroid and used to calculate accessibilities on.

                By default, the k-means method is run on 10 different initial cluster seeds (n_init) using using
                "k-means++" seeding algorithm (https://en.wikipedia.org/wiki/K-means%2B%2B). The k-means method
                runs for max_iter iterations (default=300).

        """

        # default_zone_col = 'TAZ' if not (self.network_los.zone_system == los.ONE_ZONE) else 'zone_id'
        # zone_cols = self.model_settings["zone_id_names"].get("zone_group_cols", default_zone_col)
        id_col = self.model_settings["zone_id_names"].get("index_col", "zone_id")
        method = self.model_settings.get("ORIGIN_SAMPLE_METHOD")
        n_samples = self.model_settings.get("ORIGIN_SAMPLE_SIZE", 0)

        # Get weights, need to get households first to get persons merged.
        # Note: This will cause empty zones to be excluded. Which is intended, but just know that.
        zone_weights = self.land_use[
            self.model_settings["ORIGIN_WEIGHTING_COLUMN"]
        ].to_frame("weight")
        zone_weights = zone_weights[zone_weights.weight != 0]

        # If more samples than zones, just default to all zones
        if n_samples == 0 or n_samples > len(zone_weights.index):
            n_samples = len(zone_weights.index)
            print("WARNING: ORIGIN_SAMPLE_SIZE >= n-zones. Using all zones.")
            method = "full"  # If it's a full sample, no need to sample

        if method and method == "full":
            sample_idx = self.land_use.index
        elif method and method.lower() == "uniform":
            sample_idx = sorted(random.sample(sorted(self.land_use.index), n_samples))
        elif method and method.lower() == "uniform-taz":
            # Randomly select one MAZ per TAZ by randomizing the index and then select the first MAZ in each TAZ
            # Then truncate the sampled indices by N samples and sort it
            sample_idx = (
                self.land_use.sample(frac=1)
                .reset_index()
                .groupby("TAZ")[id_col]
                .first()
            )
            sample_idx = sorted(sample_idx)
        elif method and method.lower() == "kmeans":
            # Only implemented for 2-zone system for now
            assert (
                self.network_los.zone_system == los.TWO_ZONE
            ), "K-Means only implemented for 2-zone systems for now"

            # Performs a simple k-means clustering using centroid XY coordinates
            centroids_df = pipeline.get_table("maz_centroids")

            # Assert that land_use zone ids is subset of centroid zone ids
            assert set(self.land_use.index).issubset(set(centroids_df.index))

            # Join the land_use pop on centroids,
            # this also filter only zones we need (relevant if running scaled model)
            centroids_df = centroids_df.join(
                self.land_use[self.model_settings["ORIGIN_WEIGHTING_COLUMN"]],
                how="inner",
            )
            xy_list = list(centroids_df[["X", "Y"]].itertuples(index=False, name=None))
            xy_weights = np.array(
                centroids_df[self.model_settings["ORIGIN_WEIGHTING_COLUMN"]]
            )

            # Initializer k-means class
            """
            init: (default='k-means++')
                ‘k-means++’ : selects initial cluster centroids using sampling based on an
                empirical probability distribution of the points’ contribution to the overall inertia.
                This technique speeds up convergence, and is theoretically proven to be O(log k) -optimal.
                See the description of n_init for more details.

                ‘random’: choose n_clusters observations (rows) at random from data for the initial centroids.

            n_init: (default=10)
                Number of time the k-means algorithm will be run with different centroid seeds.
                The final results will be the best output of n_init consecutive runs in terms of inertia.

            max_iter: (default=300)
                Maximum number of iterations of the k-means algorithm for a single run.

            n_clusters (pass n-samples):
                The number of clusters to form as well as the number of centroids to generate.
                This is the n-samples we are choosing.
            """

            kmeans = KMeans(
                init="k-means++",
                n_clusters=n_samples,
                n_init=10,
                max_iter=300,
                random_state=self.seed,
            )

            # Calculate the k-means cluster points
            # Find the nearest MAZ for each cluster
            kmeans_res = kmeans.fit(X=xy_list, sample_weight=xy_weights)
            sample_idx = [
                util.nearest_node_index(_xy, xy_list)
                for _xy in kmeans_res.cluster_centers_
            ]
        else:
            # Default method.
            # First sample the TAZ then select subzones weighted by the population size
            if self.network_los.zone_system == los.TWO_ZONE:
                # Join on TAZ and aggregate
                maz_candidates = zone_weights.merge(
                    self.network_los.maz_taz_df, left_index=True, right_on="MAZ"
                )
                taz_candidates = maz_candidates.groupby("TAZ").sum().drop(columns="MAZ")

                # Sample TAZs then sample sample 1 MAZ per TAZ for all TAZs, repeat MAZ sampling until no samples left
                n_samples_remaining = n_samples
                maz_sample_idx = []

                while len(maz_candidates.index) > 0 and n_samples_remaining > 0:
                    # To ensure that each TAZ gets selected at least once when n > n-TAZs
                    if n_samples_remaining >= len(maz_candidates.groupby("TAZ").size()):
                        # Sample 1 MAZ per TAZ based on weight
                        maz_sample_idx += list(
                            maz_candidates.groupby("TAZ")
                            .sample(
                                n=1,
                                weights="weight",
                                replace=False,
                                random_state=self.seed,
                            )
                            .MAZ
                        )
                    else:
                        # If there are more TAZs than samples remaining, then sample from TAZs first, then MAZs
                        # Otherwise we would end up with more samples than we want
                        taz_sample_idx = list(
                            taz_candidates.sample(
                                n=n_samples_remaining,
                                weights="weight",
                                replace=False,
                                random_state=self.seed,
                            ).index
                        )
                        # Now keep only those TAZs and sample MAZs from them
                        maz_candidates = maz_candidates[
                            maz_candidates.TAZ.isin(taz_sample_idx)
                        ]
                        maz_sample_idx += list(
                            maz_candidates.groupby("TAZ")
                            .sample(
                                n=1,
                                weights="weight",
                                replace=False,
                                random_state=self.seed,
                            )
                            .MAZ
                        )

                    # Remove selected candidates from weight list
                    maz_candidates = maz_candidates[
                        ~maz_candidates.MAZ.isin(maz_sample_idx)
                    ]
                    # Calculate the remaining samples to collect
                    n_samples_remaining = n_samples - len(maz_sample_idx)
                    n_samples_remaining = (
                        0 if n_samples_remaining < 0 else n_samples_remaining
                    )

                # The final MAZ list
                sample_idx = maz_sample_idx
            else:
                sample_idx = list(
                    zone_weights.sample(
                        n=n_samples,
                        weights="weight",
                        replace=True,
                        random_state=self.seed,
                    ).index
                )

        return {id_col: sorted(sample_idx)}

    def read_table_settings(self):
        # Check if setup properly

        assert "CREATE_TABLES" in self.model_settings.keys()

        # Set zone_id name if not already specified
        self.model_settings["zone_id_names"] = self.model_settings.get(
            "zone_id_names", {"index_col": "zone_id"}
        )
        create_tables = self.model_settings.get("CREATE_TABLES")
        from_templates = self.model_settings.get("FROM_TEMPLATES", False)
        zone_list = self.zone_sampler()
        params = {}

        assert all(
            [
                True
                for x in ["PERSONS", "HOUSEHOLDS", "TOURS"]
                if x in create_tables.keys()
            ]
        )

        if from_templates:
            params = {
                k.lower(): {"file": v, "index_col": k.lower()[:-1] + "_id"}
                for k, v in create_tables.items()
            }
            params = {**params, **zone_list}
            params["proto_households"]["zone_col"] = "home_zone_id"
        else:
            assert all(
                [True for k, v in create_tables.items() if "VARIABLES" in v.keys()]
            )
            for name, table in create_tables.items():
                # Ensure table variables are all lists
                params[name.lower()] = {
                    "variables": {
                        k: (v if isinstance(v, list) else [v])
                        for k, v in table["VARIABLES"].items()
                    },
                    "mapped": table.get("mapped_fields", []),
                    "filter": table.get("filter_rows", []),
                    "join_on": table.get("JOIN_ON", []),
                    "index_col": table.get("index_col", []),
                    "zone_col": table.get("zone_col", []),
                    "rename_columns": table.get("rename_columns", []),
                }

                # Add zones to households dicts as vary_on variable
                params["proto_households"]["variables"] = {
                    **params["proto_households"]["variables"],
                    **zone_list,
                }

        return params

    def generate_replicates(self, table_name):
        """
        Generates replicates finding the cartesian product of the non-mapped field variables.
        The mapped fields are then annotated after replication
        """
        # Generate replicates
        df = pd.DataFrame(util.named_product(**self.params[table_name]["variables"]))

        # Applying mapped variables
        if len(self.params[table_name]["mapped"]) > 0:
            for mapped_from, mapped_to_pair in self.params[table_name][
                "mapped"
            ].items():
                for name, mapped_to in mapped_to_pair.items():
                    df[name] = df[mapped_from].map(mapped_to)

        # Perform filter step
        if (len(self.params[table_name]["filter"])) > 0:
            for filt in self.params[table_name]["filter"]:
                df[eval(filt)].reset_index(drop=True)

        return df

    def expand_template_zones(self, tables):
        assert (
            len(
                set(tables["proto_households"].proto_household_id).difference(
                    tables["proto_persons"].proto_household_id
                )
            )
            == 0
        ), "Unmatched template_household_id in households/persons templates"

        assert (
            len(
                set(tables["proto_persons"].proto_person_id).difference(
                    tables["proto_tours"].proto_person_id
                )
            )
            == 0
        ), "Unmatched template_household_id in persons/tours templates"

        # Create one master template
        master_template = (
            tables["proto_households"]
            .merge(tables["proto_persons"], on="proto_household_id", how="left")
            .merge(
                tables["proto_tours"],
                on=["proto_household_id", "proto_person_id"],
                how="left",
            )
            .reset_index(drop=True)
        )

        # Run cartesian product on the index vs zones
        index_params = {
            "index": master_template.index,
            "home_zone_id": self.params.get("zone_id"),
        }

        # Create cartesian product on the index and zone id
        _expanded = pd.DataFrame(util.named_product(**index_params)).set_index("index")

        # Use result to join template onto expanded table of zones
        ex_table = _expanded.join(master_template).reset_index()

        # Concatenate a new unique set of ids
        cols = ["home_zone_id", "proto_household_id", "proto_person_id"]
        col_filler = {
            x: len(ex_table[x].unique().max().astype(str))
            for x in ["proto_household_id", "proto_person_id"]
        }

        # Convert IDs to string and pad zeroes
        df_ids = ex_table[cols].astype(str)
        for col, fill in col_filler.items():
            df_ids[col] = df_ids[col].str.zfill(fill)

        ex_table["proto_person_id"] = df_ids[cols].apply("".join, axis=1).astype(int)
        ex_table["proto_household_id"] = (
            df_ids[cols[:-1]].apply("".join, axis=1).astype(int)
        )

        # Separate out into households, persons, tours
        col_keys = {k: list(v.columns) for k, v in tables.items()}
        col_keys["proto_households"].append("home_zone_id")

        proto_tables = {k: ex_table[v].drop_duplicates() for k, v in col_keys.items()}
        proto_tables["proto_tours"] = (
            proto_tables["proto_tours"]
            .reset_index()
            .rename(columns={"index": "proto_tour_id"})
        )
        proto_tables["proto_tours"].index += 1

        return [x for x in proto_tables.values()]

    def create_proto_pop(self):
        # Separate out the mapped data from the varying data and create base replicate tables
        klist = ["proto_households", "proto_persons", "proto_tours"]

        # Create ID columns, defaults to "%tablename%_id"
        hhid, perid, tourid = [
            self.params[x]["index_col"]
            if len(self.params[x]["index_col"]) > 0
            else x + "_id"
            for x in klist
        ]

        if self.model_settings.get("FROM_TEMPLATES"):
            table_params = {k: self.params.get(k) for k in klist}
            tables = {
                k: pd.read_csv(config.config_file_path(v.get("file")))
                for k, v in table_params.items()
            }
            households, persons, tours = self.expand_template_zones(tables)
            households["household_serial_no"] = households[hhid]
        else:
            households, persons, tours = [self.generate_replicates(k) for k in klist]

            # Names
            households.name, persons.name, tours.name = klist

            # Create hhid
            households[hhid] = households.index + 1
            households["household_serial_no"] = households[hhid]

            # Assign persons to households
            rep = (
                pd.DataFrame(
                    util.named_product(hhid=households[hhid], index=persons.index)
                )
                .set_index("index")
                .rename(columns={"hhid": hhid})
            )
            persons = rep.join(persons).sort_values(hhid).reset_index(drop=True)
            persons[perid] = persons.index + 1

            # Assign persons to tours
            tkey, pkey = list(self.params["proto_tours"]["join_on"].items())[0]
            tours = tours.merge(
                persons[[pkey, hhid, perid]], left_on=tkey, right_on=pkey
            )
            tours.index = tours.index.set_names([tourid])
            tours.index += 1
            tours = tours.reset_index().drop(columns=[pkey])

        # Set index
        households.set_index(hhid, inplace=True, drop=False)
        persons.set_index(perid, inplace=True, drop=False)
        tours.set_index(tourid, inplace=True, drop=False)

        # Store tables
        self.proto_pop = {
            "proto_households": households,
            "proto_persons": persons,
            "proto_tours": tours,
        }

        # Rename any columns. Do this first before any annotating
        for tablename, df in self.proto_pop.items():
            colnames = self.params[tablename].get("rename_columns", [])
            if len(colnames) > 0:
                df.rename(columns=colnames, inplace=True)

    def inject_tables(self):
        # Update canonical tables lists
        inject.add_injectable(
            "traceable_tables",
            inject.get_injectable("traceable_tables") + list(self.proto_pop.keys()),
        )
        for tablename, df in self.proto_pop.items():
            inject.add_table(tablename, df)
            pipeline.get_rn_generator().add_channel(tablename, df)
            tracing.register_traceable_table(tablename, df)
            # pipeline.get_rn_generator().drop_channel(tablename)

    def annotate_tables(self):
        # Extract annotations
        for annotations in self.model_settings["annotate_proto_tables"]:
            tablename = annotations["tablename"]
            df = pipeline.get_table(tablename)
            assert df is not None
            assert annotations is not None
            assign_columns(
                df=df,
                model_settings={
                    **annotations["annotate"],
                    **self.model_settings["suffixes"],
                },
                trace_label=tracing.extend_trace_label("ProtoPop.annotate", tablename),
            )
            pipeline.replace_table(tablename, df)

    def merge_persons(self):
        persons = pipeline.get_table("proto_persons")
        households = pipeline.get_table("proto_households")

        # For dropping any extra columns created during merge
        cols_to_use = households.columns.difference(persons.columns)

        # persons_merged to emulate the persons_merged table in the pipeline
        persons_merged = persons.join(
            households[cols_to_use], on=self.params["proto_households"]["index_col"]
        ).merge(
            self.land_use,
            left_on=self.params["proto_households"]["zone_col"],
            right_on=self.model_settings["zone_id_names"]["index_col"],
        )

        perid = self.params["proto_persons"]["index_col"]
        persons_merged.set_index(perid, inplace=True, drop=True)
        self.proto_pop["proto_persons_merged"] = persons_merged

        # Store in pipeline
        inject.add_table("proto_persons_merged", persons_merged)


def get_disaggregate_logsums(network_los, chunk_size, trace_hh_id):
    logsums = {}
    persons_merged = pipeline.get_table("proto_persons_merged").sort_index(
        inplace=False
    )
    disagg_model_settings = read_disaggregate_accessibility_yaml(
        "disaggregate_accessibility.yaml"
    )

    for model_name in [
        "workplace_location",
        "school_location",
        "non_mandatory_tour_destination",
    ]:
        trace_label = tracing.extend_trace_label(model_name, "accessibilities")
        print("Running model {}".format(trace_label))
        model_settings = config.read_model_settings(model_name + ".yaml")
        model_settings["SAMPLE_SIZE"] = disagg_model_settings.get(
            "DESTINATION_SAMPLE_SIZE"
        )
        estimator = estimation.manager.begin_estimation(trace_label)
        if estimator:
            location_choice.write_estimation_specs(
                estimator, model_settings, model_name + ".yaml"
            )

        # Append table references in settings with "proto_"
        # This avoids having to make duplicate copies of config files for disagg accessibilities
        model_settings = util.suffix_tables_in_settings(model_settings)
        model_settings["CHOOSER_ID_COLUMN"] = "proto_person_id"

        # Include the suffix tags to pass onto downstream logsum models (e.g., tour mode choice)
        if model_settings.get("LOGSUM_SETTINGS", None):
            suffixes = util.concat_suffix_dict(disagg_model_settings.get("suffixes"))
            suffixes.insert(0, model_settings.get("LOGSUM_SETTINGS"))
            model_settings["LOGSUM_SETTINGS"] = " ".join(suffixes)

        if model_name != "non_mandatory_tour_destination":
            spc = shadow_pricing.load_shadow_price_calculator(model_settings)
            # explicitly turning off shadow pricing for disaggregate accessibilities
            spc.use_shadow_pricing = False
            # filter to only workers or students
            chooser_filter_column = model_settings["CHOOSER_FILTER_COLUMN_NAME"]
            choosers = persons_merged[persons_merged[chooser_filter_column]]

            # run location choice and return logsums
            _logsums, _ = location_choice.run_location_choice(
                choosers,
                network_los,
                shadow_price_calculator=spc,
                want_logsums=True,
                want_sample_table=True,
                estimator=estimator,
                model_settings=model_settings,
                chunk_size=chunk_size,
                chunk_tag=trace_label,
                trace_hh_id=trace_hh_id,
                trace_label=trace_label,
                skip_choice=True,
            )

            # Merge onto persons
            if _logsums is not None and len(_logsums.index) > 0:
                keep_cols = list(set(_logsums.columns).difference(choosers.columns))
                logsums[model_name] = persons_merged.merge(
                    _logsums[keep_cols], on="proto_person_id"
                )

        else:
            tours = pipeline.get_table("proto_tours")
            tours = tours[tours.tour_category == "non_mandatory"]

            _logsums, _ = tour_destination.run_tour_destination(
                tours,
                persons_merged,
                want_logsums=True,
                want_sample_table=True,
                model_settings=model_settings,
                network_los=network_los,
                estimator=estimator,
                chunk_size=chunk_size,
                trace_hh_id=trace_hh_id,
                trace_label=trace_label,
                skip_choice=True,
            )

            # Merge onto persons & tours
            if _logsums is not None and len(_logsums.index) > 0:
                tour_logsums = tours.merge(
                    _logsums["logsums"].to_frame(), left_index=True, right_index=True
                )
                keep_cols = list(
                    set(tour_logsums.columns).difference(persons_merged.columns)
                )
                logsums[model_name] = persons_merged.merge(
                    tour_logsums[keep_cols], on="proto_person_id"
                )

    return logsums


@inject.step()
def initialize_proto_population(network_los, chunk_size):
    # Synthesize the proto-population
    ProtoPop(network_los, chunk_size)
    return


@inject.step()
def compute_disaggregate_accessibility(network_los, chunk_size, trace_hh_id):
    """
    Compute enhanced disaggregate accessibility for user specified population segments,
    as well as each zone in land use file using expressions from accessibility_spec.

    """

    # Re-Register tables in this step, necessary for multiprocessing
    for tablename in ["proto_households", "proto_persons", "proto_tours"]:
        df = inject.get_table(tablename).to_frame()
        traceables = inject.get_injectable("traceable_tables")
        if tablename not in pipeline.get_rn_generator().channels:
            pipeline.get_rn_generator().add_channel(tablename, df)
        if tablename not in traceables:
            inject.add_injectable("traceable_tables", traceables + [tablename])
            tracing.register_traceable_table(tablename, df)
        del df

    # Run location choice
    logsums = get_disaggregate_logsums(network_los, chunk_size, trace_hh_id)
    logsums = {k + "_accessibility": v for k, v in logsums.items()}

    # Combined accessibility table
    # Setup dict for fixed location accessibilities
    access_list = []
    for k, df in logsums.items():
        if "non_mandatory_tour_destination" in k:
            # cast non-mandatory purposes to wide
            df = pd.pivot(
                df,
                index=["proto_household_id", "proto_person_id"],
                columns="tour_type",
                values="logsums",
            )
            df.columns = ["_".join([str(x), "accessibility"]) for x in df.columns]
            access_list.append(df)
        else:
            access_list.append(
                df[["proto_household_id", "logsums"]].rename(columns={"logsums": k})
            )
    # Merge to wide data frame. Merged on household_id, logsums are at household level
    access_df = reduce(
        lambda x, y: pd.merge(x, y, on="proto_household_id", how="outer"), access_list
    )

    # Merge in the proto pop data and inject it
    access_df = (
        access_df.merge(
            pipeline.get_table("proto_persons_merged").reset_index(),
            on="proto_household_id",
        )
        .set_index("proto_person_id")
        .sort_index()
    )

    logsums["proto_disaggregate_accessibility"] = access_df

    # Drop any tables prematurely created
    for tablename in [
        "school_destination_size",
        "workplace_destination_size",
    ]:
        pipeline.drop_table(tablename)

    for ch in list(pipeline.get_rn_generator().channels.keys()):
        pipeline.get_rn_generator().drop_channel(ch)

    # Drop any prematurely added traceables
    for trace in [
        x for x in inject.get_injectable("traceable_tables") if "proto_" not in x
    ]:
        tracing.deregister_traceable_table(trace)

    # need to clear any premature tables that were added during the previous run
    orca._TABLES.clear()
    for name, func in inject._DECORATED_TABLES.items():
        logger.debug("reinject decorated table %s" % name)
        orca.add_table(name, func)

    # Inject accessibility results into pipeline
    [inject.add_table(k, df) for k, df in logsums.items()]

    return
