import random
import logging
import pandas as pd
from functools import reduce

from activitysim.core import inject, tracing, config, pipeline, util

from activitysim.abm.models import location_choice
from activitysim.abm.models.util import tour_destination, estimation
from activitysim.abm.tables import shadow_pricing
from activitysim.core.expressions import assign_columns

from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


class ProtoPop:
    def __init__(self, land_use_df, model_settings):
        # Random seed
        random.seed(len(land_use_df.index))

        self.proto_pop = {}
        self.land_use = land_use_df
        self.model_settings = model_settings
        self.params = self.read_table_settings()

        self.create_proto_pop()
        self.inject_tables()
        self.annotate_tables()
        self.merge_persons()

    def zone_sampler(self):
        """
        This is a "pre"-sampling method, which selects a sample from the total zones and generates a proto-pop on it.
        This is particularly useful for multi-zone models where there are many MAZs.
        Otherwise it could cause memory usage and computation time to explode.

        method:
            None (default) - Simply sample N zones independent of each other.
            taz - Sample 1 zone per taz up to the N samples specified.
            k-means - ? FIXME.NF need to explore further
        """

        id_col = self.model_settings["zone_id_names"].get("index_col", "zone_id")
        zone_cols = self.model_settings["zone_id_names"].get("zone_group_cols")

        method = self.model_settings.get("zone_pre_sample_method")
        n = self.model_settings.get("zone_pre_sample_size", 0)

        if n == 0 or n > len(self.land_use.index):
            n = len(self.land_use.index)
            print(
                "Pre-sample size equals total number of zones. Using default sampling method."
            )
            method = None  # If it's a full sample, there's no need to run an aggregator

        if method and method.lower() == "taz":
            assert zone_cols is not None
            # Randomly select one MAZ per TAZ by randomizing the index and then select the first MAZ in each TAZ
            # Then truncate the sampled indices by N samples and sort it
            sample_idx = (
                self.land_use.sample(frac=1)
                .reset_index()
                .groupby(zone_cols)[id_col]
                .first()
            )
            sample_idx = sorted(sample_idx)

        elif method and method.lower() == "kmeans":
            # Performs a simple k-means clustering using centroid XY coordinates
            centroids_df = pipeline.get_table("maz_centroids")

            # Filter only the zones in the land use file (relevant if running scaled model)
            centroids_df = centroids_df[centroids_df.index.isin(self.land_use.index)]
            xy_list = list(centroids_df[["X", "Y"]].itertuples(index=False, name=None))

            # Initializer k-means class
            kmeans = KMeans(
                init="random", n_clusters=n, n_init=10, max_iter=300, random_state=42
            )

            # Calculate the k-means cluster points
            # Find the nearest MAZ for each cluster
            kmeans_res = kmeans.fit(xy_list)
            sample_idx = [
                util.nearest_node_index(_xy, xy_list)
                for _xy in kmeans_res.cluster_centers_
            ]
        else:
            sample_idx = sorted(random.sample(sorted(self.land_use.index), n))

        return {id_col: sample_idx}

    def read_table_settings(self):
        # Check if setup properly
        assert "CREATE_TABLES" in self.model_settings.keys()
        create_tables = self.model_settings["CREATE_TABLES"]
        assert all([True for k, v in create_tables.items() if "VARIABLES" in v.keys()])
        assert all(
            [
                True
                for x in ["PERSONS", "HOUSEHOLDS", "TOURS"]
                if x in create_tables.keys()
            ]
        )

        params = {}
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

        # Set zone_id name if not already specified
        self.model_settings["zone_id_names"] = self.model_settings.get(
            "zone_id_names", {"index_cols": "zone_id"}
        )

        # Add in the zone variables
        zone_list = self.zone_sampler()

        # Add zones to households dicts as vary_on variable
        params["proto_households"]["variables"] = {
            **params["proto_households"]["variables"],
            **zone_list,
        }

        # Add suffixes if not defined
        if not self.model_settings.get("suffixes"):
            self.model_settings["suffixes"] = {
                "SUFFIX": "proto_",
                "ROOTS": ["persons", "households", "tours", "persons_merged"],
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

    def create_proto_pop(self):
        # Separate out the mapped data from the varying data and create base replicate tables
        klist = ["proto_households", "proto_persons", "proto_tours"]
        households, persons, tours = [self.generate_replicates(k) for k in klist]

        # Names
        households.name, persons.name, tours.name = klist

        # Create ID columns, defaults to "%tablename%_id"
        hhid, perid, tourid = [
            self.params[x]["index_col"]
            if len(self.params[x]["index_col"]) > 0
            else x + "_id"
            for x in klist
        ]

        # Create hhid
        households[hhid] = households.index + 1
        households["household_serial_no"] = households[hhid]

        # Assign persons to households
        rep = (
            pd.DataFrame(util.named_product(hhid=households[hhid], index=persons.index))
            .set_index("index")
            .rename(columns={"hhid": hhid})
        )
        persons = rep.join(persons).sort_values(hhid).reset_index(drop=True)
        persons[perid] = persons.index + 1

        # Assign persons to tours
        tkey, pkey = list(self.params["proto_tours"]["join_on"].items())[0]
        tours = tours.merge(persons[[pkey, hhid, perid]], left_on=tkey, right_on=pkey)
        tours.index = tours.index.set_names([tourid])
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
            colnames = self.params[tablename]["rename_columns"]
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
    disagg_model_settings = config.read_model_settings(
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
        model_settings["SAMPLE_SIZE"] = disagg_model_settings.get("SAMPLE_SIZE")
        estimator = estimation.manager.begin_estimation(trace_label)
        if estimator:
            location_choice.write_estimation_specs(
                estimator, model_settings, model_name + ".yaml"
            )

        # Append table references in settings with "proto_"
        # This avoids having to make duplicate copies of config files for disagg accessibilities
        model_settings = util.suffix_tables_in_settings(model_settings)

        # Include the suffix tags to pass onto downstream logsum models (e.g., tour mode choice)
        if model_settings.get("LOGSUM_SETTINGS", None):
            suffixes = util.concat_suffix_dict(disagg_model_settings.get("suffixes"))
            suffixes.insert(0, model_settings.get("LOGSUM_SETTINGS"))
            model_settings["LOGSUM_SETTINGS"] = " ".join(suffixes)

        if model_name != "non_mandatory_tour_destination":
            shadow_price_calculator = shadow_pricing.load_shadow_price_calculator(
                model_settings
            )
            # filter to only workers or students
            chooser_filter_column = model_settings["CHOOSER_FILTER_COLUMN_NAME"]
            choosers = persons_merged[persons_merged[chooser_filter_column]]

            # run location choice and return logsums
            _logsums, _ = location_choice.run_location_choice(
                choosers,
                network_los,
                shadow_price_calculator=shadow_price_calculator,
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
                    _logsums[keep_cols], on="person_id"
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
                    tour_logsums[keep_cols], on="person_id"
                )

    return logsums


@inject.step()
def initialize_proto_population():
    # Synthesize the proto-population
    model_settings = config.read_model_settings("disaggregate_accessibility.yaml")
    land_use_df = pipeline.get_table("land_use")
    ProtoPop(land_use_df, model_settings)

    return


@inject.step()
def compute_disaggregate_accessibility(network_los, chunk_size, trace_hh_id):
    """
    Compute enhanced disaggregate accessibility for user specified population segments,
    as well as each zone in land use file using expressions from accessibility_spec.

    """

    # Synthesize the proto-population
    model_settings = config.read_model_settings("disaggregate_accessibility.yaml")

    # - initialize shadow_pricing size tables after annotating household and person tables
    # since these are scaled to model size, they have to be created while single-process
    # this can now be called as a standalone model step instead, add_size_tables
    add_size_tables = model_settings.get("add_size_tables", True)
    if add_size_tables:
        # warnings.warn(f"Calling add_size_tables from initialize will be removed in the future.", FutureWarning)
        shadow_pricing.add_size_tables(model_settings.get("suffixes"))

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

    # # De-register the channel, so it can get re-registered with actual pop tables
    [
        pipeline.drop_table(x)
        for x in ["school_destination_size", "workplace_destination_size", "tours"]
    ]

    # Combined accessibility table
    # Setup dict for fixed location accessibilities
    access_list = []
    for k, df in logsums.items():
        if "non_mandatory_tour_destination" in k:
            # cast non-mandatory purposes to wide
            df = pd.pivot(
                df,
                index=["household_id", "person_id"],
                columns="tour_type",
                values="logsums",
            )
            df.columns = ["_".join([str(x), "accessibility"]) for x in df.columns]
            access_list.append(df)
        else:
            access_list.append(
                df[["household_id", "logsums"]].rename(columns={"logsums": k})
            )
    # Merge to wide data frame. Merged on household_id, logsums are at household level
    access_df = reduce(
        lambda x, y: pd.merge(x, y, on="household_id", how="outer"), access_list
    )

    # Merge in the proto pop data and inject it
    access_df = (
        access_df.merge(
            pipeline.get_table("proto_persons_merged").reset_index(), on="household_id"
        )
        .set_index("person_id")
        .sort_index()
    )
    inject.add_table("proto_disaggregate_accessibility", access_df)

    # Inject separate accessibilities into pipeline
    [inject.add_table(k, df) for k, df in logsums.items()]

    return
