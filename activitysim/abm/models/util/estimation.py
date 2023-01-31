# ActivitySim
# See full license in LICENSE.txt.

import logging
import os
import shutil

import pandas as pd
import yaml

from activitysim.abm.models.util import canonical_ids as cid
from activitysim.core import config, simulate
from activitysim.core.util import reindex

logger = logging.getLogger("estimation")

ESTIMATION_SETTINGS_FILE_NAME = "estimation.yaml"


def unlink_files(directory_path, file_types=("csv", "yaml")):
    for file_name in os.listdir(directory_path):
        if file_name.endswith(file_types):
            file_path = os.path.join(directory_path, file_name)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    print(f"deleted {file_path}")
            except Exception as e:
                print(e)


class Estimator(object):
    def __init__(self, bundle_name, model_name, estimation_table_recipes):

        logger.info("Initialize Estimator for'%s'" % (model_name,))

        self.bundle_name = bundle_name
        self.model_name = model_name
        self.settings_name = model_name
        self.estimation_table_recipes = estimation_table_recipes
        self.estimating = True

        # ensure the output data directory exists
        output_dir = self.output_directory()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)  # make directory if needed

        # delete estimation files
        unlink_files(self.output_directory(), file_types=("csv", "yaml"))
        if self.bundle_name != self.model_name:
            # kind of inelegant to always delete these, but ok as they are redundantly recreated for each sub model
            unlink_files(
                self.output_directory(bundle_directory=True), file_types=("csv", "yaml")
            )

        # FIXME - not required?
        # assert 'override_choices' in self.model_settings, \
        #     "override_choices not found for %s in %s." % (model_name, ESTIMATION_SETTINGS_FILE_NAME)

        self.omnibus_tables = self.estimation_table_recipes["omnibus_tables"]
        self.omnibus_tables_append_columns = self.estimation_table_recipes[
            "omnibus_tables_append_columns"
        ]
        self.tables = {}
        self.tables_to_cache = [
            table_name
            for tables in self.omnibus_tables.values()
            for table_name in tables
        ]
        self.alt_id_column_name = None
        self.chooser_id_column_name = None

    @property
    def want_unsampled_alternatives(self):
        # use complete set of alternatives for estimation of interaction simulation (e.g. destination choice)
        # WARNING location_choice expects unsample alts so it can retrieve mode choice logsums for overridden choice
        # if we allow estimation based on sampled alternatives, location_choice may need to compute logsum
        return True

    def log(self, msg, level=logging.INFO):
        logger.log(level, "%s: %s" % (self.model_name, msg))

    def info(self, msg):
        self.log(msg, level=logging.INFO)

    def debug(self, msg):
        self.log(msg, level=logging.DEBUG)

    def warning(self, msg):
        self.log(msg, level=logging.WARNING)

    def set_alt_id(self, alt_id):
        self.alt_id_column_name = alt_id

    def get_alt_id(self):
        if self.alt_id_column_name is None:
            self.warning("alt_id is None. Did you forget to call set_alt_id()?")
        assert self.alt_id_column_name is not None
        return self.alt_id_column_name

    def set_chooser_id(self, chooser_id_column_name):
        self.chooser_id_column_name = chooser_id_column_name

    def get_chooser_id(self):
        if self.chooser_id_column_name is None:
            self.warning("chooser_id is None. Did you forget to call set_chooser_id()?")
        assert self.chooser_id_column_name is not None
        return self.chooser_id_column_name

    def end_estimation(self):

        self.write_omnibus_table()

        self.estimating = False
        self.tables = None

        self.info("end estimation")

        manager.release(self)

    def output_directory(self, bundle_directory=False):

        # shouldn't be asking for this if not estimating
        assert self.estimating
        assert self.model_name is not None

        dir = os.path.join(
            config.output_file_path("estimation_data_bundle"), self.bundle_name
        )

        if bundle_directory:
            # shouldn't be asking - probably confused
            assert self.bundle_name != self.model_name

        if self.bundle_name != self.model_name and not bundle_directory:
            dir = os.path.join(dir, self.model_name)

        return dir

    def output_file_path(self, table_name, file_type=None, bundle_directory=False):

        # shouldn't be asking for this if not estimating
        assert self.estimating

        output_dir = self.output_directory(bundle_directory)

        if bundle_directory:
            file_name = f"{self.bundle_name}_{table_name}"
        else:
            if self.model_name == self.bundle_name:
                file_name = f"{self.model_name}_{table_name}"
            else:
                file_name = f"{self.bundle_name}_{table_name}"

        if file_type and os.path.splitext(file_name)[1] != f".{file_type}":
            file_name = f"{file_name}.{file_type}"

        return os.path.join(output_dir, file_name)

    def write_table(
        self, df, table_name, index=True, append=True, bundle_directory=False
    ):
        """

        Parameters
        ----------
        df
        table_name: str
            if table_name has file type, then pass through filename without adding model or bundle name prefix
        index: booelan
        append: boolean
        bundle_directory: boolean

        """

        def cache_table(df, table_name, append):
            if table_name in self.tables and not append:
                raise RuntimeError(
                    "cache_table %s append=False and table exists" % (table_name,)
                )
            if table_name in self.tables:
                self.tables[table_name] = pd.concat([self.tables[table_name], df])
            else:
                self.tables[table_name] = df.copy()

        def write_table(df, table_name, index, append, bundle_directory):
            if table_name.endswith(".csv"):
                # pass through filename without adding model or bundle name prefix
                file_path = os.path.join(
                    self.output_directory(bundle_directory), table_name
                )
            else:
                file_path = self.output_file_path(table_name, "csv", bundle_directory)
            file_exists = os.path.isfile(file_path)
            if file_exists and not append:
                raise RuntimeError(
                    "write_table %s append=False and file exists: %s"
                    % (table_name, file_path)
                )
            df.to_csv(file_path, mode="a", index=index, header=(not file_exists))

        assert self.estimating

        cache = table_name in self.tables_to_cache
        write = not cache
        # write = True

        if cache:
            cache_table(df, table_name, append)
            self.debug("write_table cache: %s" % table_name)

        if write:
            write_table(df, table_name, index, append, bundle_directory)
            self.debug("write_table write: %s" % table_name)

    def write_omnibus_table(self):

        if len(self.omnibus_tables) == 0:
            return

        for omnibus_table, table_names in self.omnibus_tables.items():

            self.debug(
                "write_omnibus_table: %s table_names: %s" % (omnibus_table, table_names)
            )
            for t in table_names:
                if t not in self.tables:
                    self.warning(
                        "write_omnibus_table: %s table '%s' not found"
                        % (omnibus_table, t)
                    )

            # ignore any tables not in cache
            table_names = [t for t in table_names if t in self.tables]
            concat_axis = (
                1 if omnibus_table in self.omnibus_tables_append_columns else 0
            )

            df = pd.concat([self.tables[t] for t in table_names], axis=concat_axis)
            df.sort_index(ascending=True, inplace=True, kind="mergesort")

            file_path = self.output_file_path(omnibus_table, "csv")
            assert not os.path.isfile(file_path)

            df.to_csv(file_path, mode="a", index=True, header=True)

            self.debug("write_omnibus_choosers: %s" % file_path)

    def write_dict(self, d, dict_name, bundle_directory):

        assert self.estimating

        file_path = self.output_file_path(dict_name, "yaml", bundle_directory)

        # we don't know how to concat, and afraid to overwrite
        assert not os.path.isfile(file_path)

        with open(file_path, "w") as f:
            # write ordered dict as array
            yaml.dump(d, f)

        self.debug("estimate.write_dict: %s" % file_path)

    def write_coefficients(
        self, coefficients_df=None, model_settings=None, file_name=None
    ):
        """
        Because the whole point of estimation is to generate new coefficient values
        we want to make it easy to put the coefficients file back in configs
        So we make a point of preserving the same filename as the original config file
        """

        if model_settings is not None:
            assert file_name is None
            file_name = model_settings["COEFFICIENTS"]

        assert file_name is not None

        if coefficients_df is None:
            coefficients_df = simulate.read_model_coefficients(file_name=file_name)

        # preserve original config file name
        base_file_name = os.path.basename(file_name)

        assert self.estimating
        self.write_table(coefficients_df, base_file_name, append=False)

    def write_coefficients_template(self, model_settings):
        assert self.estimating

        coefficients_df = simulate.read_model_coefficient_template(model_settings)
        tag = "coefficients_template"
        self.write_table(coefficients_df, tag, append=False)

    def write_choosers(self, choosers_df):
        self.write_table(choosers_df, "choosers", append=True)

    def write_choices(self, choices):
        if isinstance(choices, pd.Series):
            choices = choices.to_frame(name="model_choice")
        assert list(choices.columns) == ["model_choice"]
        self.write_table(choices, "choices", append=True)

    def write_override_choices(self, choices):
        if isinstance(choices, pd.Series):
            choices = choices.to_frame(name="override_choice")
        assert list(choices.columns) == ["override_choice"]
        self.write_table(choices, "override_choices", append=True)

    def write_constants(self, constants):
        self.write_dict(self, constants, "model_constants")

    def write_nest_spec(self, nest_spec):
        self.write_dict(self, nest_spec, "nest_spec")

    def copy_model_settings(
        self, settings_file_name, tag="model_settings", bundle_directory=False
    ):

        input_path = config.base_settings_file_path(settings_file_name)

        output_path = self.output_file_path(tag, "yaml", bundle_directory)

        shutil.copy(input_path, output_path)

    def write_model_settings(
        self, model_settings, settings_file_name, bundle_directory=False
    ):

        if "include_settings" in model_settings:
            file_path = self.output_file_path(
                "model_settings", "yaml", bundle_directory
            )
            assert not os.path.isfile(file_path)
            with open(file_path, "w") as f:
                yaml.dump(model_settings, f)
        else:
            self.copy_model_settings(
                settings_file_name, bundle_directory=bundle_directory
            )
        if "inherit_settings" in model_settings:
            self.write_dict(
                model_settings, "inherited_model_settings", bundle_directory
            )

    def melt_alternatives(self, df):

        alt_id_name = self.alt_id_column_name

        assert alt_id_name is not None, (
            "alt_id not set. Did you forget to call set_alt_id()? (%s)"
            % self.model_name
        )
        assert (
            alt_id_name in df
        ), "alt_id_column_name '%s' not in alternatives table (%s)" % (
            alt_id_name,
            self.model_name,
        )

        variable_column = "variable"

        #            alt_dest  util_dist_0_1  util_dist_1_2  ...
        # person_id                                          ...
        # 31153             1            1.0           0.75  ...
        # 31153             2            1.0           0.46  ...
        # 31153             3            1.0           0.28  ...

        if df.index.name is not None:
            chooser_name = df.index.name
            assert self.chooser_id_column_name in (chooser_name, None)
            df = df.reset_index()
        else:
            assert self.chooser_id_column_name is not None
            chooser_name = self.chooser_id_column_name
            assert chooser_name in df

        # mergesort is the only stable sort, and we want the expressions to appear in original df column order
        melt_df = (
            pd.melt(df, id_vars=[chooser_name, alt_id_name])
            .sort_values(by=chooser_name, kind="mergesort")
            .rename(columns={"variable": variable_column})
        )

        # person_id,alt_dest,expression,value
        # 31153,1,util_dist_0_1,1.0
        # 31153,2,util_dist_0_1,1.0
        # 31153,3,util_dist_0_1,1.0

        melt_df = melt_df.set_index(
            [chooser_name, variable_column, alt_id_name]
        ).unstack(2)
        melt_df.columns = melt_df.columns.droplevel(0)
        melt_df = melt_df.reset_index(1)

        # person_id,expression,1,2,3,4,5,...
        # 31153,util_dist_0_1,0.75,0.46,0.27,0.63,0.48,...
        # 31153,util_dist_1_2,0.0,0.0,0.0,0.0,0.0,...
        # 31153,util_dist_2_3,0.0,0.0,0.0,0.0,0.0,...

        return melt_df

    def write_interaction_expression_values(self, df):
        df = self.melt_alternatives(df)
        self.write_table(df, "interaction_expression_values", append=True)

    def write_expression_values(self, df):
        self.write_table(df, "expression_values", append=True)

    def write_alternatives(self, alternatives_df, bundle_directory=False):
        self.write_table(
            alternatives_df,
            "alternatives",
            append=True,
            bundle_directory=bundle_directory,
        )

    def write_interaction_sample_alternatives(self, alternatives_df):
        alternatives_df = self.melt_alternatives(alternatives_df)
        self.write_table(
            alternatives_df, "interaction_sample_alternatives", append=True
        )

    def write_interaction_simulate_alternatives(self, interaction_df):
        interaction_df = self.melt_alternatives(interaction_df)
        self.write_table(
            interaction_df, "interaction_simulate_alternatives", append=True
        )

    def get_survey_values(self, model_values, table_name, column_names):
        # convenience method so deep callers don't need to import estimation
        assert self.estimating
        return manager.get_survey_values(model_values, table_name, column_names)

    def get_survey_table(self, table_name):
        # convenience method so deep callers don't need to import estimation
        assert self.estimating
        return manager.get_survey_table(table_name)

    def write_spec(
        self, model_settings=None, file_name=None, tag="SPEC", bundle_directory=False
    ):

        if model_settings is not None:
            assert file_name is None
            file_name = model_settings[tag]

        input_path = config.config_file_path(file_name)

        table_name = tag  # more readable than full spec file_name
        output_path = self.output_file_path(table_name, "csv", bundle_directory)
        shutil.copy(input_path, output_path)
        self.debug("estimate.write_spec: %s" % output_path)


class EstimationManager(object):
    def __init__(self):

        self.settings_initialized = False
        self.bundles = []
        self.estimation_table_recipes = {}
        self.model_estimation_table_types = {}
        self.estimating = {}

    def initialize_settings(self):

        # FIXME - can't we just initialize in init and handle no-presence of settings file as not enabled
        if self.settings_initialized:
            return

        assert not self.settings_initialized
        settings = config.read_model_settings(ESTIMATION_SETTINGS_FILE_NAME)
        self.enabled = settings.get("enable", "True")
        self.bundles = settings.get("bundles", [])

        self.model_estimation_table_types = settings.get(
            "model_estimation_table_types", {}
        )
        self.estimation_table_recipes = settings.get("estimation_table_recipes", {})

        if self.enabled:

            self.survey_tables = settings.get("survey_tables", {})
            for table_name, table_info in self.survey_tables.items():
                assert (
                    "file_name" in table_info
                ), "No file name specified for survey_table '%s' in %s" % (
                    table_name,
                    ESTIMATION_SETTINGS_FILE_NAME,
                )
                file_path = config.data_file_path(
                    table_info["file_name"], mandatory=True
                )
                assert os.path.exists(
                    file_path
                ), "File for survey table '%s' not found: %s" % (table_name, file_path)
                df = pd.read_csv(file_path)
                index_col = table_info.get("index_col")
                if index_col is not None:
                    assert (
                        index_col in df.columns
                    ), "Index col '%s' not in survey_table '%s' in file: %s % (index_col, table_name, file_path)"
                    df.set_index(index_col, inplace=True)

                # add the table df to survey_tables
                table_info["df"] = df

        self.settings_initialized = True

    def begin_estimation(self, model_name, bundle_name=None):
        """
        begin estimating of model_name is specified as model to estimate, otherwise return False

        Parameters
        ----------
        model_name

        Returns
        -------

        """
        # load estimation settings file
        if not self.settings_initialized:
            self.initialize_settings()

        # global estimation setting
        if not self.enabled:
            return None

        bundle_name = bundle_name or model_name

        if bundle_name not in self.bundles:
            logger.warning(
                f"estimation bundle {bundle_name} not in settings file {ESTIMATION_SETTINGS_FILE_NAME}"
            )
            return None

        # can't estimate the same model simultaneously
        assert (
            model_name not in self.estimating
        ), "Cant begin estimating %s - already estimating that model." % (model_name,)

        assert (
            bundle_name in self.model_estimation_table_types
        ), "No estimation_table_type for %s in %s." % (
            bundle_name,
            ESTIMATION_SETTINGS_FILE_NAME,
        )

        model_estimation_table_type = self.model_estimation_table_types[bundle_name]

        assert (
            model_estimation_table_type in self.estimation_table_recipes
        ), "model_estimation_table_type '%s' for model %s no in %s." % (
            model_estimation_table_type,
            model_name,
            ESTIMATION_SETTINGS_FILE_NAME,
        )

        self.estimating[model_name] = Estimator(
            bundle_name,
            model_name,
            estimation_table_recipes=self.estimation_table_recipes[
                model_estimation_table_type
            ],
        )

        return self.estimating[model_name]

    def release(self, estimator):

        self.estimating.pop(estimator.model_name)

    def get_survey_table(self, table_name):
        assert self.enabled
        if table_name not in self.survey_tables:
            logger.warning(
                "EstimationManager. get_survey_table: survey table '%s' not in survey_tables"
                % table_name
            )
        df = self.survey_tables[table_name].get("df")
        return df

    def get_survey_values(self, model_values, table_name, column_names):

        assert isinstance(
            model_values, (pd.Series, pd.DataFrame, pd.Index)
        ), "get_survey_values model_values has unrecognized type %s" % type(
            model_values
        )

        dest_index = (
            model_values if isinstance(model_values, (pd.Index)) else model_values.index
        )

        # read override_df table
        survey_df = manager.get_survey_table(table_name)

        assert survey_df is not None, "get_survey_values: table '%s' not found" % (
            table_name,
        )

        column_name = column_names if isinstance(column_names, str) else None
        if column_name:
            column_names = [column_name]

        if not set(column_names).issubset(set(survey_df.columns)):
            missing_columns = list(set(column_names) - set(survey_df.columns))
            logger.error(
                "missing columns (%s) in survey table %s"
                % (missing_columns, table_name)
            )
            print("survey table columns: %s" % (survey_df.columns,))
            raise RuntimeError(
                "missing columns (%s) in survey table %s"
                % (missing_columns, table_name)
            )

        assert set(column_names).issubset(set(survey_df.columns)), (
            f"missing columns ({list(set(column_names) - set(survey_df.columns))}) "
            f"in survey table {table_name} {list(survey_df.columns)}"
        )

        # for now tour_id is asim_tour_id in survey_df
        asim_df_index_name = dest_index.name
        if asim_df_index_name == survey_df.index.name:
            # survey table has same index as activitysim
            survey_df_index_column = "index"
        elif asim_df_index_name in survey_df.columns:
            # survey table has activitysim index as column
            survey_df_index_column = asim_df_index_name
        elif "asim_%s" % asim_df_index_name in survey_df.columns:
            # survey table has activitysim index as column with asim_ prefix
            survey_df_index_column = "asim_%s" % asim_df_index_name
        else:
            logger.error(
                "get_survey_values:index '%s' not in survey table" % dest_index.name
            )
            # raise RuntimeError("index '%s' not in survey table %s" % (dest_index.name, table_name)
            survey_df_index_column = None

        logger.debug(
            "get_survey_values: reindexing using %s.%s"
            % (table_name, survey_df_index_column)
        )

        values = pd.DataFrame(index=dest_index)
        for c in column_names:
            if survey_df_index_column == "index":
                survey_values = survey_df[c]
            else:
                survey_values = pd.Series(
                    survey_df[c].values, index=survey_df[survey_df_index_column]
                )

            survey_values = reindex(survey_values, dest_index)

            # shouldn't be any choices we can't override
            missing_values = survey_values.isna()
            if missing_values.any():
                logger.error(
                    "missing survey_values for %s\n%s" % (c, dest_index[missing_values])
                )
                logger.error(
                    "couldn't get_survey_values for %s in %s\n" % (c, table_name)
                )
                raise RuntimeError(
                    "couldn't get_survey_values for %s in %s\n" % (c, table_name)
                )

            values[c] = survey_values

        return values[column_name] if column_name else values


manager = EstimationManager()
