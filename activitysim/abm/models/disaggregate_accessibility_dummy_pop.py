import pandas as pd
import yaml
import collections, os
from itertools import product
from activitysim.core import expressions


def ordered_load(stream, Loader=yaml.SafeLoader, object_pairs_hook=collections.OrderedDict):
    class OrderedLoader(Loader):
        pass
    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)


def read_table_settings(model_settings):
    # Check if setup properly
    assert 'CREATE_TABLES' in model_settings.keys()
    assert all([True for k, v in model_settings['CREATE_TABLES'].items() if 'VARIABLES' in v.keys()])
    assert all([True for x in ['PERSONS', 'HOUSEHOLDS', 'TOURS', 'BLAH'] if x in model_settings['CREATE_TABLES'].keys()])

    table_params = {}
    vary_on = {}
    for name, table in model_settings['CREATE_TABLES'].items():
        table_params[name.lower()] = table['VARIABLES']
        vary_on[name.lower()] = table.get('VARY_ON', [])

    join_on = dict(model_settings['CREATE_TABLES']['TOURS']['JOIN_ON'])

    return table_params, vary_on, join_on


def named_product(**d):
    names = d.keys()
    vals = d.values()
    for res in product(*vals):
        yield dict(zip(names, res))


def generate_replicates(table_name, vars, varying):
    static_vars = {k: v for k, v in vars.items() if k not in varying}
    vary_vars = {k: v for k, v in vars.items() if k in varying}
    static_df = pd.DataFrame(static_vars)

    # Create replicated table based on varying columns, using the static dataframe index as a varying column
    rep = pd.DataFrame(named_product(index=static_df.index, **vary_vars)).set_index('index')

    # Join the replicated data back to the data frame
    return rep.join(static_df).reset_index(drop=True)


def create_dummy_pop(land_use_df, model_settings):
    # Get table settings
    table_params, vary_on, tours_join_on = read_table_settings(model_settings)

    # Add in the zone variables
    # TODO need to implement the crosswalk join downstream for 3 zone
    zones = {'MAZ': land_use_df.MAZ.tolist()} if model_settings['zone_system'] == 3 else {'TAZ': land_use_df.TAZ.tolist()}

    # Add zones to households dicts as vary_on variable
    table_params['households'] = {**table_params['households'], **zones}
    vary_on['households'].extend(zones.keys())

    # Separate out the mapped data from the varying data and create base replicate tables
    replicated = {k: generate_replicates(k, td, vary_on[k]) for k, td in table_params.items()}

    # Create hhid
    replicated['households']['hhid'] = replicated['households'].index + 1
    replicated['households']['household_serial_no'] = replicated['households']['hhid']

    # Assign persons to households
    rep = pd.DataFrame(named_product(hhid=replicated['households'].hhid,
                                     index=replicated['persons'].index)
                       ).set_index('index')
    replicated['persons'] = rep.join(replicated['persons']).sort_values('hhid').reset_index(drop=True)
    replicated['persons']['perid'] = replicated['persons'].index + 1

    # Assign persons to tours
    tkey, pkey = list(tours_join_on.items())[0]
    replicated['tours'] = replicated['tours'].merge(replicated['persons'][['pnum','hhid','perid']],
                                                    left_on=tkey, right_on=pkey)
    replicated['tours'].index = replicated['tours'].index.set_names(['tourid'])
    replicated['tours'] = replicated['tours'].reset_index().drop(columns=[pkey])

    # Output
    return replicated


def create_tours(self):

    # Extract previously created households_df and person_df
    persons_df = self.output['persons']
    households_df = self.output['households']

    # create individual tours
    indivTours_df = pd.DataFrame.from_dict(self.model_settings['base_tables']['indivTours'])

    # duplicate for all persons
    perid_list = sorted(persons_df["PERID"].tolist())
    indivTours_df = self.replicate_df_for_variable(indivTours_df, "PERID", perid_list)

    # merge person file
    indivTours_df = pd.merge(left=indivTours_df, right=persons_df, on="PERID", how="outer").drop(
        columns=["person_num_y"])
    indivTours_df = indivTours_df.rename(columns={"HHID_x": "HHID", "person_num_x": "person_num"})

    # keep mandatory tours for FT workers and non-mandatory tours for PT workers
    indivTours_df = indivTours_df[
        ((indivTours_df.pemploy == 1) & (indivTours_df.tour_category == "MANDATORY")) |
        ((indivTours_df.pemploy == 2) & (indivTours_df.tour_category == "INDIVIDUAL_NON_MANDATORY"))]

    # merge households file so that we can set origin zone
    indivTours_df = pd.merge(left=indivTours_df, right=households_df, on="HHID", how="outer")
    #print(indivTours_df.columns)
    indivTours_df["orig_taz"] = indivTours_df["TAZ"]
    indivTours_df["orig_walk_segment"] = 0
    indivTours_df["avAvailable"] = indivTours_df["AV_AVAIL"]

    # drop households and person variable fields
    indivTours_df = indivTours_df.drop(
        columns=["join_key", "AGE", "SEX", "pemploy", "pstudent", "TAZ", "HINC", "hworkers", "PERSONS", "HHT",
                 "hinccat1"])
    indivTours_df = indivTours_df.rename(
        columns={"HHID": "hh_id", "ptype": "person_type", "PERID": "person_id"})
    indivTours_df = indivTours_df.sort_values(by=["person_id"])

    # reorder columns
    self.indivTours_df = indivTours_df[
        ["hh_id", "person_id", "person_num", "person_type", "tour_id", "tour_category",
         "tour_purpose", "dest_taz", "dest_walk_segment", "start_hour", "end_hour", "tour_mode", "atWork_freq",
         "num_ob_stops", "num_ib_stops",
         "avAvailable"]]

    self.output.update({'indivTours': indivTours_df})

def save_output(self):
    for name, model in self.output.items():
        outfile = "{}_{}.csv".format(self.OUTPUT_PREFIX, name)
        model.to_csv(outfile, index=False)
        print("Wrote {} lines to {}".format(len(model), outfile))


if __name__ == "__main__":
    config_dir = 'C:/gitclones/activitysim-SANDAG/activitysim/examples/example_mtc_extended/configs/disaggregate_accessibility.yaml'
    data_dir = 'C:/gitclones/activitysim-SANDAG/activitysim/examples/example_mtc/data'

    # Model Settings
    with open(config_dir, 'r') as file:
        model_settings = ordered_load(file)

    # Setup input parameters for testing
    land_use_df = pd.read_csv(os.path.join(data_dir, 'land_use.csv'))

    replications = create_dummy_pop(land_use_df, model_settings)
    #





