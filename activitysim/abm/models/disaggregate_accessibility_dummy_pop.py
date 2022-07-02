import pandas as pd
import yaml
import json
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
    assert all([True for x in ['PERSONS', 'HOUSEHOLDS', 'TOURS'] if x in model_settings['CREATE_TABLES'].keys()])

    table_params = {}
    mapped_fields = {}
    for name, table in model_settings['CREATE_TABLES'].items():
        # Ensure table variables are all lists
        table_params[name.lower()] = {k: (v if isinstance(v, list) else [v]) for k, v in table['VARIABLES'].items()}
        mapped_fields[name.lower()] = table.get('MAPPED_FIELDS', [])

    join_on = dict(model_settings['CREATE_TABLES']['TOURS']['JOIN_ON'])

    return table_params, mapped_fields, join_on


def named_product(**d):
    names = d.keys()
    vals = d.values()
    for res in product(*vals):
        yield dict(zip(names, res))


def generate_replicates(vars, mapped):
    """
    Generates replicates finding the cartesian product of the non-mapped field variables.
    The mapped fields are then annotated after replication
    """

    # Generate replicates
    rep = pd.DataFrame(named_product(**vars))

    # Applying mapped variables
    if len(mapped) > 0:
        for mapped_from, mapped_to_pair in mapped.items():
            name, mapped_to = list(mapped_to_pair.items())[0]
            rep[name] = rep[mapped_from].map(mapped_to)

    return rep


def create_dummy_pop(land_use_df, model_settings):
    # Get table settings
    table_params, mapped_fields, tours_join_on = read_table_settings(model_settings)

    # Add in the zone variables
    # TODO need to implement the crosswalk join downstream for 3 zone
    zones = {'MAZ': land_use_df.MAZ.tolist()} if model_settings['zone_system'] == 3 else {'TAZ': land_use_df.TAZ.tolist()}

    # Add zones to households dicts as vary_on variable
    table_params['households'] = {**table_params['households'], **zones}
    # vary_on['households'].extend(zones.keys())
    # json.loads(json.dumps(mapped_fields))

    # Separate out the mapped data from the varying data and create base replicate tables
    # replicated = {k: generate_replicates(k, td, vary_on[k]) for k, td in table_params.items()}
    # replicated = {k: pd.DataFrame(named_product(**td)) for k, td in table_params.items()}
    replicated = {k: generate_replicates(td, mapped_fields[k]) for k, td in table_params.items()}

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


def save_output(self):
    for name, model in self.output.items():
        outfile = "{}_{}.csv".format(self.OUTPUT_PREFIX, name)
        model.to_csv(outfile, index=False)
        print("Wrote {} lines to {}".format(len(model), outfile))


if __name__ == "__main__":
    config_dir = 'C:/gitclones/activitysim-disagg_accessibilities/activitysim/examples/example_mtc_extended/configs/disaggregate_accessibility.yaml'
    data_dir = 'C:/gitclones/activitysim-disagg_accessibilities/activitysim/examples/example_mtc/data'

    # Model Settings
    with open(config_dir, 'r') as file:
        #model_settings = ordered_load(file)
        model_settings = yaml.load(file, Loader=yaml.SafeLoader)

    # Setup input parameters for testing
    land_use_df = pd.read_csv(os.path.join(data_dir, 'land_use.csv'))

    replications = create_dummy_pop(land_use_df, model_settings)
    #





