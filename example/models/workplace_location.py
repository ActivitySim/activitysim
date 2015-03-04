"""
The workplace location model predicts the zones in which various people will
work.  Interestingly there's not really any supply side to this model - we
assume there are workplaces for the people to work.
"""


@sim.injectable()
def workplace_location_spec():
    f = os.path.join('configs', "workplace_location.csv")
    # FIXME should read in all variables and comment out ones not used
    return asim.read_model_spec(f).head(15)


@sim.table()
def workplace_size_terms(land_use, destination_choice_size_terms):
    """
    This method takes the land use data and multiplies various columns of the
    land use data by coefficients from the workplace_size_spec table in order
    to yield a size term (a linear combination of land use variables) with
    specified coefficients for different segments (like low, med, and high
    income)
    """
    land_use = land_use.to_frame()
    df = destination_choice_size_terms.to_frame().query("purpose == 'work'")
    df = df.drop("purpose", axis=1).set_index("segment")
    new_df = {}
    for index, row in df.iterrows():
        missing = row[~row.index.isin(land_use.columns)]
        if len(missing) > 0:
            print "WARNING: missing columns in land use\n", missing.index
        row = row[row.index.isin(land_use.columns)]
        sparse = land_use[list(row.index)]
        new_df["size_"+index] = np.dot(sparse.as_matrix(), row.values)
    new_df = pd.DataFrame(new_df, index=land_use.index)
    return new_df


# FIXME there are three school models that go along with this one which have
# FIXME not been implemented yet
@sim.model()
def workplace_location_simulate(persons,
                                households,
                                zones,
                                workplace_location_spec,
                                distance_skim,
                                workplace_size_terms):

    choosers = sim.merge_tables(persons.name, tables=[persons, households])
    alternatives = zones.to_frame().join(workplace_size_terms.to_frame())

    skims = {
        "distance": distance_skim
    }

    choices, model_design = \
        asim.simple_simulate(choosers,
                             alternatives,
                             workplace_location_spec,
                             skims,
                             skim_join_name="TAZ",
                             mult_by_alt_col=False,
                             sample_size=50)

    print "Describe of choices:\n", choices.describe()
    sim.add_column("persons", "workplace_taz", choices)

    return model_design
