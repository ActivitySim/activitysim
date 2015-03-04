import urbansim.sim.simulation as sim


@sim.table(cache=True)
def land_use(store):
    return store["land_use/taz_data"]


sim.broadcast('land_use', 'households', cast_index=True, onto_on='TAZ')


@sim.column("land_use")
def total_households(land_use):
    return land_use.local.TOTHH


@sim.column("land_use")
def total_employment(land_use):
    return land_use.local.TOTEMP


@sim.column("land_use")
def total_acres(land_use):
    return land_use.local.TOTACRE


@sim.column("land_use")
def county_id(land_use):
    return land_use.local.COUNTY


@sim.column("land_use")
def household_density(land_use):
    return land_use.total_households / land_use.total_acres


@sim.column("land_use")
def employment_density(land_use):
    return land_use.total_employment / land_use.total_acres


@sim.column("land_use")
def density_index(land_use):
    return (land_use.household_density * land_use.employment_density) / \
        (land_use.household_density + land_use.employment_density)


@sim.column("land_use")
def county_name(land_use, settings):
    assert "county_map" in settings
    inv_map = {v: k for k, v in settings["county_map"].items()}
    return land_use.county_id.map(inv_map)
