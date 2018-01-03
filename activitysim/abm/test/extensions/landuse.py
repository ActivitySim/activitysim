import numpy as np
import pandas as pd

from activitysim.core import inject


@inject.column("land_use")
def total_households(land_use):
    return land_use.local.TOTHH


@inject.column("land_use")
def total_employment(land_use):
    return land_use.local.TOTEMP


@inject.column("land_use")
def total_acres(land_use):
    return land_use.local.TOTACRE


@inject.column("land_use")
def county_id(land_use):
    return land_use.local.COUNTY


@inject.column("land_use")
def household_density(land_use):
    return land_use.total_households / land_use.total_acres


@inject.column("land_use")
def employment_density(land_use):
    return land_use.total_employment / land_use.total_acres


@inject.column("land_use")
def density_index(land_use):
    # FIXME - avoid div by 0
    return (land_use.household_density * land_use.employment_density) / \
        (land_use.household_density + land_use.employment_density).clip(lower=1)


@inject.column("land_use")
def county_name(land_use, settings):
    assert "county_map" in settings
    inv_map = {v: k for k, v in settings["county_map"].items()}
    return land_use.county_id.map(inv_map)
