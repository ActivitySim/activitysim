"""
Workplace location calibration helper functions.

Notes:
    The lru_cache decorator to ensure that survey data is loaded and computed only once per Python process.
    Context is developed in _build_expression_context() in activitysim.core.calibration.
"""

import matplotlib.pyplot as plt
import pandas as pd
import os
from functools import lru_cache

SURVEY_DATA_FOLDER = r"C:\Users\david.hensle\OneDrive - Resource Systems Group, Inc\Documents\projects\activitysim\rsg_activitysim\activitysim\examples\example_estimation\data_test\survey_data"

def compute_distances(context, origins, destinations):
    # Compute distances between origins and destinations using the network level of service
    # using non-time-dependent DIST skim
    distances = context["skim_dict"].lookup(origins.clip(upper=24), destinations.clip(upper=24), 'DIST')
    # time dependent example
    # distances = skim_dict.lookup_3d(origins, destinations, 'AM', 'SOV_DIST')
    return distances


# @lru_cache(maxsize=1)
def _survey_persons() -> pd.DataFrame:
    """Load survey persons once per Python process."""
    return pd.read_csv(os.path.join(SURVEY_DATA_FOLDER, "override_persons.csv"))


# @lru_cache(maxsize=1)
def _survey_households() -> pd.DataFrame:
    """Load survey households once per Python process."""
    return pd.read_csv(os.path.join(SURVEY_DATA_FOLDER, "override_households.csv"))


# @lru_cache(maxsize=1)
def _survey_worker_distances(context):
    """Compute survey worker distances once and reuse across calibration rows."""
    survey_persons = _survey_persons()
    survey_workers = survey_persons[survey_persons["workplace_zone_id"] > 0]
    survey_home_zone_ids = _survey_households().set_index("household_id")["home_zone_id"]
    survey_home_zone_ids = survey_workers["household_id"].map(survey_home_zone_ids)
    survey_workplace_zone_ids = survey_workers["workplace_zone_id"]
    return compute_distances(context, survey_home_zone_ids, survey_workplace_zone_ids)


def summarize_model(context, min_dist=1, max_dist=2):
    """Summarize the model results for workplaces within the specified distance range."""
    persons = context['persons']
    workers = persons[persons['workplace_zone_id'] > 0]
    home_zone_ids = workers['home_zone_id']
    workplace_zone_ids = workers['workplace_zone_id']
    
    distances = compute_distances(context, home_zone_ids, workplace_zone_ids)

    # Filter distances within the specified range
    mask = (distances >= min_dist) & (distances < max_dist)
    filtered_distances = distances[mask]

    share = len(filtered_distances) / len(distances) if len(distances) > 0 else 0
    return share

def summarize_survey(context, min_dist=1, max_dist=2):
    """Summarize the survey results for workplaces within the specified distance range."""

    distances = _survey_worker_distances(context)

    # Filter distances within the specified range
    mask = (distances >= min_dist) & (distances < max_dist)
    filtered_distances = distances[mask]

    share = len(filtered_distances) / len(distances) if len(distances) > 0 else 0
    return share

def report_workplace_location(context):
    """Workplace location distance frequency plot comparing model results with observed data."""
    print("summarizing workplace location model")
    model_persons = context["persons"]
    model_workers = model_persons[model_persons['workplace_zone_id'] > 0]
    model_home_zone_ids = model_workers['home_zone_id']
    model_workplace_zone_ids = model_workers['workplace_zone_id']

    model_distances = compute_distances(context, model_home_zone_ids, model_workplace_zone_ids)

    survey_distances = _survey_worker_distances(context)
    
    # Here you can add code to compare model_distances and survey_distances, 
    # for example by plotting histograms or computing summary statistics.
    plt.hist(model_distances, bins=20, density=True, alpha=0.5, label='Model')
    plt.hist(survey_distances, bins=20, density=True, alpha=0.5, label='Survey')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.legend()
    # component_output_dir set in the evaluation context
    plt.savefig(os.path.join(context["component_output_dir"], 'workplace_location_comparison.png'))
    plt.close()