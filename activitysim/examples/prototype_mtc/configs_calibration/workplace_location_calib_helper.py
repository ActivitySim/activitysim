import matplotlib.pyplot as plt
import pandas as pd
import os
from functools import lru_cache

SURVEY_DATA_FOLDER = r"U:\projects\clients\ASIM\autocalibration\calibration_test_mtc\data"

def compute_distances(origins, destinations):
    # Compute distances between origins and destinations using the network level of service
    # using non-time-dependent DIST skim
    distances = skim_dict.lookup(origins, destinations, 'DIST')
    # time dependent example
    # distances = skim_dict.lookup_3d(origins, destinations, 'AM', 'SOV_DIST')
    return distances


@lru_cache(maxsize=1)
def _survey_persons() -> pd.DataFrame:
    """Load survey persons once per Python process."""
    return pd.read_csv(os.path.join(SURVEY_DATA_FOLDER, "override_persons.csv"))


@lru_cache(maxsize=1)
def _survey_households() -> pd.DataFrame:
    """Load survey households once per Python process."""
    return pd.read_csv(os.path.join(SURVEY_DATA_FOLDER, "override_households.csv"))


@lru_cache(maxsize=1)
def _survey_worker_distances():
    """Compute survey worker distances once and reuse across calibration rows."""
    survey_persons = _survey_persons()
    survey_workers = survey_persons[survey_persons["workplace_zone_id"] > 0]
    survey_home_zone_ids = _survey_households().set_index("household_id")["home_zone_id"]
    survey_home_zone_ids = survey_workers["household_id"].map(survey_home_zone_ids)
    survey_workplace_zone_ids = survey_workers["workplace_zone_id"]
    return compute_distances(survey_home_zone_ids, survey_workplace_zone_ids)


def summarize_model(min_dist=1, max_dist=2):
    """Summarize the model results for workplaces within the specified distance range."""
    workers = persons[persons['workplace_zone_id'] > 0]
    home_zone_ids = workers['home_zone_id']
    workplace_zone_ids = workers['workplace_zone_id']
    
    distances = compute_distances(home_zone_ids, workplace_zone_ids)

    # Filter distances within the specified range
    mask = (distances >= min_dist) & (distances < max_dist)
    filtered_distances = distances[mask]

    share = len(filtered_distances) / len(distances) if len(distances) > 0 else 0
    return share

def summarize_survey(min_dist=1, max_dist=2):
    """Summarize the survey results for workplaces within the specified distance range."""

    distances = _survey_worker_distances()

    # Filter distances within the specified range
    mask = (distances >= min_dist) & (distances < max_dist)
    filtered_distances = distances[mask]

    share = len(filtered_distances) / len(distances) if len(distances) > 0 else 0
    return share

def report_workplace_location():
    """Workplace location distance frequency plot comparing model results with observed data."""
    print("summarizing workplace location model")
    model_persons = persons
    model_workers = model_persons[model_persons['workplace_zone_id'] > 0]
    model_home_zone_ids = model_workers['home_zone_id']
    model_workplace_zone_ids = model_workers['workplace_zone_id']

    model_distances = compute_distances(model_home_zone_ids, model_workplace_zone_ids)

    # survey_distances = _survey_worker_distances()
    survey_distances = model_distances
    
    # Here you can add code to compare model_distances and survey_distances, 
    # for example by plotting histograms or computing summary statistics.
    plt.hist(model_distances, bins=20, density=True, alpha=0.5, label='Model')
    plt.hist(survey_distances, bins=20, density=True, alpha=0.5, label='Survey')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.legend()
    # component_output_dir set in the evaluation context
    plt.savefig(os.path.join(component_output_dir, 'workplace_location_comparison.png'))
    plt.close()